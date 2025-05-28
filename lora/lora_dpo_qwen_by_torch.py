import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
# from datasets import Dataset
import logging
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self,
                 original_layer: nn.Linear,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0
                 ):
        super().__init__()
        # 冻结原始权重
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA参数
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 获取原始层的参数
        if isinstance(original_layer, nn.Linear):  # 未量化
            in_features = original_layer.in_features
            out_features = original_layer.out_features
            self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))  # LoRA的A矩阵 (向下投影)
            self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))  # LoRA的B矩阵 (向上投影)
        elif isinstance(original_layer, QuantLinear):  # gptq 量化
            in_features = original_layer.infeatures
            out_features = original_layer.outfeatures
            self.lora_A = nn.Parameter(torch.zeros((rank, in_features), dtype=torch.half))
            self.lora_B = nn.Parameter(torch.zeros((out_features, rank), dtype=torch.half))

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        """
        前向传播过程
        """
        # 原始层的前向传播
        original_output = self.original_layer(x)

        # LoRA的前向传播: B(Ax)
        x = self.dropout(x)
        x = F.linear(x, self.lora_A)  # x:(1,7,896)   lora_A:(8,896)   lora_B:(896,8)
        lora_output = F.linear(x, self.lora_B)

        # 合并结果
        return original_output + self.scaling * lora_output

    def merge_weights(self):
        """
        合并LoRA权重到基础层中，用于推理
        """
        merged_weight = self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling
        return merged_weight


class DPODataset(Dataset):
    """
    偏好数据集类，包含prompt、chosen和rejected响应
    """

    def __init__(self, data, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []
        self.chosen_responses = []
        self.rejected_responses = []

        for item in data:
            self.prompts.append(item['conversations'])
            self.chosen_responses.append(item['chosen'])
            self.rejected_responses.append(item['rejected'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]

        chosen_encodings = self.tokenizer(
            prompt[0]["value"] + chosen["value"],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        rejected_encodings = self.tokenizer(
            prompt[0]["value"] + rejected["value"],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # ['longest', 'max_length', 'do_not_pad']
            return_tensors='pt'  # “tf”、“pt”、“np”（返回 tf.constant 对象, torch.Tensor 对象, np.ndarray 对象
        )

        return {
            'input_ids_chosen': chosen_encodings['input_ids'].squeeze(),
            'attention_mask_chosen': chosen_encodings['attention_mask'].squeeze(),
            'input_ids_rejected': rejected_encodings['input_ids'].squeeze(),
            'attention_mask_rejected': rejected_encodings['attention_mask'].squeeze()
        }


class DPOLoss(nn.Module):
    # 2. 实现DPO损失函数
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta  # 控制KL散度正则化的强度

    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps):
        """
        计算DPO损失
        参数:
            policy_chosen_logps: 当前策略模型对chosen响应的对数概率
            policy_rejected_logps: 当前策略模型对rejected响应的对数概率
            reference_chosen_logps: 参考模型(通常为初始模型)对chosen响应的对数概率
            reference_rejected_logps: 参考模型对rejected响应的对数概率
        """
        # 计算策略和参考模型之间的对数概率比
        log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        log_ratio_rejected = policy_rejected_logps - reference_rejected_logps

        # 计算DPO损失
        losses = -torch.log(
            torch.sigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected))
        )

        return losses.mean()


def apply_lora(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=16):
    """
    将LoRA应用于模型中的特定模块
    """
    for name, module in model.named_modules():
        # 替换原始线性层为LoRALayer
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                path = name.split(".")
                for p in path[:-1]:
                    parent = getattr(parent, p)      # 等于 parent = parent.p
                setattr(parent, path[-1], LoRALayer(module, rank, alpha))  # 等于 parent.path[-1] = LoRALayer(module, rank, alpha)
            elif isinstance(module, QuantLinear):
                parent = model
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], LoRALayer(module, rank, alpha))
    return model


def prepare_model_and_tokenizer(model_name="Qwen/Qwen-1_8B"):
    """
    准备模型和tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 应用LoRA
    model = apply_lora(model)

    return model, tokenizer


def train_epoch(model, reference_model, train_loader, optimizer, scheduler, device, epochs=3, accumulation_steps=4):
    model.train()
    reference_model.train()
    loss_fn = DPOLoss(beta=0.1)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移动到设备
            input_ids_chosen = batch['input_ids_chosen'].to(device)
            attention_mask_chosen = batch['attention_mask_chosen'].to(device)
            input_ids_rejected = batch['input_ids_rejected'].to(device)
            attention_mask_rejected = batch['attention_mask_rejected'].to(device)

            # 获取策略模型的log概率
            policy_chosen_outputs = model(
                input_ids=input_ids_chosen,
                attention_mask=attention_mask_chosen,
                labels=input_ids_chosen
            )
            policy_chosen_logps = policy_chosen_outputs.logits.log_softmax(dim=-1)

            policy_rejected_outputs = model(
                input_ids=input_ids_rejected,
                attention_mask=attention_mask_rejected,
                labels=input_ids_rejected
            )
            policy_rejected_logps = policy_rejected_outputs.logits.log_softmax(dim=-1)

            # 获取参考模型的log概率(不计算梯度)
            with torch.no_grad():
                ref_chosen_outputs = reference_model(
                    input_ids=input_ids_chosen,
                    attention_mask=attention_mask_chosen,
                    labels=input_ids_chosen
                )
                ref_chosen_logps = ref_chosen_outputs.logits.log_softmax(dim=-1)

                ref_rejected_outputs = reference_model(
                    input_ids=input_ids_rejected,
                    attention_mask=attention_mask_rejected,
                    labels=input_ids_rejected
                )
                ref_rejected_logps = ref_rejected_outputs.logits.log_softmax(dim=-1)

            # 计算损失
            loss = loss_fn(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )
            print(f"step {batch_idx + 1} loss: {loss}")

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}')

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch + 1} completed. Average Loss: {avg_loss}')


def main():
    # 初始化模型和tokenizer
    model_name = "F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct-GPTQ-Int4"  # 根据实际使用的Qwen模型调整
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

    # 加载模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    reference_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    # 冻结参考模型参数
    for param in reference_model.parameters():
        param.requires_grad = False

    # 假设我们有训练数据
    # 这里应该是你的实际偏好数据集
    # train_data = [
    #     {
    #         "prompt": "解释一下量子计算的基本原理",
    #         "chosen": "量子计算利用量子比特的叠加和纠缠特性...",  # 高质量回答
    #         "rejected": "量子计算就是比经典计算机快的计算机..."  # 低质量回答
    #     },
    #     # 更多数据...
    # ]

    with open("F:\inspur\GPU\code\LLM-FineTuning\data\dpo_zh_demo.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    # print(train_data)

    # 创建数据集和数据加载器
    train_dataset = DPODataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=5e-6)
    total_steps = len(train_loader) * 3  # epochs=3
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    # 开始训练
    train_epoch(
        model=model,
        reference_model=reference_model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=3
    )

    # 保存微调后的模型
    model.save_pretrained("./qwen_dpo_finetuned")
    tokenizer.save_pretrained("./qwen_dpo_finetuned")


if __name__ == "__main__":
    main()
