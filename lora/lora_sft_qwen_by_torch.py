import json
import math
import argparse
from tqdm import tqdm
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear  # gptq
from lora.utils import CustomDatasetSFT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


class LoRALayer(nn.Module):
    def __init__(self,
                 original_layer: nn.Linear,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.original_layer = original_layer  # 原始线性层（冻结）

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


def apply_lora(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], rank=8, alpha=16):
    """
    将LoRA应用于模型中的特定模块
    """
    for name, module in model.named_modules():
        # 替换原始线性层为LoRALayer
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], LoRALayer(module, rank, alpha))
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # 应用LoRA
    model = apply_lora(model)

    return model, tokenizer


# 训练数据预处理方法
def preprocess(tokenizer, batch_messages):
    input_list = []
    target_list = []

    im_start = tokenizer('<|im_start|>').input_ids
    im_end = tokenizer('<|im_end|>').input_ids
    newline = tokenizer('\n').input_ids
    pad = tokenizer('<|endoftext|>').input_ids            # 给attention mask使用
    ignore = [-100]

    for group in batch_messages:
        input_ids = []
        target_ids = []
        for msg in group:
            role = tokenizer(msg['role']).input_ids
            content = tokenizer(msg['content']).input_ids
            # if msg['role'] in ['system', 'user']:
            #     ignore_parts = role + newline + content
            #     input_ids += im_start + ignore_parts + im_end + newline
            #     target_ids += im_start + ignore * len(ignore_parts) + im_end + newline
            # else:
            #     ignore_parts = role + newline
            #     input_ids += im_start + ignore_parts + content + im_end + newline
            #     target_ids += im_start + ignore * len(ignore_parts) + content + im_end + newline

            if msg['role'] in ['system', 'user']:
                ignore_parts = im_start + role + newline + content + im_end + newline
                input_ids += ignore_parts
                target_ids += ignore * len(ignore_parts)
            else:
                ignore_parts = im_start + role + newline
                input_ids += ignore_parts + content + im_end + newline
                target_ids += ignore * len(ignore_parts) + content + im_end + newline
        input_list.append(input_ids)
        target_list.append(target_ids)

    # padding
    max_len = max([len(ids) for ids in input_list])
    for input_ids, target_ids in zip(input_list, target_list):
        input_ids += pad * (max_len - len(input_ids))
        target_ids += ignore * (max_len - len(target_ids))

    batch_input_ids = torch.tensor(input_list, dtype=torch.long)
    batch_target_ids = torch.tensor(target_list, dtype=torch.long)
    batch_attention_mask = batch_input_ids.ne(pad[0]).type(torch.long)
    return batch_input_ids, batch_target_ids, batch_attention_mask


def train_epoch(model,
                train_loader,
                batch_input_ids,
                batch_target_ids,
                batch_mask,
                loss_fn,
                optimizer, device, accumulation_steps=4):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_loader)):
        # input_ids = batch["input_ids"].to(device)
        # attention_mask = batch["attention_mask"].to(device)
        # labels = batch["labels"].to(device)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target_ids"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        outputs_logps = outputs.logits.log_softmax(dim=-1)
        loss = outputs.loss
        loss = loss / accumulation_steps
        print(f"step {1 + 1} loss: {loss}")

        loss.backward()
        # 梯度累积
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)


def prepare_dataloader(tokenizer, dataset, batch_size=4):
    """
    准备数据加载器
    """

    def collate_fn(batch):
        input_texts, target_texts = [], []
        for item in batch:
            prompt = '''f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"'''
            prompt = prompt.replace("{input}", item["instruction"] + item["input"])
            input_texts.append(prompt)
            output = prompt + item["output"]
            target_texts.append(output)

        # 对于SFT，我们只需要计算response部分的loss
        input_encodings = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # 首先找到 input 的长度`
        input_len = input_encodings["input_ids"].shape[1]

        target_encodings = tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # 创建attention mask
        target_ids = target_encodings["input_ids"].squeeze()
        attention_mask = target_encodings["attention_mask"].squeeze()

        # 创建labels，将 input 部分设置为-100（在计算loss时忽略）
        labels = target_ids.clone()
        for label in labels:
            label[:input_len] = -100

        samples = {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        return samples

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def main():
    parser = argparse.ArgumentParser()
    # "model_name": r"F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct",
    parser.add_argument("--model_name", type=str,
                        default=r"F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct-GPTQ-Int4", help="预训练模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--dataset", type=str, default=r"F:\inspur\GPU\code\LLM-FineTuning\data\alpaca_zh_demo.json",
                        help="")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="")

    args = parser.parse_args()

    # 配置参数
    config = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "device": args.device,
    }

    # 准备模型和tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config["model_name"])
    model.to(config["device"])

    # 准备数据集
    # 方式1：load_dataset
    from datasets import load_dataset
    if isinstance(args.dataset, str):
        sft_dataset = load_dataset(
            path="json",
            name=None,
            data_dir=None,
            data_files=['../data/alpaca_zh_demo.json'],
            split='train',
            cache_dir=None,
            token=None,
            num_proc=1,
            trust_remote_code=True,
            streaming=False,
        )
        split_dataset = sft_dataset.train_test_split(test_size=0.3, seed=1)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        # 对数据预处理
        # 可参考 llamafactory/data/processor/supervised.py
        format_system = '<|im_start|>system\n{{content}}<|im_end|>\n'
        format_user = '<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n'
        format_assistant = '{{content}}<|im_end|>\n'
        pass

        # 示例数据集 - 替换为你的实际数据
        messages = []
        for data in test_dataset:
            tmp = []
            tmp.append({"role": "system", "content": "You are a helpful assistant."})
            tmp.append({"role": "user", "content": data["instruction"] + data["input"]})
            tmp.append({"role": "assistant", "content": data["output"]})
            messages.append(tmp)
        # print(messages)
        batch_input_ids, batch_target_ids, batch_mask = preprocess(tokenizer, messages)

    # 方式2：自定义dataset
    sft_dataset = CustomDatasetSFT(tokenizer, data="../data/alpaca_zh_demo.json", max_length=256)




    # 准备数据加载器
    train_dataset = CustomDatasetSFT(tokenizer=tokenizer, data=args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    # train_loader = prepare_dataloader(tokenizer, dataset, config["batch_size"])

    # 损失函数
    loss_fn = CrossEntropyLoss()

    # 准备优化器 - 只训练LoRA参数
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"]
    )

    # 训练循环
    for epoch in range(config["epochs"]):
        avg_loss = train_epoch(
            model,
            train_loader,
            batch_input_ids,
            batch_target_ids,
            batch_mask,
            loss_fn,
            optimizer,
            config["device"],
            config["accumulation_steps"]
        )
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 保存LoRA权重
    torch.save(
        {"lora_params": [p for p in model.named_parameters() if "lora_" in p[0]]},
        "lora_weights.pth"
    )


if __name__ == "__main__":
    main()
