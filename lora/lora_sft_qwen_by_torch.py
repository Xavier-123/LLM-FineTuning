import json
import math
from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from typing import List
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import argparse


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
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA的A矩阵 (向下投影)
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        # LoRA的B矩阵 (向上投影)
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))

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
    named_modules_list = model.named_modules()
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换原始线性层为LoRALayer
                parent = model
                path = name.split('.')
                for p in path[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, path[-1], LoRALayer(module, rank, alpha))
            elif isinstance(module, QuantLinear):
                # 替换原始线性层为LoRALayer
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


def train_epoch(model, train_loader, optimizer, device, accumulation_steps=4):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(train_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss = loss / accumulation_steps  # 梯度累积

        loss.backward()

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
        texts = [item["text"] for item in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # 对于因果语言建模，标签就是输入的偏移版本
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def main():
    # 配置参数
    config = {
        "model_name": r"F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct",
        # "model_name": r"F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct-GPTQ-Int4",
        "batch_size": 1,
        "accumulation_steps": 4,
        "learning_rate": 1e-4,
        "epochs": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "device": "cpu"
        # "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 准备模型和tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config["model_name"])
    model.to(config["device"])

    # 示例数据集 - 替换为你的实际数据
    dataset = [{"text": "这是一条示例文本1"}, {"text": "这是一条示例文本2"}]

    # 准备数据加载器
    train_loader = prepare_dataloader(tokenizer, dataset, config["batch_size"])

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
