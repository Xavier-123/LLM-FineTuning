import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,

class CustomDatasetSFT(Dataset):
    def __init__(self, tokenizer, data="../../data/alpaca_zh_demo.json", max_length=128):
        super().__init__()
        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as file:
                data_samples = json.load(file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        instruction = self.data_samples[idx]['instruction']
        input = self.data_samples[idx]['input']
        output = self.data_samples[idx]['output']

        bos_token = self.tokenizer.bos_token     # 开始 token
        eos_token = self.tokenizer.eos_token     # 结束 token
        pad_token = self.tokenizer.pad_token     # padding token
        unk_token = self.tokenizer.unk_token     # 未识别 token

        prompt = '''f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"'''
        prompt = prompt.replace("{input}", instruction + '\n' + input)
        output = prompt + output

        # 对于SFT，我们只需要计算response部分的loss
        # padding='longest': 填充至批次最长长度; padding='max_length': 填充至模型最大长度; padding='max_length,max_length=128: 填充至指定长度
        input_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',                    # 'longest', 'max_length', 'do_not_pad', True
            return_tensors="pt"
        )
        # 首先找到 input 的长度
        input_len = input_encodings["input_ids"].shape[1]

        target_encodings = self.tokenizer(
            output,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )

        # 创建attention mask
        target_ids = target_encodings["input_ids"].squeeze()
        attention_mask = target_encodings["attention_mask"].squeeze()

        # 创建labels，将 input 部分设置为-100（在计算loss时忽略）
        labels = target_ids.clone()
        labels[:input_len] = -100

        samples = {
            "input_ids": target_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        return samples


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


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("F:\inspur\LLM_MODEL\Qwen\Qwen2___5-0___5B-Instruct-GPTQ-Int4")
    # 创建数据集和数据加载器
    my_dataset = CustomDatasetSFT(tokenizer=tokenizer, data="../../data/alpaca_zh_demo.json")
    my_loader = DataLoader(my_dataset, batch_size=4, shuffle=True)

    for batch_idx, batch in enumerate(my_loader):
        print(123)
        print(batch_idx, batch)