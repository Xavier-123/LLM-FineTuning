import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict

# bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,

# 加载json文件样例
'''
dataset = load_dataset(
    path="json",
    name=None,
    data_dir=None,
    data_files=['../../data/alpaca_zh_demo.json'],
    split='train',
    cache_dir=None,
    token=None,
    num_proc=1,
    trust_remote_code=True,
    streaming=False,
)
'''

dataset_sample = {
    "input_ids": "input_ids",                # 输入： system + user + assistant
    "attention_mask": "attention_mask",      # [1] * len(system + user + assistant)，超过部分设置为
    # 0，（在llamafactory中，会将长度处理为 2 的倍数，暂时没见到奇数长度）
    "labels": "labels"                       # 标签： 除了assistant部分其它全为为-100（ [-100]*len(system + user) + assistant
    # ），以及 attention_mask为0部分设置为 -100
}


class CustomDatasetSFT(Dataset):
    ''' 为了适配自己的数据类型'''
    def __init__(self, tokenizer, data="../../data/alpaca_zh_demo.json", max_length=256):
        super().__init__()

        self.load_data()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.im_start = self.tokenizer('<|im_start|>').input_ids  # 开始 token
        self.im_end = self.tokenizer('<|im_end|>').input_ids  # 结束 token
        self.newline = self.tokenizer('\n').input_ids  # 换行
        self.pad = self.tokenizer('<|endoftext|>').input_ids  # padding token
        self.ignore = [-100]

    def load_data(self):
        if isinstance(self.data, str):
            # 判断是什么类型数据
            if self.data.endswith(".json") or self.data.endswith(".jsonl"):
                with open(self.data, "r", encoding="utf-8") as file:
                    data_samples = json.load(file)
            elif self.data.endswith(".csv"):
                pass
            elif self.data.endswith(".excel"):
                pass
        self.data_samples = data_samples


    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        input_ids = []
        target_ids = []

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.data_samples[idx]['instruction'] + "\n" + self.data_samples[idx]['input']},
            {"role": "assistant", "content": self.data_samples[idx]['output']},
        ]

        for msg in messages:
            role = self.tokenizer(msg['role']).input_ids
            content = self.tokenizer(msg['content']).input_ids
            assistant = self.tokenizer('assistant').input_ids
            if msg['role'] in ['system', 'user']:
                ignore_parts = role + self.newline + content
                input_ids += self.im_start + ignore_parts + self.im_end + self.newline
                # target_ids += self.im_start + self.ignore * len(ignore_parts) + self.im_end + self.newline
            else:
                ignore_parts = role + self.newline
                _input_ids = input_ids + self.im_start + ignore_parts
                input_ids += self.im_start + ignore_parts + content + self.im_end + self.newline
                # target_ids += self.im_start + self.ignore * len(ignore_parts) + content + self.im_end + self.newline

        # 转换为string
        # _input_str = self.tokenizer.decode(_input_ids, skip_special_tokens=False)
        # input_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        labels_ids = input_ids.copy()

        # input_ids包含所有输入ids
        if len(input_ids) < self.max_length:
            input_ids += self.pad * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 把input_ids中system和user部分屏蔽，设置为-100
        if len(labels_ids) < self.max_length:
            labels_ids += self.ignore * (self.max_length - len(labels_ids))
        else:
            labels_ids = labels_ids[:self.max_length]
        labels = torch.tensor(labels_ids, dtype=torch.long)
        labels[:len(_input_ids)] = self.ignore[0]

        # tensor.ne() 是 PyTorch 中的一个逐元素比较操作，表示 "not equal"（不等于）。它用于比较两个张量中对应位置的元素是否不相等。
        # mask，超过input_ids长度，且小于max_length部分，设置为0。
        attention_mask = input_ids.ne(self.pad[0]).type(torch.long)

        samples = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return samples

    def __getitem_from_llamafactory__(self, idx):
        model_inputs = defaultdict(list)
        input_ids, attention_mask, labels =[], [], []
        # 如果有1000条数据，则 len(model_inputs["input_ids"]) == len(model_inputs["attention_mask"]) == 1000
        # model_inputs["input_ids"].append()
        # model_inputs["attention_mask"]
        # model_inputs["labels"]

        pass


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
    my_dataset = CustomDatasetSFT(tokenizer=tokenizer, data="../../data/self_sft.json")
    my_loader = DataLoader(my_dataset, batch_size=2, shuffle=True)

    for batch_idx, batch in enumerate(my_loader):
        print(123)
        print(batch_idx, batch)
