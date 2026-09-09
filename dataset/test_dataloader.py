from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pretrain_dataset import PretrainDataset


# 加载 tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "../tokenizer/tokenizer_k"
)


# 加载 Dataset

dataset = PretrainDataset(
    data_path="../data/processed/pretrain.jsonl",
    tokenizer=tokenizer,
    max_length=32
)


# 创建 DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# 取一个 batch

batch = next(iter(dataloader))


print(batch)


print(
    "input shape:",
    batch["input_ids"].shape
)


print(
    "label shape:",
    batch["labels"].shape
)