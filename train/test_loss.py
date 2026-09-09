import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.pretrain_dataset import PretrainDataset
from model.config import ModelConfig
from model.model import TinyLlama


# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer/tokenizer_k",
    local_files_only=True
)


# 2. 创建 Dataset
dataset = PretrainDataset(
    data_path="data/processed/pretrain.jsonl",
    tokenizer=tokenizer,
    max_length=32
)


# 3. 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# 4. 模型配置
config = ModelConfig(
    vocab_size=len(tokenizer),
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    intermediate_dim=344,
    max_seq_len=32
)


# 5. 创建模型
model = TinyLlama(config)


# 6. 取一个 batch
batch = next(iter(dataloader))

input_ids = batch["input_ids"]
labels = batch["labels"]


# 7. 前向传播
logits = model(input_ids)


# 8. 计算 Cross Entropy Loss
loss = F.cross_entropy(
    logits.reshape(
        -1,
        config.vocab_size
    ),
    labels.reshape(-1),
    ignore_index=-100
)


print("input_ids shape:", input_ids.shape)
print("labels shape:", labels.shape)
print("logits shape:", logits.shape)
print("loss:", loss.item())