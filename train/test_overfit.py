import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.pretrain_dataset import PretrainDataset
from model.config import ModelConfig
from model.model import TinyLlama


# =========================
# 1. Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer/tokenizer_k",
    local_files_only=True
)


# =========================
# 2. Dataset
# =========================

dataset = PretrainDataset(
    data_path="data/processed/pretrain.jsonl",
    tokenizer=tokenizer,
    max_length=32
)


# =========================
# 3. DataLoader
# =========================

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# =========================
# 4. Model Config
# =========================

config = ModelConfig(
    vocab_size=len(tokenizer),
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    intermediate_dim=344,
    max_seq_len=32
)


# =========================
# 5. Model
# =========================

model = TinyLlama(config)

model.train()


# =========================
# 6. Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


# =========================
# 7. 固定一个 batch
# =========================

batch = next(iter(dataloader))

input_ids = batch["input_ids"]
labels = batch["labels"]


# =========================
# 8. 反复训练同一个 batch
# =========================

for step in range(200):

    # 清空上一轮梯度
    optimizer.zero_grad()


    # 前向传播
    logits = model(
        input_ids
    )


    # 计算 loss
    loss = F.cross_entropy(
        logits.reshape(
            -1,
            config.vocab_size
        ),
        labels.reshape(-1),
        ignore_index=-100
    )


    # 反向传播
    loss.backward()


    # 更新参数
    optimizer.step()


    # 打印
    if step % 20 == 0:

        print(
            f"step {step:3d} | "
            f"loss {loss.item():.4f}"
        )