import torch

from .config import ModelConfig
from .model import TinyLlama


config = ModelConfig(
    vocab_size=8192,
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    intermediate_dim=344,
    max_seq_len=128
)


model = TinyLlama(
    config
)


input_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 32)
)


logits = model(
    input_ids
)


print(
    "input_ids shape:",
    input_ids.shape
)

print(
    "logits shape:",
    logits.shape
)