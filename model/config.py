from dataclasses import dataclass


@dataclass
class ModelConfig:

    vocab_size: int

    hidden_dim: int = 128

    num_layers: int = 2

    num_heads: int = 4

    intermediate_dim: int = 344

    max_seq_len: int = 128