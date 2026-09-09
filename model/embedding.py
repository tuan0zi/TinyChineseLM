import torch
from torch import nn


class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int
    ):
        super().__init__()


        self.embedding = nn.Embedding(    # 创建 vocab_size*hidden_size 的矩阵
            vocab_size,
            hidden_dim
        )


    def forward(
        self,
        input_ids
    ):

        x = self.embedding(
            input_ids
        )

        return x