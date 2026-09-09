#LLaMA2的旋转位置编码
import torch
from torch import nn



class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000
    ):
        super().__init__()


        self.dim = dim

        self.max_position_embeddings = max_position_embeddings

        self.base = base


        # 计算 theta
        inv_freq = 1.0 / (
            base
            **
            (
                torch.arange(
                    0,
                    dim,
                    2
                ).float()
                /
                dim
            )
        )


        self.register_buffer(
            "inv_freq",
            inv_freq
        )


    def forward(
        self,
        seq_len: int,
        device
    ):

        # position index

        position_ids = torch.arange(
            seq_len,
            device=device
        )


        # [seq_len, dim/2]

        freqs = torch.outer(
            position_ids,
            self.inv_freq
        )


        # 拼接成完整维度

        emb = torch.cat(
            [
                freqs,
                freqs
            ],
            dim=-1
        )


        cos = emb.cos()

        sin = emb.sin()


        return cos, sin