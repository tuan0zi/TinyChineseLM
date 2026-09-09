import math

import torch
from torch import nn

from .rope import RotaryEmbedding


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        [-x2, x1],
        dim=-1
    )


def apply_rope(
    q,
    k,
    cos,
    sin
):
    q_embed = (
        q * cos
        +
        rotate_half(q) * sin
    )

    k_embed = (
        k * cos
        +
        rotate_half(k) * sin
    )

    return q_embed, k_embed
# 只对QK的位置关心，所以只对 QK进行位置编码

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 2048
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.head_dim = (
            hidden_dim
            //
            num_heads
        )

        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_seq_len
        )


    def forward(
        self,
        x
    ):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)


        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
#对QKV进行多头切分

        cos, sin = self.rotary_emb(
            seq_len=seq_len,
            device=x.device
        )


        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)


        q, k = apply_rope(
            q,
            k,
            cos,
            sin
        )


        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = (
            scores
            /
            math.sqrt(self.head_dim)
        )
#缩放点积注意力

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )


        scores = scores.masked_fill(  #将bool=True 的位置进行填充
            causal_mask,
            float("-inf")
        )


        attn_weights = torch.softmax(
            scores,
            dim=-1
        )   #查询对所有其他的位置


        output = torch.matmul(
            attn_weights,
            v
        )


        output = output.transpose(
            1,
            2
        ).contiguous()


        output = output.view(
            batch_size,
            seq_len,
            self.hidden_dim
        )


        output = self.o_proj(
            output
        )


        return output