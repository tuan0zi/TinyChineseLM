from torch import nn

from .rmsnorm import RMSNorm
from .attention import MultiHeadSelfAttention
from .mlp import SwiGLUMLP


class DecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 2048
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            hidden_dim
        )

        self.attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        self.ffn_norm = RMSNorm(
            hidden_dim
        )

        self.mlp = SwiGLUMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim
        )


    def forward(self, x):

        h = x + self.attention(
            self.attention_norm(x)
        )

        output = h + self.mlp(
            self.ffn_norm(h)
        )

        return output