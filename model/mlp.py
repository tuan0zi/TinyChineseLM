import torch
from torch import nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_dim,
            hidden_dim,
            bias=False
        )


    def forward(self, x):

        gate = self.gate_proj(x)

        gate = F.silu(gate)

        up = self.up_proj(x)

        hidden = gate * up

        output = self.down_proj(hidden)

        return output