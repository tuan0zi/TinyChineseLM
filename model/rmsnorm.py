import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-6
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_dim)
        )


    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            )
            +
            self.eps
        )

        x = x * rms

        return self.weight * x