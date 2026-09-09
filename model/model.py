import torch
from torch import nn

from .config import ModelConfig
from .embedding import TokenEmbedding
from .block import DecoderBlock
from .rmsnorm import RMSNorm


class TinyLlama(nn.Module):

    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()

        self.config = config


        # Token ID -> Token Embedding
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim
        )


        # N 个 Decoder Block
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    intermediate_dim=config.intermediate_dim,
                    max_seq_len=config.max_seq_len
                )
                for _ in range(config.num_layers)
            ]
        )


        # LLaMA 最后的 RMSNorm
        self.norm = RMSNorm(
            config.hidden_dim
        )


        # hidden state -> vocabulary logits
        self.lm_head = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False
        )


    def forward(
        self,
        input_ids
    ):

        # [B, T]
        # ↓
        # [B, T, D]

        x = self.embed_tokens(
            input_ids
        )


        # 依次经过所有 Decoder Block
        for layer in self.layers:

            x = layer(x)


        # 最终归一化
        x = self.norm(x)


        # [B, T, D]
        # ↓
        # [B, T, V]

        logits = self.lm_head(
            x
        )


        return logits