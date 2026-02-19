"""Text Encoder — token embeddings + input projection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from daria_former.config import DariaFormerConfig


class TextEncoder(nn.Module):
    """Encodes token IDs into the unified latent space.

    Components:
        - Token embedding table
        - Optional dropout
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.scale = math.sqrt(config.hidden_dim)

        self._init_weights(config.initializer_range)

    def _init_weights(self, std: float):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S) — token IDs

        Returns:
            embeddings: (B, S, H)
        """
        x = self.token_embedding(input_ids)
        return self.dropout(x)

    @property
    def weight(self) -> nn.Parameter:
        """Expose embedding weight for weight tying with output head."""
        return self.token_embedding.weight
