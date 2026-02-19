"""AdaptiveNorm — LayerNorm conditioned on Emotion State Vector (ESV)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveNorm(nn.Module):
    """Layer normalization with affine modulation driven by the ESV.

    ``output = (LN(x)) * (1 + scale(esv)) + shift(esv)``

    When *esv* is ``None`` this falls back to plain LayerNorm.
    """

    def __init__(self, hidden_dim: int, emotion_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.scale_proj = nn.Linear(emotion_dim, hidden_dim)
        self.shift_proj = nn.Linear(emotion_dim, hidden_dim)

        # Initialize near-identity
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        esv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            esv: (batch, emotion_dim) or None
        """
        h = self.norm(x)
        if esv is not None:
            # esv: (B, E) -> (B, 1, H)
            scale = self.scale_proj(esv).unsqueeze(1)
            shift = self.shift_proj(esv).unsqueeze(1)
            h = h * (1.0 + scale) + shift
        return h
