"""Emotion-Modulated Feed-Forward Network."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionModulatedFFN(nn.Module):
    """FFN where the activation bias is modulated by the Emotion State Vector.

    Supports SwiGLU, GELU, and ReLU activations.

    ``y = down(act(gate(x) + emotion_bias) * up(x))``   (SwiGLU)
    ``y = down(act(up(x) + emotion_bias))``              (GELU/ReLU)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int,
        emotion_dim: int,
        activation: str = "swiglu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.activation_name = activation

        if activation == "swiglu":
            self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        else:
            self.up_proj = nn.Linear(hidden_dim, ffn_hidden_dim)
            self.gate_proj = None

        self.down_proj = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Emotion bias injection
        self.emotion_bias = nn.Linear(emotion_dim, ffn_hidden_dim)
        nn.init.zeros_(self.emotion_bias.weight)
        nn.init.zeros_(self.emotion_bias.bias)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "swiglu":
            return F.silu(x)
        elif self.activation_name == "gelu":
            return F.gelu(x)
        else:
            return F.relu(x)

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
        e_bias = torch.zeros(
            x.shape[0], 1, self.down_proj.in_features,
            device=x.device, dtype=x.dtype,
        )
        if esv is not None:
            e_bias = self.emotion_bias(esv).unsqueeze(1)  # (B, 1, ffn_dim)

        if self.gate_proj is not None:
            # SwiGLU: silu(gate(x) + bias) * up(x)
            gate = self._activate(self.gate_proj(x) + e_bias)
            up = self.up_proj(x)
            h = gate * up
        else:
            h = self._activate(self.up_proj(x) + e_bias)

        h = self.dropout(h)
        return self.down_proj(h)
