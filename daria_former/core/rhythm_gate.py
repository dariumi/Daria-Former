"""RhythmGate — sigmoid gate controlling information flow based on variance/entropy."""

from __future__ import annotations

import torch
import torch.nn as nn


class RhythmGate(nn.Module):
    """Gating mechanism driven by local variance and entropy features.

    Computes a per-position sigmoid gate from the hidden state's
    statistical properties, then scales the residual stream.

    This encourages the model to vary its output structure,
    reducing monotonous repetition.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Input features: [mean, var, hidden] -> gate logit
        self.gate_proj = nn.Linear(hidden_dim + 2, hidden_dim)
        # Initialize near-open gate (bias = 1 → sigmoid ≈ 0.73)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.ones_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            gated x: (batch, seq_len, hidden_dim)
        """
        # Per-position statistics across hidden dim
        mean = x.mean(dim=-1, keepdim=True)    # (B, S, 1)
        var = x.var(dim=-1, keepdim=True)       # (B, S, 1)

        features = torch.cat([x, mean, var], dim=-1)  # (B, S, H+2)
        gate = torch.sigmoid(self.gate_proj(features))  # (B, S, H)

        return x * gate
