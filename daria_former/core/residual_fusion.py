"""Residual Fusion — learned weighted combination of attention + FFN + skip."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFusion(nn.Module):
    """Fuses attention output, FFN output, and the skip connection
    via learned scalar gates.

    ``output = g_attn * attn_out + g_ffn * ffn_out + g_skip * skip``

    Gates are produced by a small projection from the concatenated inputs
    and passed through softmax so they sum to 1.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Input: concat of 3 streams (each hidden_dim) -> 3 gate logits
        self.gate_proj = nn.Linear(hidden_dim * 3, 3, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        # Initialize bias so that skip connection dominates at init
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.gate_proj.bias.data[2] = 1.0  # favor skip at init

    def forward(
        self,
        attn_out: torch.Tensor,
        ffn_out: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs: (batch, seq_len, hidden_dim)
        Returns: (batch, seq_len, hidden_dim)
        """
        combined = torch.cat([attn_out, ffn_out, skip], dim=-1)
        # Per-position gate: (B, S, 3)
        gates = F.softmax(self.gate_proj(combined), dim=-1)

        g_attn = gates[..., 0:1]
        g_ffn = gates[..., 1:2]
        g_skip = gates[..., 2:3]

        return g_attn * attn_out + g_ffn * ffn_out + g_skip * skip
