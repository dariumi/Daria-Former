"""Rotary Positional Embedding with Dynamic NTK Scaling."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """RoPE with optional dynamic NTK-aware scaling for long contexts.

    When ``scaling_factor > 1``, the base frequency is adjusted dynamically
    so that positions beyond the original ``max_seq_len`` can still be
    encoded, following the NTK-aware interpolation approach.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache up to *seq_len*."""
        self._cached_seq_len = seq_len

        if self.scaling_factor > 1.0 and seq_len > self.max_seq_len:
            # Dynamic NTK scaling: adjust base for extended context
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_seq_len)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device)
                    / self.dim
                )
            )
        else:
            inv_freq = self.inv_freq

        t = torch.arange(seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape ``(seq_len, dim)``."""
        total = offset + seq_len
        if total > self._cached_seq_len:
            self._build_cache(total)
        return (
            self.cos_cached[offset:total],
            self.sin_cached[offset:total],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to *q* and *k*.

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        Rotated (q, k) with same shapes.
    """
    # Broadcast cos/sin to match q/k shape
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated, k_rotated
