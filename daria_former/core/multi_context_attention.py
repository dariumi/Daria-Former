"""Multi-Context Attention (MCA) — the core attention system of Daria-Former.

Five parallel attention heads, each with its own projections and scaling,
merged via dynamic gating.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from daria_former.core.positional import apply_rotary_emb


# ── Helpers ──────────────────────────────────────────────────────────────

def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Upper-triangular causal mask (True = masked)."""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _sliding_window_mask(
    seq_len: int, window: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Causal + sliding window mask (True = masked)."""
    causal = _causal_mask(seq_len, device, dtype)
    # Mask everything further than `window` positions back
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    too_far = (row_idx - col_idx) > window
    return causal | too_far


# ── Attention sub-modules ────────────────────────────────────────────────

class _SelfAttentionHead(nn.Module):
    """Single self-attention with its own Q/K/V/O projections."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=self.num_heads)

        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out), new_cache


class _CrossAttentionHead(nn.Module):
    """Cross-attention: Q from hidden, K/V from external source."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, S, H)
            context: (B, M, H) — external key/value source
        """
        B, S, _ = query.shape
        q = rearrange(self.q_proj(query), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.k_proj(context), "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(self.v_proj(context), "b m (h d) -> b h m d", h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


# ── Main MCA ─────────────────────────────────────────────────────────────

class MultiContextAttention(nn.Module):
    """Multi-Context Attention system with 5 parallel attention pathways.

    1. Local Context Attention  — sliding window causal
    2. Long Context Attention   — full causal with global tokens
    3. Memory Attention         — cross-attention over memory KV
    4. Persona/Identity Attention — cross-attention over persona embeddings
    5. Modality Attention       — cross-attention for image/audio tokens (optional)

    Each pathway has independent projections and scaling.
    A learned dynamic gate merges all active outputs.
    """

    NUM_PATHS = 5

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        sliding_window_size: int = 1024,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        modality_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window_size = sliding_window_size
        self.modality_attention = modality_attention

        # 1. Local context — sliding window self-attention
        self.local_attn = _SelfAttentionHead(
            hidden_dim, num_heads, head_dim, attention_dropout
        )

        # 2. Long context — full causal self-attention
        self.long_attn = _SelfAttentionHead(
            hidden_dim, num_heads, head_dim, attention_dropout
        )

        # 3. Memory attention — cross-attention
        self.memory_attn = _CrossAttentionHead(
            hidden_dim, num_heads, head_dim, attention_dropout
        )

        # 4. Persona attention — cross-attention
        self.persona_attn = _CrossAttentionHead(
            hidden_dim, num_heads, head_dim, attention_dropout
        )

        # 5. Modality attention (optional) — cross-attention
        if modality_attention:
            self.modality_attn = _CrossAttentionHead(
                hidden_dim, num_heads, head_dim, attention_dropout
            )
        else:
            self.modality_attn = None

        # Dynamic gating — merges all pathways
        num_active = 5 if modality_attention else 4
        self.gate_proj = nn.Linear(hidden_dim * num_active, num_active, bias=True)
        # Initialize uniform gating
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        memory_kv: torch.Tensor | None = None,
        persona_kv: torch.Tensor | None = None,
        modality_kv: torch.Tensor | None = None,
        kv_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        emotion_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, S, H)
            cos, sin: RoPE embeddings, (S, head_dim)
            memory_kv: (B, M_mem, H) or None
            persona_kv: (B, M_per, H) or None
            modality_kv: (B, M_mod, H) or None
            kv_cache: dict of cached KV per pathway
            emotion_scale: scalar to modulate attention

        Returns:
            output: (B, S, H)
            new_kv_cache: dict
        """
        B, S, H = x.shape
        cache = kv_cache or {}
        new_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        outputs = []

        # 1. Local context (sliding window)
        local_mask = _sliding_window_mask(S, self.sliding_window_size, x.device, x.dtype)
        local_out, new_cache["local"] = self.local_attn(
            x, cos, sin, local_mask, cache.get("local"),
        )
        outputs.append(local_out)

        # 2. Long context (full causal)
        causal_mask = _causal_mask(
            S + (cache["long"][0].shape[2] if "long" in cache else 0),
            x.device, x.dtype,
        ) if kv_cache is None else None
        # For long context with cache, mask is handled inside
        long_mask = _causal_mask(S, x.device, x.dtype) if causal_mask is None or kv_cache else causal_mask
        long_out, new_cache["long"] = self.long_attn(
            x, cos, sin, long_mask, cache.get("long"),
        )
        outputs.append(long_out)

        # 3. Memory attention
        if memory_kv is not None:
            mem_out = self.memory_attn(x, memory_kv)
        else:
            mem_out = torch.zeros_like(x)
        outputs.append(mem_out)

        # 4. Persona attention
        if persona_kv is not None:
            per_out = self.persona_attn(x, persona_kv)
        else:
            per_out = torch.zeros_like(x)
        outputs.append(per_out)

        # 5. Modality attention (optional)
        if self.modality_attn is not None:
            if modality_kv is not None:
                mod_out = self.modality_attn(x, modality_kv)
            else:
                mod_out = torch.zeros_like(x)
            outputs.append(mod_out)

        # Dynamic gating
        gate_input = torch.cat(outputs, dim=-1)  # (B, S, H*N)
        gates = F.softmax(self.gate_proj(gate_input), dim=-1)  # (B, S, N)

        fused = torch.zeros_like(x)
        for i, out in enumerate(outputs):
            fused = fused + gates[..., i:i + 1] * out

        # Emotion scaling
        fused = fused * emotion_scale

        return self.dropout(fused), new_cache
