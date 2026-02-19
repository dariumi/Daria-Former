"""Context Router — routes input embeddings to the appropriate attention paths."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextRouter(nn.Module):
    """Routes input tokens into local and long context segments.

    Determines:
    - Which tokens belong to the local sliding window
    - Which tokens serve as global anchor tokens for long context
    - Segment boundaries for hierarchical attention pooling
    - Memory anchor positions

    The router produces routing metadata consumed by the DAC blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        sliding_window_size: int = 1024,
        num_global_tokens: int = 64,
    ):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.num_global_tokens = num_global_tokens

        # Learnable global token embeddings (prepended to sequence)
        self.global_tokens = nn.Parameter(
            torch.randn(num_global_tokens, hidden_dim) * 0.02
        )

        # Anchor scoring — determines which positions become memory anchors
        self.anchor_scorer = nn.Linear(hidden_dim, 1)

        # Segment boundary detector
        self.segment_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, S, H) — input embeddings

        Returns:
            dict with:
                "x_with_global": (B, G+S, H) — input with prepended global tokens
                "global_mask": (B, G+S) bool — True for global token positions
                "anchor_scores": (B, S) — per-position anchor importance
                "segment_logits": (B, S) — segment boundary probabilities
        """
        B, S, H = x.shape

        # Prepend global tokens
        global_expanded = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
        x_with_global = torch.cat([global_expanded, x], dim=1)  # (B, G+S, H)

        # Global token mask
        global_mask = torch.zeros(B, self.num_global_tokens + S, device=x.device, dtype=torch.bool)
        global_mask[:, :self.num_global_tokens] = True

        # Anchor scores (for hierarchical pooling / memory anchors)
        anchor_scores = self.anchor_scorer(x).squeeze(-1)  # (B, S)
        anchor_scores = torch.sigmoid(anchor_scores)

        # Segment boundaries
        segment_logits = self.segment_proj(x).squeeze(-1)  # (B, S)

        return {
            "x_with_global": x_with_global,
            "global_mask": global_mask,
            "anchor_scores": anchor_scores,
            "segment_logits": segment_logits,
        }

    def get_memory_anchors(
        self,
        x: torch.Tensor,
        anchor_scores: torch.Tensor,
        num_anchors: int = 32,
    ) -> torch.Tensor:
        """Select top-scoring positions as memory anchors.

        Args:
            x: (B, S, H)
            anchor_scores: (B, S)
            num_anchors: number of positions to select

        Returns:
            anchors: (B, num_anchors, H)
        """
        B, S, H = x.shape
        k = min(num_anchors, S)
        _, indices = anchor_scores.topk(k, dim=-1)  # (B, k)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, H)
        return torch.gather(x, 1, indices_expanded)
