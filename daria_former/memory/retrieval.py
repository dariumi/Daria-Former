"""Memory Retrieval Module — selective top-k attention over memory banks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MemoryRetrievalModule(nn.Module):
    """Selectively retrieves relevant memory entries via top-k attention.

    Instead of attending to all memory slots equally, this module first
    scores all slots against the query, selects top-k, and returns
    only those for downstream cross-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        key_dim: int,
        top_k: int = 32,
    ):
        super().__init__()
        self.top_k = top_k
        self.query_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, key_dim, bias=False)
        self.scale = key_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, S, H) — current hidden states
            memory_keys: (B, M, H) — memory bank keys
            memory_values: (B, M, H) — memory bank values

        Returns:
            retrieved: (B, top_k, H) — selected memory values
        """
        B, S, H = query.shape
        M = memory_keys.shape[1]

        # Project and compute relevance scores
        q = self.query_proj(query)       # (B, S, K)
        k = self.key_proj(memory_keys)    # (B, M, K)

        # Aggregate query across sequence for slot scoring
        q_mean = q.mean(dim=1, keepdim=True)  # (B, 1, K)
        scores = torch.matmul(q_mean, k.transpose(-2, -1)).squeeze(1) * self.scale  # (B, M)

        # Top-k selection
        k_actual = min(self.top_k, M)
        topk_scores, topk_indices = scores.topk(k_actual, dim=-1)  # (B, top_k)

        # Gather selected values
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, H)
        retrieved = torch.gather(memory_values, 1, topk_indices_expanded)  # (B, top_k, H)

        # Weight by softmax of scores
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (B, top_k, 1)
        retrieved = retrieved * weights

        return retrieved
