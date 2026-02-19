"""Memory Banks — Working, Episodic, Persistent, and Persona memory."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    """Base memory bank: stores a fixed-capacity set of key/value vectors.

    Memory is a learnable buffer of shape ``(num_slots, hidden_dim)``
    that acts as an external KV source for cross-attention.
    """

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        # Learnable memory embeddings
        self.keys = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)

        # Write gate — controls how much new information is written
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.write_proj = nn.Linear(hidden_dim, hidden_dim)

    def read(self, batch_size: int) -> torch.Tensor:
        """Return memory values expanded for batch: (B, num_slots, H)."""
        return self.values.unsqueeze(0).expand(batch_size, -1, -1)

    def read_kv(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) each (B, num_slots, H)."""
        k = self.keys.unsqueeze(0).expand(batch_size, -1, -1)
        v = self.values.unsqueeze(0).expand(batch_size, -1, -1)
        return k, v

    def write(self, new_info: torch.Tensor) -> None:
        """Update memory with new information.

        Args:
            new_info: (num_slots, hidden_dim) or (hidden_dim,) to broadcast
        """
        if new_info.dim() == 1:
            new_info = new_info.unsqueeze(0).expand(self.num_slots, -1)

        gate = torch.sigmoid(self.write_gate(self.values.data)).squeeze(-1)
        projected = self.write_proj(new_info)
        # Gated update
        self.values.data = (
            gate.unsqueeze(-1) * projected
            + (1 - gate.unsqueeze(-1)) * self.values.data
        )


class WorkingMemory(MemoryBank):
    """Short-term working memory for the current dialog context."""

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__(num_slots, hidden_dim)
        # Working memory also compresses recent hidden states
        self.compress = nn.Linear(hidden_dim, hidden_dim)

    def update_from_hidden(self, hidden_states: torch.Tensor) -> None:
        """Compress recent hidden states into working memory.

        Args:
            hidden_states: (B, S, H) — take last num_slots positions
        """
        B, S, H = hidden_states.shape
        n = min(S, self.num_slots)
        recent = hidden_states[:, -n:, :]  # (B, n, H)
        compressed = self.compress(recent.mean(dim=0))  # (n, H)
        # Pad if needed
        if n < self.num_slots:
            pad = torch.zeros(
                self.num_slots - n, H,
                device=compressed.device, dtype=compressed.dtype,
            )
            compressed = torch.cat([pad, compressed], dim=0)
        self.write(compressed)


class EpisodicMemory(MemoryBank):
    """Session-level episodic memory with compression."""

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__(num_slots, hidden_dim)
        self.episode_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def encode_episode(self, episode_hidden: torch.Tensor) -> torch.Tensor:
        """Compress an episode (sequence of hidden states) into a single vector.

        Args:
            episode_hidden: (B, S, H)
        Returns:
            (B, H)
        """
        _, h_n = self.episode_encoder(episode_hidden)
        return h_n.squeeze(0)


class PersistentMemory(MemoryBank):
    """Long-term persistent memory with fixed capacity."""

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__(num_slots, hidden_dim)
        # Importance scoring for memory consolidation
        self.importance_scorer = nn.Linear(hidden_dim, 1)

    def consolidate(self, candidate: torch.Tensor) -> None:
        """Replace least important memory slot with candidate.

        Args:
            candidate: (hidden_dim,) — new memory to store
        """
        scores = self.importance_scorer(self.values.data).squeeze(-1)  # (num_slots,)
        min_idx = scores.argmin()
        self.values.data[min_idx] = candidate
        self.keys.data[min_idx] = candidate


class PersonaMemory(MemoryBank):
    """Identity / persona embeddings — relatively static."""

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__(num_slots, hidden_dim)
        # Persona memory is primarily read-only during inference;
        # updated during persona-LoRA fine-tuning.
