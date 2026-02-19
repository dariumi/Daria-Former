"""Memory Integration System (MIS) — orchestrates all memory banks."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from daria_former.config import DariaFormerConfig
from daria_former.memory.memory_bank import (
    WorkingMemory,
    EpisodicMemory,
    PersistentMemory,
    PersonaMemory,
)
from daria_former.memory.retrieval import MemoryRetrievalModule


class MemoryIntegrationSystem(nn.Module):
    """Orchestrates all memory banks and provides unified KV for MCA.

    Memory hierarchy:
        Working  — current dialog (refreshed each forward pass)
        Episodic — session-level (updated at episode boundaries)
        Persistent — long-term (updated via consolidation)
        Persona — identity embeddings (mostly static)

    The MIS retrieves from each bank and concatenates the results
    into a single KV tensor for the Memory Attention path in MCA.
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        H = config.hidden_dim

        self.working = WorkingMemory(config.working_memory_slots, H)
        self.episodic = EpisodicMemory(config.episodic_memory_slots, H)
        self.persistent = PersistentMemory(config.persistent_memory_slots, H)
        self.persona = PersonaMemory(config.persona_memory_slots, H)

        self.retrieval = MemoryRetrievalModule(
            hidden_dim=H,
            key_dim=config.memory_key_dim,
            top_k=config.memory_top_k,
        )

        # Projection to merge retrieved memories
        self.merge_proj = nn.Linear(H, H)

    def get_memory_kv(
        self,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve and merge memory KV for the attention layer.

        Args:
            query: (B, S, H) — current hidden states for relevance scoring

        Returns:
            memory_kv: (B, M_total, H) — merged memory values
        """
        B = query.shape[0]

        # Read all memory banks
        w_vals = self.working.read(B)
        e_vals = self.episodic.read(B)
        p_vals = self.persistent.read(B)

        # Concatenate all non-persona memory
        all_keys = torch.cat([
            self.working.keys.unsqueeze(0).expand(B, -1, -1),
            self.episodic.keys.unsqueeze(0).expand(B, -1, -1),
            self.persistent.keys.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)
        all_values = torch.cat([w_vals, e_vals, p_vals], dim=1)

        # Selective retrieval
        retrieved = self.retrieval(query, all_keys, all_values)

        return self.merge_proj(retrieved)

    def get_persona_kv(self, batch_size: int) -> torch.Tensor:
        """Return persona memory for the Persona Attention path.

        Returns:
            (B, persona_slots, H)
        """
        return self.persona.read(batch_size)

    def update_working_memory(self, hidden_states: torch.Tensor) -> None:
        """Update working memory from the latest hidden states."""
        self.working.update_from_hidden(hidden_states)

    def encode_and_store_episode(self, episode_hidden: torch.Tensor) -> None:
        """Compress an episode and attempt to store in persistent memory."""
        summary = self.episodic.encode_episode(episode_hidden)
        # Store in persistent memory if important enough
        if summary.dim() > 1:
            summary = summary.mean(dim=0)
        self.persistent.consolidate(summary.detach())
