"""Tests for memory system."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.memory.memory_bank import (
    MemoryBank, WorkingMemory, EpisodicMemory, PersistentMemory, PersonaMemory,
)
from daria_former.memory.retrieval import MemoryRetrievalModule
from daria_former.memory.memory_system import MemoryIntegrationSystem


B, S, H = 2, 16, 128


class TestMemoryBank:
    def test_read(self):
        bank = MemoryBank(32, H)
        vals = bank.read(B)
        assert vals.shape == (B, 32, H)

    def test_read_kv(self):
        bank = MemoryBank(32, H)
        k, v = bank.read_kv(B)
        assert k.shape == (B, 32, H)
        assert v.shape == (B, 32, H)

    def test_write(self):
        bank = MemoryBank(32, H)
        old_vals = bank.values.data.clone()
        bank.write(torch.randn(32, H))
        # Values should have changed
        assert not torch.allclose(bank.values.data, old_vals)


class TestWorkingMemory:
    def test_update(self):
        wm = WorkingMemory(16, H)
        hidden = torch.randn(B, S, H)
        wm.update_from_hidden(hidden)


class TestEpisodicMemory:
    def test_encode_episode(self):
        em = EpisodicMemory(16, H)
        episode = torch.randn(B, 32, H)
        summary = em.encode_episode(episode)
        assert summary.shape == (B, H)


class TestPersistentMemory:
    def test_consolidate(self):
        pm = PersistentMemory(8, H)
        candidate = torch.randn(H)
        pm.consolidate(candidate)


class TestMemoryRetrieval:
    def test_retrieval(self):
        retrieval = MemoryRetrievalModule(H, key_dim=64, top_k=8)
        query = torch.randn(B, S, H)
        keys = torch.randn(B, 32, H)
        values = torch.randn(B, 32, H)

        retrieved = retrieval(query, keys, values)
        assert retrieved.shape == (B, 8, H)


class TestMemoryIntegrationSystem:
    @pytest.fixture
    def mis(self):
        config = DariaFormerConfig(
            hidden_dim=H,
            working_memory_slots=16,
            episodic_memory_slots=8,
            persistent_memory_slots=4,
            persona_memory_slots=4,
            memory_key_dim=32,
            memory_top_k=8,
            num_heads=4,
            head_dim=32,
        )
        return MemoryIntegrationSystem(config)

    def test_get_memory_kv(self, mis):
        query = torch.randn(B, S, H)
        mem_kv = mis.get_memory_kv(query)
        assert mem_kv.shape[0] == B
        assert mem_kv.shape[2] == H

    def test_get_persona_kv(self, mis):
        pkv = mis.get_persona_kv(B)
        assert pkv.shape == (B, 4, H)
