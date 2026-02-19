"""Tests for Multi-Context Attention and positional encoding."""

import torch
import pytest

from daria_former.core.positional import RotaryEmbedding, apply_rotary_emb
from daria_former.core.multi_context_attention import MultiContextAttention


class TestRotaryEmbedding:
    def test_shape(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(seq_len=128)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_offset(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(seq_len=10, offset=100)
        assert cos.shape == (10, 64)

    def test_dynamic_scaling(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128, scaling_factor=2.0)
        # Request beyond max_seq_len to trigger dynamic scaling
        cos, sin = rope(seq_len=256)
        assert cos.shape == (256, 64)

    def test_apply_rotary(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        cos, sin = rope(seq_len=16)

        B, H, S, D = 2, 4, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)

        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Rotated should differ from original
        assert not torch.allclose(q_rot, q)


class TestMultiContextAttention:
    @pytest.fixture
    def mca(self):
        return MultiContextAttention(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            sliding_window_size=64,
            dropout=0.0,
            attention_dropout=0.0,
            modality_attention=False,
        )

    def test_basic_forward(self, mca):
        B, S, H = 2, 32, 128
        x = torch.randn(B, S, H)
        cos = torch.randn(S, 32)
        sin = torch.randn(S, 32)

        out, cache = mca(x, cos=cos, sin=sin)
        assert out.shape == (B, S, H)
        assert "local" in cache
        assert "long" in cache

    def test_with_memory(self, mca):
        B, S, H = 2, 16, 128
        x = torch.randn(B, S, H)
        memory_kv = torch.randn(B, 8, H)

        out, _ = mca(x, memory_kv=memory_kv)
        assert out.shape == (B, S, H)

    def test_with_persona(self, mca):
        B, S, H = 2, 16, 128
        x = torch.randn(B, S, H)
        persona_kv = torch.randn(B, 4, H)

        out, _ = mca(x, persona_kv=persona_kv)
        assert out.shape == (B, S, H)

    def test_with_modality(self):
        mca = MultiContextAttention(
            hidden_dim=128, num_heads=4, head_dim=32,
            modality_attention=True,
        )
        B, S, H = 2, 16, 128
        x = torch.randn(B, S, H)
        mod_kv = torch.randn(B, 10, H)

        out, _ = mca(x, modality_kv=mod_kv)
        assert out.shape == (B, S, H)
