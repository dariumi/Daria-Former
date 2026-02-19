"""Tests for DAC Block and core sublayers."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.core.adaptive_norm import AdaptiveNorm
from daria_former.core.emotion_ffn import EmotionModulatedFFN
from daria_former.core.residual_fusion import ResidualFusion
from daria_former.core.rhythm_gate import RhythmGate
from daria_former.core.dac_block import DACBlock


B, S, H, E = 2, 16, 128, 32


class TestAdaptiveNorm:
    def test_without_esv(self):
        norm = AdaptiveNorm(H, E)
        x = torch.randn(B, S, H)
        out = norm(x)
        assert out.shape == (B, S, H)

    def test_with_esv(self):
        norm = AdaptiveNorm(H, E)
        x = torch.randn(B, S, H)
        esv = torch.randn(B, E)
        out = norm(x, esv)
        assert out.shape == (B, S, H)


class TestEmotionFFN:
    @pytest.mark.parametrize("activation", ["swiglu", "gelu", "relu"])
    def test_activations(self, activation):
        ffn = EmotionModulatedFFN(H, H * 4, E, activation=activation)
        x = torch.randn(B, S, H)
        esv = torch.randn(B, E)
        out = ffn(x, esv)
        assert out.shape == (B, S, H)

    def test_without_esv(self):
        ffn = EmotionModulatedFFN(H, H * 4, E)
        x = torch.randn(B, S, H)
        out = ffn(x)
        assert out.shape == (B, S, H)


class TestResidualFusion:
    def test_forward(self):
        fusion = ResidualFusion(H)
        a = torch.randn(B, S, H)
        b = torch.randn(B, S, H)
        c = torch.randn(B, S, H)
        out = fusion(a, b, c)
        assert out.shape == (B, S, H)


class TestRhythmGate:
    def test_forward(self):
        gate = RhythmGate(H)
        x = torch.randn(B, S, H)
        out = gate(x)
        assert out.shape == (B, S, H)


class TestDACBlock:
    @pytest.fixture
    def config(self):
        return DariaFormerConfig(
            hidden_dim=H,
            num_layers=1,
            num_heads=4,
            head_dim=32,
            ffn_hidden_dim=H * 4,
            emotion_dim=E,
            max_seq_len=256,
            sliding_window_size=64,
        )

    def test_forward(self, config):
        block = DACBlock(config)
        x = torch.randn(B, S, H)
        esv = torch.randn(B, E)
        cos = torch.randn(S, 32)
        sin = torch.randn(S, 32)

        out, cache = block(x, cos=cos, sin=sin, esv=esv)
        assert out.shape == (B, S, H)
        assert isinstance(cache, dict)

    def test_without_esv(self, config):
        block = DACBlock(config)
        x = torch.randn(B, S, H)
        out, _ = block(x)
        assert out.shape == (B, S, H)
