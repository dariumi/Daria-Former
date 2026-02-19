"""Tests for input encoders."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.encoders.text_encoder import TextEncoder
from daria_former.encoders.image_encoder import ImageEncoder
from daria_former.encoders.audio_encoder import AudioEncoder


class TestTextEncoder:
    def test_forward(self):
        config = DariaFormerConfig(vocab_size=1000, hidden_dim=128, num_heads=4, head_dim=32)
        encoder = TextEncoder(config)
        ids = torch.randint(0, 1000, (2, 32))
        out = encoder(ids)
        assert out.shape == (2, 32, 128)

    def test_weight_property(self):
        config = DariaFormerConfig(vocab_size=1000, hidden_dim=128, num_heads=4, head_dim=32)
        encoder = TextEncoder(config)
        assert encoder.weight.shape == (1000, 128)


class TestImageEncoder:
    def test_forward(self):
        config = DariaFormerConfig(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            image_enabled=True,
            image_size=64,
            image_patch_size=16,
            image_channels=3,
        )
        encoder = ImageEncoder(config)
        images = torch.randn(2, 3, 64, 64)
        out = encoder(images)
        num_patches = (64 // 16) ** 2  # 16
        assert out.shape == (2, num_patches, 128)

    def test_from_features(self):
        config = DariaFormerConfig(hidden_dim=128, num_heads=4, head_dim=32, image_enabled=True)
        encoder = ImageEncoder(config)
        features = torch.randn(2, 10, 128)
        out = encoder.from_features(features)
        assert out.shape == (2, 10, 128)


class TestAudioEncoder:
    def test_forward(self):
        config = DariaFormerConfig(
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            audio_enabled=True,
            audio_n_mels=40,
            audio_max_len=200,
            audio_conv_channels=64,
        )
        encoder = AudioEncoder(config)
        mel = torch.randn(2, 40, 100)
        out = encoder(mel)
        assert out.shape[0] == 2
        assert out.shape[2] == 128
        # After stride-2 conv: T' ≈ 50
        assert out.shape[1] == 50
