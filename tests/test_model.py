"""Tests for the full DariaFormerModel."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel


def _small_config() -> DariaFormerConfig:
    """Minimal config for fast testing."""
    return DariaFormerConfig(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        max_seq_len=128,
        sliding_window_size=32,
        num_global_tokens=4,
        ffn_hidden_dim=128,
        working_memory_slots=8,
        episodic_memory_slots=4,
        persistent_memory_slots=4,
        persona_memory_slots=4,
        memory_key_dim=16,
        memory_top_k=4,
        emotion_dim=16,
        emotion_categories=4,
        persona_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        image_enabled=False,
        audio_enabled=False,
    )


class TestDariaFormerModel:
    def test_forward_basic(self):
        config = _small_config()
        model = DariaFormerModel(config)
        model.eval()

        input_ids = torch.randint(0, 256, (2, 32))
        out = model(input_ids)

        assert "logits" in out
        assert out["logits"].shape == (2, 32, 256)
        assert "hidden_states" in out
        assert "kv_cache" in out

    def test_forward_with_emotion(self):
        config = _small_config()
        model = DariaFormerModel(config)
        model.eval()

        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids, return_emotion=True)

        assert "emotion_state" in out
        assert out["emotion_state"].shape == (2, 16)
        assert "emotion_logits" in out

    def test_forward_with_multimodal(self):
        config = _small_config()
        config.image_enabled = True
        config.image_size = 32
        config.image_patch_size = 16
        config.audio_enabled = True
        config.audio_n_mels = 20
        config.audio_max_len = 100
        config.audio_conv_channels = 32

        model = DariaFormerModel(config)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 16))
        images = torch.randn(1, 3, 32, 32)
        audio = torch.randn(1, 20, 50)

        out = model(input_ids, images=images, audio=audio)
        assert out["logits"].shape == (1, 16, 256)

    def test_count_parameters(self):
        config = _small_config()
        model = DariaFormerModel(config)
        n_params = model.count_parameters()
        assert n_params > 0

    def test_backward(self):
        config = _small_config()
        model = DariaFormerModel(config)
        model.train()

        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        logits = out["logits"]

        # Compute simple loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, 256),
            shift_labels.view(-1),
        )
        loss.backward()

        # Check gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
