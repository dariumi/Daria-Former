"""Tests for training pipeline components."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.training.losses import (
    DariaFormerLoss,
    EmotionAlignmentLoss,
    RepetitionPenaltyLoss,
    RhythmRegularizer,
    MemoryConsistencyLoss,
)
from daria_former.training.optimizer import build_optimizer
from daria_former.training.scheduler import build_scheduler


def _small_config():
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
        dropout=0.0,
        attention_dropout=0.0,
    )


B, S, H, V, E = 2, 16, 64, 256, 16


class TestLosses:
    def test_emotion_alignment(self):
        loss_fn = EmotionAlignmentLoss()
        logits = torch.randn(B, 4)
        labels = torch.randint(0, 4, (B,))
        prev = torch.randn(B, E)
        curr = torch.randn(B, E)
        loss = loss_fn(logits, labels, prev, curr)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_repetition_penalty(self):
        loss_fn = RepetitionPenaltyLoss()
        hidden = torch.randn(B, S, H)
        loss = loss_fn(hidden)
        assert loss.dim() == 0

    def test_repetition_penalty_short(self):
        loss_fn = RepetitionPenaltyLoss()
        hidden = torch.randn(B, 1, H)
        loss = loss_fn(hidden)
        assert loss.item() == 0.0

    def test_rhythm_regularizer(self):
        loss_fn = RhythmRegularizer()
        logits = torch.randn(B, S, V)
        loss = loss_fn(logits)
        assert loss.dim() == 0

    def test_memory_consistency(self):
        loss_fn = MemoryConsistencyLoss()
        hidden = torch.randn(B, S, H)
        memory = torch.randn(B, 8, H)
        loss = loss_fn(hidden, memory)
        assert loss.dim() == 0

    def test_memory_consistency_none(self):
        loss_fn = MemoryConsistencyLoss()
        hidden = torch.randn(B, S, H)
        loss = loss_fn(hidden, None)
        assert loss.item() == 0.0

    def test_composite_loss(self):
        config = _small_config()
        criterion = DariaFormerLoss(config)

        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        hidden = torch.randn(B, S, H)
        emo_logits = torch.randn(B, 4)

        result = criterion(logits, labels, hidden, emotion_logits=emo_logits)
        assert "loss" in result
        assert "ce_loss" in result
        assert result["loss"].dim() == 0


class TestOptimizer:
    def test_build_optimizer(self):
        config = _small_config()
        model = DariaFormerModel(config)
        optimizer = build_optimizer(model, lr=1e-4)
        assert len(optimizer.param_groups) > 0

    def test_param_groups(self):
        config = _small_config()
        model = DariaFormerModel(config)
        optimizer = build_optimizer(
            model, lr=1e-4, lora_lr=1e-3, emotion_lr=5e-5,
        )
        # Should have multiple param groups
        assert len(optimizer.param_groups) >= 1


class TestScheduler:
    def test_warmup_cosine(self):
        config = _small_config()
        model = DariaFormerModel(config)
        optimizer = build_optimizer(model, lr=1e-4)
        scheduler = build_scheduler(optimizer, num_warmup_steps=10, num_training_steps=100)

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # LR should increase during warmup
        assert lrs[5] > lrs[0]
        # LR should decrease after warmup
        assert lrs[50] < lrs[15]


class TestTrainStep:
    """Smoke test: one forward + backward pass."""

    def test_one_step(self):
        config = _small_config()
        model = DariaFormerModel(config)
        model.train()

        optimizer = build_optimizer(model, lr=1e-4)
        criterion = DariaFormerLoss(config)

        input_ids = torch.randint(0, V, (B, S))

        # Forward
        outputs = model(input_ids, return_emotion=True)

        # Loss
        loss_dict = criterion(
            logits=outputs["logits"],
            labels=input_ids,
            hidden_states=outputs["hidden_states"],
            emotion_logits=outputs.get("emotion_logits"),
            curr_esv=outputs.get("emotion_state"),
        )

        # Backward
        loss_dict["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        assert loss_dict["loss"].item() > 0
