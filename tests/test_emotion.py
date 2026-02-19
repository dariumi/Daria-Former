"""Tests for emotion and modulation layers."""

import torch
import pytest

from daria_former.modulation.emotion import EmotionExpressionLayer
from daria_former.modulation.rhythm import ConversationRhythmLayer
from daria_former.modulation.variability import ReactionVariabilitySystem


B, S, H, E, V = 2, 16, 128, 32, 1000


class TestEmotionExpressionLayer:
    @pytest.fixture
    def emotion(self):
        return EmotionExpressionLayer(H, E, num_categories=8)

    def test_init_state(self, emotion):
        esv = emotion.init_state(B, torch.device("cpu"), torch.float32)
        assert esv.shape == (B, E)
        assert (esv == 0).all()

    def test_forward(self, emotion):
        hidden = torch.randn(B, S, H)
        prev_esv = torch.randn(B, E)
        new_esv, attn_scale, temp_mod, cat_logits = emotion(hidden, prev_esv)
        assert new_esv.shape == (B, E)
        assert isinstance(attn_scale, float)
        assert isinstance(temp_mod, float)
        assert cat_logits.shape == (B, 8)

    def test_style_bias(self, emotion):
        esv = torch.randn(B, E)
        bias = emotion.get_style_bias(esv)
        assert bias.shape == (B, 1, H)


class TestConversationRhythmLayer:
    def test_forward_train(self):
        rhythm = ConversationRhythmLayer(H, noise_std=0.01)
        x = torch.randn(B, S, H)
        out = rhythm(x, training=True)
        assert out.shape == (B, S, H)

    def test_forward_eval(self):
        rhythm = ConversationRhythmLayer(H, noise_std=0.01)
        x = torch.randn(B, S, H)
        out = rhythm(x, training=False)
        assert out.shape == (B, S, H)

    def test_rhythm_loss(self):
        rhythm = ConversationRhythmLayer(H)
        logits = torch.randn(B, S, V)
        loss = rhythm.compute_rhythm_loss(logits)
        assert loss.dim() == 0  # scalar


class TestReactionVariabilitySystem:
    @pytest.fixture
    def var_sys(self):
        return ReactionVariabilitySystem(H, V, ngram_size=3, penalty_weight=0.1)

    def test_similarity_penalty(self, var_sys):
        h1 = torch.randn(B, H)
        h2 = torch.randn(B, H)
        penalty = var_sys.compute_similarity_penalty(h1, h2)
        assert penalty.shape == (B,)

    def test_similarity_penalty_no_prev(self, var_sys):
        h1 = torch.randn(B, H)
        penalty = var_sys.compute_similarity_penalty(h1)
        assert (penalty == 0).all()

    def test_ngram_detection(self, var_sys):
        # Create sequence with repetition
        ids = torch.tensor([[1, 2, 3, 1, 2, 3, 4, 5, 6, 7]])
        score = var_sys.detect_ngram_repetition(ids)
        assert score.shape == (1,)
        assert score[0] > 0  # should detect repetition

    def test_full_forward(self, var_sys):
        logits = torch.randn(B, V)
        hidden = torch.randn(B, H)
        prev = torch.randn(B, H)
        gen_ids = torch.randint(0, V, (B, 20))

        adj_logits, temp = var_sys(
            logits, hidden, prev, gen_ids,
            base_temperature=1.0,
        )
        assert adj_logits.shape == (B, V)
        assert temp.shape == (B,)
