"""Tests for generation system."""

import torch
import pytest

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.generation.generator import DariaGenerator


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


class TestDariaGenerator:
    @pytest.fixture
    def model_and_gen(self):
        config = _small_config()
        model = DariaFormerModel(config)
        model.eval()
        gen = DariaGenerator(model)
        return model, gen

    def test_greedy_generation(self, model_and_gen):
        model, gen = model_and_gen
        input_ids = torch.randint(0, 256, (1, 8))

        output = gen.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            use_variability=False,
        )
        assert output.shape[0] == 1
        assert output.shape[1] == 13  # 8 prompt + 5 generated

    def test_sampling_generation(self, model_and_gen):
        model, gen = model_and_gen
        input_ids = torch.randint(0, 256, (1, 4))

        output = gen.generate(
            input_ids,
            max_new_tokens=3,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            do_sample=True,
            use_variability=False,
        )
        assert output.shape[1] == 7  # 4 + 3

    def test_eos_stopping(self, model_and_gen):
        model, gen = model_and_gen
        input_ids = torch.tensor([[1, 2, 3]])

        output = gen.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            eos_token_id=0,
            use_variability=False,
        )
        # Should stop at max or at EOS
        assert output.shape[1] <= 103
