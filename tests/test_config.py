"""Tests for DariaFormerConfig."""

import tempfile
from pathlib import Path

import pytest

from daria_former.config import DariaFormerConfig


class TestConfig:
    def test_default_creation(self):
        config = DariaFormerConfig()
        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.head_dim == 64
        assert config.ffn_hidden_dim == 768 * 4

    def test_post_init_head_dim(self):
        config = DariaFormerConfig(hidden_dim=512, num_heads=8, head_dim=0)
        assert config.head_dim == 64

    def test_post_init_ffn_dim(self):
        config = DariaFormerConfig(hidden_dim=256, num_heads=8, ffn_hidden_dim=0)
        assert config.ffn_hidden_dim == 1024

    def test_assertion_error(self):
        with pytest.raises(AssertionError):
            DariaFormerConfig(hidden_dim=100, num_heads=3)

    def test_presets(self):
        small = DariaFormerConfig.small()
        base = DariaFormerConfig.base()
        large = DariaFormerConfig.large()
        assert small.num_layers < base.num_layers <= large.num_layers
        assert small.hidden_dim < base.hidden_dim < large.hidden_dim

    def test_save_load_yaml(self):
        config = DariaFormerConfig(hidden_dim=512, num_layers=6, num_heads=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)
            loaded = DariaFormerConfig.load(path)
            assert loaded.hidden_dim == 512
            assert loaded.num_layers == 6

    def test_save_load_json(self):
        config = DariaFormerConfig(hidden_dim=256, num_layers=4, num_heads=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)
            loaded = DariaFormerConfig.load(path)
            assert loaded.hidden_dim == 256

    def test_to_dict(self):
        config = DariaFormerConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "hidden_dim" in d
        assert "num_layers" in d

    def test_num_parameters_estimate(self):
        config = DariaFormerConfig.small()
        est = config.num_parameters_estimate
        assert est > 0
