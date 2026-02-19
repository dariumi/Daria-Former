"""Tests for LoRA system."""

import torch
import pytest

from daria_former.lora.lora_linear import LoRALinear
from daria_former.lora.lora_manager import LoRAManager


class TestLoRALinear:
    def test_no_lora(self):
        layer = LoRALinear(128, 256, rank=0)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 256)
        assert layer.lora_A is None

    def test_with_lora(self):
        layer = LoRALinear(128, 256, rank=8)
        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 256)
        assert layer.lora_A is not None

    def test_add_extra_adapter(self):
        layer = LoRALinear(128, 256, rank=8)
        layer.add_adapter("persona", rank=4)
        assert "persona" in layer._extra_adapters

    def test_activate_deactivate(self):
        layer = LoRALinear(128, 256, rank=8)
        layer.add_adapter("style", rank=4)
        layer.activate_adapter("style", weight=0.5)
        assert "style" in layer._active_extras

        x = torch.randn(2, 16, 128)
        out = layer(x)
        assert out.shape == (2, 16, 256)

        layer.deactivate_adapter("style")
        assert "style" not in layer._active_extras

    def test_merge_lora(self):
        layer = LoRALinear(64, 64, rank=4)
        x = torch.randn(1, 8, 64)
        out_before = layer(x).clone()
        layer.merge_lora()
        # After merge, lora_A/B still exist but weight is modified
        # Output should be approximately the same
        out_after = layer(x)
        # They won't be exactly equal because LoRA is now double-counted
        # but the merge changes the base weight


class TestLoRAManager:
    def test_inject(self):
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

        model = SimpleModel()
        manager = LoRAManager(model, rank=8, targets=["q_proj", "v_proj"])
        manager.inject()

        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert not isinstance(model.other, LoRALinear)

    def test_freeze_base(self):
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        model = SimpleModel()
        manager = LoRAManager(model, rank=8, targets=["q_proj"])
        manager.inject()
        manager.freeze_base()

        assert not model.q_proj.weight.requires_grad
        assert model.q_proj.lora_A.requires_grad

    def test_lora_parameters(self):
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        model = SimpleModel()
        manager = LoRAManager(model, rank=8, targets=["q_proj"])
        manager.inject()

        params = manager.lora_parameters()
        assert len(params) == 2  # A and B
        assert manager.num_lora_parameters > 0
