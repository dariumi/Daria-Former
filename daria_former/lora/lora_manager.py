"""LoRA Manager — inject, stack, and switch LoRA adapters at runtime."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from daria_former.lora.lora_linear import LoRALinear


class LoRAManager:
    """Manages LoRA injection and runtime switching across a model.

    Usage::

        manager = LoRAManager(model, rank=16, alpha=32, targets=["q_proj", "v_proj"])
        manager.inject()  # replace targeted nn.Linear with LoRALinear

        # Add extra persona adapter on top
        manager.add_adapter("persona_style", rank=8)
        manager.activate_adapter("persona_style")

        # Switch at runtime
        manager.deactivate_adapter("persona_style")
        manager.activate_adapter("emotion_intense", weight=0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        targets: Optional[List[str]] = None,
    ):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targets = set(targets or ["q_proj", "v_proj"])
        self._lora_modules: List[LoRALinear] = []
        self._injected = False

    def inject(self) -> None:
        """Replace all targeted ``nn.Linear`` modules with ``LoRALinear``."""
        if self._injected:
            return
        self._replace_recursive(self.model, self.targets)
        self._injected = True

    def _replace_recursive(self, module: nn.Module, targets: Set[str]):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                # Copy base weights
                lora_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, lora_layer)
                self._lora_modules.append(lora_layer)
            else:
                self._replace_recursive(child, targets)

    # ── Adapter management ──────────────────────────────────────────────
    def add_adapter(self, name: str, rank: Optional[int] = None, alpha: Optional[float] = None):
        """Add a named extra adapter to all LoRA modules."""
        r = rank or self.rank
        a = alpha or self.alpha
        for m in self._lora_modules:
            m.add_adapter(name, rank=r, alpha=a)

    def activate_adapter(self, name: str, weight: float = 1.0):
        for m in self._lora_modules:
            m.activate_adapter(name, weight=weight)

    def deactivate_adapter(self, name: str):
        for m in self._lora_modules:
            m.deactivate_adapter(name)

    def deactivate_all_extras(self):
        for m in self._lora_modules:
            m.deactivate_all_extras()

    # ── Freeze / Unfreeze ───────────────────────────────────────────────
    def freeze_base(self):
        """Freeze base weights, keep only LoRA trainable."""
        for m in self._lora_modules:
            m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)

    def unfreeze_base(self):
        for m in self._lora_modules:
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

    def lora_parameters(self) -> List[nn.Parameter]:
        """Return only LoRA parameters (for optimizer param group)."""
        params = []
        for m in self._lora_modules:
            if m.lora_A is not None:
                params.extend([m.lora_A, m.lora_B])
            for name in m._extra_adapters:
                A, B = m._extra_adapters[name]
                params.extend([A, B])
        return params

    def merge_all(self) -> None:
        """Merge primary LoRA into base weights (for export)."""
        for m in self._lora_modules:
            m.merge_lora()

    @property
    def num_lora_parameters(self) -> int:
        return sum(p.numel() for p in self.lora_parameters())
