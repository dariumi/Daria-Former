"""LoRA (Low-Rank Adaptation) linear layer — native Daria-Former implementation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with optional low-rank adapters.

    When ``rank == 0`` this behaves identically to a plain ``nn.Linear``.

    Supports multi-LoRA stacking: multiple (A, B) pairs stored by name,
    with a single active adapter at inference time (or weighted merge).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Base weight (always present)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Primary LoRA adapter
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        # Extra named adapters for multi-LoRA stacking
        self._extra_adapters: dict[str, tuple[nn.Parameter, nn.Parameter]] = {}
        self._active_extras: dict[str, float] = {}  # name -> weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # ── Multi-LoRA management ───────────────────────────────────────────
    def add_adapter(self, name: str, rank: int, alpha: float = 16.0):
        """Register a named extra LoRA adapter."""
        A = nn.Parameter(torch.empty(rank, self.in_features, device=self.weight.device))
        B = nn.Parameter(torch.zeros(self.out_features, rank, device=self.weight.device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self._extra_adapters[name] = (A, B)
        # Register as module parameters so optimizer can see them
        self.register_parameter(f"extra_lora_A_{name}", A)
        self.register_parameter(f"extra_lora_B_{name}", B)

    def activate_adapter(self, name: str, weight: float = 1.0):
        self._active_extras[name] = weight

    def deactivate_adapter(self, name: str):
        self._active_extras.pop(name, None)

    def deactivate_all_extras(self):
        self._active_extras.clear()

    # ── Forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        out = F.linear(x, self.weight, self.bias)

        # Primary LoRA
        if self.lora_A is not None:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            out = out + lora_out * self.scaling

        # Extra stacked adapters
        for name, w in self._active_extras.items():
            if name in self._extra_adapters:
                A, B = self._extra_adapters[name]
                extra_out = x @ A.T @ B.T
                scaling = self.alpha / A.shape[0]
                out = out + extra_out * scaling * w

        return out

    def merge_lora(self) -> None:
        """Merge primary LoRA weights into base weight (for export)."""
        if self.lora_A is not None:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def extra_repr(self) -> str:
        s = f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
        if self._extra_adapters:
            s += f", extras={list(self._extra_adapters.keys())}"
        return s
