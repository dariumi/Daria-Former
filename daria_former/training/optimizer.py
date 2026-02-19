"""Optimizer builder with separate parameter groups for base, LoRA, and emotion."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    lora_lr: Optional[float] = None,
    emotion_lr: Optional[float] = None,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Build AdamW optimizer with separate parameter groups.

    Groups:
        1. Base parameters (all except LoRA and emotion) — lr, weight_decay
        2. LoRA parameters (if any) — lora_lr (default: 10x lr), no weight_decay
        3. Emotion parameters — emotion_lr (default: lr), no weight_decay

    Args:
        model: DariaFormerModel
        lr: base learning rate
        weight_decay: base weight decay
        lora_lr: learning rate for LoRA params (default: 10 * lr)
        emotion_lr: learning rate for emotion params (default: lr)
        betas: Adam betas
        eps: Adam epsilon

    Returns:
        AdamW optimizer
    """
    if lora_lr is None:
        lora_lr = lr * 10
    if emotion_lr is None:
        emotion_lr = lr

    lora_params = []
    emotion_params = []
    decay_params = []
    no_decay_params = []

    lora_names = {"lora_A", "lora_B", "extra_lora"}
    emotion_names = {"emotion", "esv", "style_proj", "category_head"}
    no_decay_types = (nn.LayerNorm, nn.Embedding)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_lora = any(ln in name for ln in lora_names)
        is_emotion = any(en in name for en in emotion_names)

        if is_lora:
            lora_params.append(param)
        elif is_emotion:
            emotion_params.append(param)
        elif isinstance(
            dict(model.named_modules()).get(
                ".".join(name.split(".")[:-1]), None
            ),
            no_decay_types,
        ) or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
        })
    if lora_params:
        param_groups.append({
            "params": lora_params,
            "lr": lora_lr,
            "weight_decay": 0.0,
        })
    if emotion_params:
        param_groups.append({
            "params": emotion_params,
            "lr": emotion_lr,
            "weight_decay": 0.0,
        })

    return AdamW(param_groups, betas=betas, eps=eps)
