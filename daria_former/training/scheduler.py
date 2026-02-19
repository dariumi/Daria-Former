"""Learning rate scheduler with warmup + cosine decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Warmup + cosine decay scheduler.

    Args:
        optimizer: optimizer to schedule
        num_warmup_steps: linear warmup steps
        num_training_steps: total training steps
        min_lr_ratio: minimum LR as fraction of peak

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine * (1.0 - min_lr_ratio) + min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)
