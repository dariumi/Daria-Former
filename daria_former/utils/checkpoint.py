"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from daria_former.config import DariaFormerConfig


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    config: DariaFormerConfig,
    path: str,
) -> None:
    """Save a training checkpoint.

    Saves:
        - model state dict
        - optimizer state dict
        - scheduler state dict
        - global step
        - config
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path / "training_state.pt",
    )
    config.save(path / "config.yaml")


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Returns:
        dict with "model", "optimizer", "scheduler", "step", "config"
    """
    path = Path(path)

    model_state = torch.load(path / "model.pt", map_location=device, weights_only=True)
    training_state = torch.load(
        path / "training_state.pt", map_location=device, weights_only=True,
    )
    config = DariaFormerConfig.load(path / "config.yaml")

    return {
        "model": model_state,
        "optimizer": training_state["optimizer"],
        "scheduler": training_state["scheduler"],
        "step": training_state["step"],
        "config": config,
    }
