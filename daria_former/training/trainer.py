"""Trainer — full training loop with AMP, gradient accumulation, DDP, checkpointing."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.training.losses import DariaFormerLoss
from daria_former.training.optimizer import build_optimizer
from daria_former.training.scheduler import build_scheduler
from daria_former.utils.checkpoint import save_checkpoint, load_checkpoint
from daria_former.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Daria-Former training loop.

    Features:
        - Mixed precision (AMP)
        - Gradient accumulation
        - Gradient clipping
        - Distributed training (DDP)
        - Checkpoint saving/loading
        - Logging with optional wandb
        - Evaluation loop
    """

    def __init__(
        self,
        model: DariaFormerModel,
        config: DariaFormerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_steps: int = 100000,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        save_dir: str = "checkpoints",
        save_every: int = 1000,
        eval_every: int = 500,
        log_every: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "daria-former",
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.use_wandb = use_wandb
        self.global_step = 0

        # Device setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_distributed = dist.is_initialized()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.is_main = self.local_rank == 0

        # Model
        self.model = model.to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Loss
        self.criterion = DariaFormerLoss(config).to(self.device)

        # Optimizer & Scheduler
        self.optimizer = build_optimizer(
            self.model, lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Wandb
        if self.use_wandb and self.is_main:
            try:
                import wandb
                wandb.init(project=wandb_project, config=config.to_dict())
            except ImportError:
                logger.warning("wandb not installed, disabling")
                self.use_wandb = False

        # Resume
        if resume_from:
            self._resume(resume_from)

    def train(self):
        """Run the full training loop."""
        self.model.train()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        data_iter = iter(self.train_dataloader)
        accum_loss = 0.0

        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.is_main,
        )

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            loss_dict = self._train_step(batch)
            accum_loss += loss_dict["loss"].item()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.log_every == 0 and self.is_main:
                avg_loss = accum_loss / self.log_every
                lr = self.scheduler.get_last_lr()[0]
                msg = f"step={self.global_step} loss={avg_loss:.4f} lr={lr:.2e}"
                for k, v in loss_dict.items():
                    if k != "loss":
                        msg += f" {k}={v.item():.4f}"
                logger.info(msg)

                if self.use_wandb:
                    import wandb
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                    }
                    for k, v in loss_dict.items():
                        log_dict[f"train/{k}"] = v.item()
                    wandb.log(log_dict, step=self.global_step)

                accum_loss = 0.0

            # Eval
            if (
                self.eval_dataloader is not None
                and self.global_step % self.eval_every == 0
                and self.is_main
            ):
                eval_loss = self.evaluate()
                logger.info(f"step={self.global_step} eval_loss={eval_loss:.4f}")
                if self.use_wandb:
                    import wandb
                    wandb.log({"eval/loss": eval_loss}, step=self.global_step)
                self.model.train()

            # Save
            if self.global_step % self.save_every == 0 and self.is_main:
                self._save()

        pbar.close()

        # Final save
        if self.is_main:
            self._save()
            logger.info("Training complete.")

    def _train_step(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        images = batch.get("images")
        audio = batch.get("audio")

        if images is not None:
            images = images.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)

        with autocast(enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                images=images,
                audio=audio,
                return_emotion=True,
            )

            loss_dict = self.criterion(
                logits=outputs["logits"],
                labels=labels,
                hidden_states=outputs["hidden_states"],
                emotion_logits=outputs.get("emotion_logits"),
                curr_esv=outputs.get("emotion_state"),
            )

        # Scale loss for gradient accumulation
        loss = loss_dict["loss"] / self.gradient_accumulation_steps
        self.scaler.scale(loss).backward()

        return loss_dict

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = self.model(input_ids=input_ids, return_emotion=True)
            loss_dict = self.criterion(
                logits=outputs["logits"],
                labels=labels,
                hidden_states=outputs["hidden_states"],
                emotion_logits=outputs.get("emotion_logits"),
                curr_esv=outputs.get("emotion_state"),
            )
            total_loss += loss_dict["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save(self):
        path = self.save_dir / f"checkpoint-{self.global_step}"
        model_to_save = self.model.module if self.is_distributed else self.model
        save_checkpoint(
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            config=self.config,
            path=str(path),
        )
        logger.info(f"Saved checkpoint to {path}")

    def _resume(self, path: str):
        state = load_checkpoint(path, self.device)
        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["step"]
        logger.info(f"Resumed from {path} at step {self.global_step}")
