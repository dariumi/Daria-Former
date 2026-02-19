"""CLI: Train a Daria-Former model."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.data.dataset import TextDataset
from daria_former.data.tokenizer_wrapper import TokenizerWrapper
from daria_former.data.collator import DataCollator
from daria_former.training.trainer import Trainer
from daria_former.utils.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Daria-Former")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to eval data")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="daria-former")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    setup_logging()
    logger = get_logger("cli.train")
    args = parse_args()

    # Config
    config = DariaFormerConfig.load(args.config)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Estimated parameters: {config.num_parameters_estimate:,}")

    # Tokenizer
    tokenizer = TokenizerWrapper(args.tokenizer, max_length=config.max_seq_len)
    if tokenizer.vocab_size != config.vocab_size:
        logger.warning(
            f"Tokenizer vocab ({tokenizer.vocab_size}) != config vocab ({config.vocab_size}). "
            f"Updating config."
        )
        config.vocab_size = tokenizer.vocab_size

    # Dataset
    train_dataset = TextDataset(
        args.train_data,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
    )
    logger.info(f"Train dataset: {len(train_dataset)} chunks")

    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=config.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    eval_loader = None
    if args.eval_data:
        eval_dataset = TextDataset(
            args.eval_data,
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        logger.info(f"Eval dataset: {len(eval_dataset)} chunks")

    # Model
    model = DariaFormerModel(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Trainer
    use_amp = args.use_amp and not args.no_amp
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        use_amp=use_amp,
        save_dir=args.save_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        resume_from=args.resume,
    )

    trainer.train()


if __name__ == "__main__":
    main()
