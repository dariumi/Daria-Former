"""CLI: Export a Daria-Former model (merge LoRA, save for inference)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.lora.lora_manager import LoRAManager
from daria_former.utils.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export Daria-Former model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base weights")
    parser.add_argument("--half", action="store_true", help="Export in float16")
    return parser.parse_args()


def main():
    setup_logging()
    logger = get_logger("cli.export")
    args = parse_args()

    # Load
    config = DariaFormerConfig.load(f"{args.checkpoint}/config.yaml")
    model = DariaFormerModel(config)

    state_dict = torch.load(
        f"{args.checkpoint}/model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {args.checkpoint}")

    # Merge LoRA if requested
    if args.merge_lora and config.lora_rank > 0:
        manager = LoRAManager(model, rank=config.lora_rank, targets=config.lora_targets)
        manager.merge_all()
        logger.info("Merged LoRA weights into base")

    # Convert to half precision
    if args.half:
        model = model.half()
        logger.info("Converted to float16")

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")
    config.save(output_path / "config.yaml")

    logger.info(f"Exported model to {output_path}")
    logger.info(f"Parameters: {model.count_parameters(trainable_only=False):,}")


if __name__ == "__main__":
    main()
