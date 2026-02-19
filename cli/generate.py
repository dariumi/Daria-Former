"""CLI: Generate text using a trained Daria-Former model."""

from __future__ import annotations

import argparse

import torch

from daria_former.config import DariaFormerConfig
from daria_former.model import DariaFormerModel
from daria_former.generation.generator import DariaGenerator
from daria_former.data.tokenizer_wrapper import TokenizerWrapper
from daria_former.utils.logging_utils import setup_logging, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with Daria-Former")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--prompt", type=str, default="Hello", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--no_variability", action="store_true", help="Disable variability system")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    setup_logging()
    logger = get_logger("cli.generate")
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config and model
    config = DariaFormerConfig.load(f"{args.checkpoint}/config.yaml")
    model = DariaFormerModel(config)

    state_dict = torch.load(
        f"{args.checkpoint}/model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint} ({model.count_parameters():,} params)")

    # Tokenizer
    tokenizer = TokenizerWrapper(args.tokenizer, max_length=config.max_seq_len)

    # Encode prompt
    encoded = tokenizer.encode(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # Generate
    generator = DariaGenerator(model)
    output_ids = generator.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        use_variability=not args.no_variability,
    )

    # Decode and print
    text = tokenizer.decode(output_ids[0])
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"\nGenerated {output_ids.shape[1] - input_ids.shape[1]} tokens")


if __name__ == "__main__":
    main()
