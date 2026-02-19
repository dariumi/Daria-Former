"""DariaGenerator — autoregressive generation with KV-cache and variability control."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DariaGenerator:
    """Autoregressive text generation for DariaFormerModel.

    Supports:
        - Greedy decoding
        - Top-k sampling
        - Top-p (nucleus) sampling
        - Temperature scaling
        - Dynamic temperature via ReactionVariabilitySystem
        - Emotion state propagation
        - KV-cache for efficient generation
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        use_variability: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: (B, S) — prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: base sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            do_sample: if False, use greedy decoding
            eos_token_id: stop generation at this token
            images: optional image input
            audio: optional audio input
            use_variability: whether to use ReactionVariabilitySystem

        Returns:
            generated_ids: (B, S + max_new_tokens) — full sequence
        """
        self.model.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        generated = input_ids.clone()
        kv_cache = None
        emotion_state = None
        prev_hidden = None

        # Prefill: process the full prompt
        outputs = self.model(
            input_ids=generated,
            images=images,
            audio=audio,
            emotion_state=emotion_state,
            kv_cache=None,
            return_emotion=True,
        )
        kv_cache = outputs["kv_cache"]
        emotion_state = outputs.get("emotion_state")
        logits = outputs["logits"][:, -1, :]  # (B, V)
        prev_hidden = outputs["hidden_states"][:, -1, :]  # (B, H)

        # Decode step by step
        for step in range(max_new_tokens):
            # Apply variability system
            if use_variability:
                emotion_temp = 1.0
                if emotion_state is not None:
                    emotion_temp = 1.0 + 0.5 * torch.tanh(
                        self.model.emotion.temperature_proj(emotion_state)
                    ).mean().item()

                logits, dyn_temp = self.model.variability(
                    logits=logits,
                    hidden_state=outputs["hidden_states"][:, -1, :],
                    prev_hidden=prev_hidden,
                    generated_ids=generated,
                    base_temperature=temperature,
                    emotion_temp_mod=emotion_temp,
                )
            else:
                dyn_temp = torch.full((B,), temperature, device=device)

            # Sample next token
            next_token = self._sample(
                logits, dyn_temp, top_k, top_p, do_sample,
            )  # (B,)

            # Append to generated
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Store previous hidden for next step
            prev_hidden = outputs["hidden_states"][:, -1, :]

            # Forward pass for next token (with KV-cache)
            outputs = self.model(
                input_ids=next_token.unsqueeze(1),
                emotion_state=emotion_state,
                kv_cache=kv_cache,
                return_emotion=True,
            )
            kv_cache = outputs["kv_cache"]
            emotion_state = outputs.get("emotion_state")
            logits = outputs["logits"][:, -1, :]

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """Sample from logits with temperature, top-k, and top-p.

        Args:
            logits: (B, V)
            temperature: (B,) — per-batch temperature
            top_k: int
            top_p: float
            do_sample: bool

        Returns:
            token_ids: (B,)
        """
        # Apply temperature (per-batch)
        temp = temperature.unsqueeze(-1).clamp(min=0.01)  # (B, 1)
        logits = logits / temp

        if not do_sample:
            return logits.argmax(dim=-1)

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            kth_val = logits.topk(top_k, dim=-1).values[:, -1:]
            logits = logits.where(logits >= kth_val, torch.full_like(logits, float("-inf")))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative probability above the threshold
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")

            # Unsort
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
