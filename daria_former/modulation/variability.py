"""ReactionVariabilitySystem — reduces repetition and template patterns during inference.

Uses token similarity penalty, semantic redundancy detection,
dynamic temperature control, and anti-template bias.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionVariabilitySystem(nn.Module):
    """Inference-time module that detects and penalizes repetitive patterns.

    Components:
        1. Token similarity penalty — penalizes cosine-similar consecutive tokens
        2. Semantic redundancy detector — n-gram repetition detection
        3. Dynamic temperature control — raises temperature when repetition detected
        4. Anti-template bias layer — penalizes frequent token patterns
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        ngram_size: int = 4,
        penalty_weight: float = 0.1,
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.penalty_weight = penalty_weight
        self.vocab_size = vocab_size

        # Anti-template bias (learned, can be fine-tuned)
        self.anti_template_bias = nn.Parameter(torch.zeros(vocab_size))

        # Similarity penalty projection
        self.sim_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def compute_similarity_penalty(
        self,
        current_hidden: torch.Tensor,
        prev_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cosine similarity penalty between consecutive hidden states.

        Args:
            current_hidden: (B, H)
            prev_hidden: (B, H) or None

        Returns:
            penalty: (B,) — scalar penalty per batch element
        """
        if prev_hidden is None:
            return torch.zeros(current_hidden.shape[0], device=current_hidden.device)

        h1 = self.sim_proj(current_hidden)
        h2 = self.sim_proj(prev_hidden)
        sim = F.cosine_similarity(h1, h2, dim=-1)
        return torch.clamp(sim, min=0.0) * self.penalty_weight

    def detect_ngram_repetition(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Detect n-gram repetitions in generated sequence.

        Args:
            token_ids: (B, S) — generated token IDs so far

        Returns:
            repetition_score: (B,) — 0 to 1, higher = more repetitive
        """
        B, S = token_ids.shape
        if S < self.ngram_size * 2:
            return torch.zeros(B, device=token_ids.device)

        scores = torch.zeros(B, device=token_ids.device)
        for b in range(B):
            seen = set()
            repeats = 0
            total = 0
            seq = token_ids[b].tolist()
            for i in range(len(seq) - self.ngram_size + 1):
                ngram = tuple(seq[i:i + self.ngram_size])
                total += 1
                if ngram in seen:
                    repeats += 1
                seen.add(ngram)
            scores[b] = repeats / max(total, 1)

        return scores

    def compute_dynamic_temperature(
        self,
        base_temperature: float,
        repetition_score: torch.Tensor,
        emotion_temp_mod: float = 1.0,
    ) -> torch.Tensor:
        """Adjust temperature based on repetition score.

        Args:
            base_temperature: base sampling temperature
            repetition_score: (B,)
            emotion_temp_mod: emotion-based temperature modifier

        Returns:
            adjusted_temperature: (B,)
        """
        # Raise temperature when repetition is detected
        temp_boost = 1.0 + repetition_score * 0.5
        return base_temperature * temp_boost * emotion_temp_mod

    def apply_anti_template_bias(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply anti-template bias to logits.

        Args:
            logits: (B, V)
            generated_ids: (B, S) — previously generated tokens

        Returns:
            adjusted logits: (B, V)
        """
        # Apply learned anti-template bias
        logits = logits + self.anti_template_bias.unsqueeze(0)

        # Frequency-based penalty
        if generated_ids is not None and generated_ids.shape[1] > 0:
            B, V = logits.shape
            freq = torch.zeros(B, V, device=logits.device)
            for b in range(B):
                ids, counts = generated_ids[b].unique(return_counts=True)
                ids = ids[ids < V]
                counts = counts[:ids.shape[0]]
                freq[b, ids] = counts.float()
            # Penalize tokens proportional to their frequency
            freq_penalty = (freq / (freq.sum(dim=-1, keepdim=True) + 1e-8)) * self.penalty_weight
            logits = logits - freq_penalty

        return logits

    def forward(
        self,
        logits: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_hidden: torch.Tensor | None = None,
        generated_ids: torch.Tensor | None = None,
        base_temperature: float = 1.0,
        emotion_temp_mod: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full variability pipeline.

        Args:
            logits: (B, V) — raw output logits for current position
            hidden_state: (B, H) — current hidden state
            prev_hidden: (B, H) — previous step hidden state
            generated_ids: (B, S) — previously generated token IDs
            base_temperature: base sampling temperature
            emotion_temp_mod: emotion-based temperature modifier

        Returns:
            adjusted_logits: (B, V)
            temperature: (B,)
        """
        B = logits.shape[0]

        # Similarity penalty
        sim_penalty = self.compute_similarity_penalty(hidden_state, prev_hidden)

        # N-gram repetition detection
        rep_score = torch.zeros(B, device=logits.device)
        if generated_ids is not None:
            rep_score = self.detect_ngram_repetition(generated_ids)

        # Dynamic temperature
        temperature = self.compute_dynamic_temperature(
            base_temperature, rep_score + sim_penalty, emotion_temp_mod,
        )

        # Anti-template bias
        logits = self.apply_anti_template_bias(logits, generated_ids)

        return logits, temperature
