"""Composite loss for Daria-Former training.

Main loss: CrossEntropy (next token prediction)
Auxiliary losses:
    - Emotion alignment
    - Repetition penalty
    - Rhythm regularization
    - Memory consistency
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from daria_former.config import DariaFormerConfig


class EmotionAlignmentLoss(nn.Module):
    """Encourages emotion state consistency across adjacent steps.

    Penalizes abrupt ESV changes (smoothness) and optionally
    aligns with emotion labels if provided.
    """

    def forward(
        self,
        emotion_logits: torch.Tensor | None,
        emotion_labels: torch.Tensor | None = None,
        prev_esv: torch.Tensor | None = None,
        curr_esv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=emotion_logits.device if emotion_logits is not None else "cpu")

        # Classification loss if labels provided
        if emotion_logits is not None and emotion_labels is not None:
            loss = loss + F.cross_entropy(emotion_logits, emotion_labels)

        # Smoothness: penalize large ESV jumps
        if prev_esv is not None and curr_esv is not None:
            smoothness = (curr_esv - prev_esv).pow(2).mean()
            loss = loss + 0.1 * smoothness

        return loss


class RepetitionPenaltyLoss(nn.Module):
    """Penalizes the model for producing repeated token patterns.

    Computes cosine similarity between consecutive hidden states
    and penalizes high similarity.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, H)
        """
        if hidden_states.shape[1] < 2:
            return torch.tensor(0.0, device=hidden_states.device)

        h1 = hidden_states[:, :-1, :]  # (B, S-1, H)
        h2 = hidden_states[:, 1:, :]   # (B, S-1, H)

        # Cosine similarity across hidden dim
        sim = F.cosine_similarity(h1, h2, dim=-1)  # (B, S-1)

        # Penalize high similarity (> 0.9)
        penalty = torch.clamp(sim - 0.9, min=0.0)
        return penalty.mean()


class RhythmRegularizer(nn.Module):
    """Encourages diverse output entropy across positions.

    Maximizes variance of per-position entropy to prevent
    monotonous output patterns.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, S, V)
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, S)

        # Maximize entropy variance → minimize negative variance
        entropy_var = entropy.var(dim=-1)  # (B,)
        return -entropy_var.mean()


class MemoryConsistencyLoss(nn.Module):
    """Ensures memory retrieval aligns with contextual needs.

    Penalizes low similarity between retrieved memory and
    hidden states that triggered the retrieval.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_kv: torch.Tensor | None,
    ) -> torch.Tensor:
        if memory_kv is None:
            return torch.tensor(0.0, device=hidden_states.device)

        # Average hidden state as query
        query = hidden_states.mean(dim=1)  # (B, H)
        mem_mean = memory_kv.mean(dim=1)   # (B, H)

        # Encourage alignment
        sim = F.cosine_similarity(query, mem_mean, dim=-1)
        return (1.0 - sim).mean()


class DariaFormerLoss(nn.Module):
    """Composite training loss for Daria-Former.

    loss = CE + w_emotion * emotion_loss + w_rep * rep_loss
         + w_rhythm * rhythm_loss + w_mem * memory_loss
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.emotion_loss = EmotionAlignmentLoss()
        self.repetition_loss = RepetitionPenaltyLoss()
        self.rhythm_loss = RhythmRegularizer()
        self.memory_loss = MemoryConsistencyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        emotion_logits: torch.Tensor | None = None,
        emotion_labels: torch.Tensor | None = None,
        prev_esv: torch.Tensor | None = None,
        curr_esv: torch.Tensor | None = None,
        memory_kv: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, S, V) — model output logits
            labels: (B, S) — target token IDs (-100 for padding)
            hidden_states: (B, S, H)
            emotion_logits: (B, num_categories) or None
            emotion_labels: (B,) or None
            prev_esv / curr_esv: (B, E) or None
            memory_kv: (B, M, H) or None

        Returns:
            dict: "loss" (total), "ce_loss", "emotion_loss", "repetition_loss",
                  "rhythm_loss", "memory_loss"
        """
        # Main CE loss: shift logits and labels for autoregressive
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )

        # Auxiliary losses
        emo = self.emotion_loss(emotion_logits, emotion_labels, prev_esv, curr_esv)
        rep = self.repetition_loss(hidden_states)
        rhy = self.rhythm_loss(logits)
        mem = self.memory_loss(hidden_states, memory_kv)

        total = (
            ce
            + self.config.emotion_loss_weight * emo
            + self.config.repetition_loss_weight * rep
            + self.config.rhythm_loss_weight * rhy
            + self.config.memory_loss_weight * mem
        )

        return {
            "loss": total,
            "ce_loss": ce,
            "emotion_loss": emo,
            "repetition_loss": rep,
            "rhythm_loss": rhy,
            "memory_loss": mem,
        }
