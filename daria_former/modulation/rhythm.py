"""ConversationRhythmLayer — rhythmic stabilization for output diversity.

Controls phrase length, pauses, structure variability, and reduces
monotonous patterns through variance projection, entropy regulation,
and controlled structural noise injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConversationRhythmLayer(nn.Module):
    """Rhythmic stabilization module.

    Components:
        1. Variance Projection Head — projects hidden states to rhythm features
        2. Entropy Regulator — monitors and adjusts output entropy
        3. Structural Noise Injection — controlled noise for variability
    """

    def __init__(
        self,
        hidden_dim: int,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.noise_std = noise_std

        # Variance Projection Head
        self.variance_proj = nn.Linear(hidden_dim, hidden_dim)

        # Entropy regulator: predicts target entropy, modulates accordingly
        self.entropy_predictor = nn.Linear(hidden_dim, 1)
        self.entropy_scale = nn.Linear(1, hidden_dim)

        # Noise injection gate (learned, per-position)
        self.noise_gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.noise_gate.weight)
        nn.init.constant_(self.noise_gate.bias, -2.0)  # start mostly closed

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, H)
            training: whether to inject noise

        Returns:
            modulated x: (B, S, H)
        """
        # Variance features
        var_features = self.variance_proj(x)  # (B, S, H)

        # Compute per-position entropy estimate
        entropy_est = torch.sigmoid(self.entropy_predictor(x))  # (B, S, 1)
        entropy_mod = self.entropy_scale(entropy_est)  # (B, S, H)

        # Modulate via variance and entropy
        x = x + var_features * entropy_mod * 0.1

        # Structural noise injection (training only)
        if training and self.noise_std > 0:
            gate = torch.sigmoid(self.noise_gate(x))  # (B, S, H)
            noise = torch.randn_like(x) * self.noise_std
            x = x + gate * noise

        return x

    def compute_rhythm_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute rhythm regularization loss encouraging entropy diversity.

        Args:
            logits: (B, S, V) — output logits

        Returns:
            scalar loss
        """
        # Per-position entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, S)

        # Penalize low variance of entropy across positions
        # (we want diverse entropy → diverse output structure)
        entropy_var = entropy.var(dim=-1)  # (B,)
        loss = -entropy_var.mean()  # maximize variance → minimize negative

        return loss
