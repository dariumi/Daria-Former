"""EmotionExpressionLayer — dynamic latent emotion module.

ESV_t = f(ESV_{t-1}, hidden_state_t, context_features)

The Emotion State Vector (ESV) is updated at every autoregressive step
and modulates: attention scaling, FFN bias, sampling temperature,
and lexical style.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionExpressionLayer(nn.Module):
    """Computes and updates the Emotion State Vector (ESV).

    Uses a GRU-like gating mechanism for continuous, context-dependent
    emotion state updates.

    Outputs:
        - Updated ESV
        - Attention emotion scale
        - Temperature modulation factor
        - Emotion category logits (for auxiliary loss)
    """

    def __init__(
        self,
        hidden_dim: int,
        emotion_dim: int,
        num_categories: int = 8,
    ):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.num_categories = num_categories

        # GRU-like update for ESV (input: context_proj output + prev_esv = 2*E)
        self.update_gate = nn.Linear(emotion_dim * 2, emotion_dim)
        self.reset_gate = nn.Linear(emotion_dim * 2, emotion_dim)
        self.candidate = nn.Linear(emotion_dim * 2, emotion_dim)

        # Context feature extraction
        self.context_proj = nn.Linear(hidden_dim, emotion_dim)

        # Modulation outputs
        self.attention_scale_proj = nn.Linear(emotion_dim, 1)
        self.temperature_proj = nn.Linear(emotion_dim, 1)

        # Emotion classification head (for auxiliary loss)
        self.category_head = nn.Linear(emotion_dim, num_categories)

        # Style modulation vector (affects hidden state via additive bias)
        self.style_proj = nn.Linear(emotion_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize so that attention scale starts near 1.0
        nn.init.zeros_(self.attention_scale_proj.weight)
        nn.init.zeros_(self.attention_scale_proj.bias)
        # Temperature starts near 1.0
        nn.init.zeros_(self.temperature_proj.weight)
        nn.init.zeros_(self.temperature_proj.bias)
        # Style starts near zero
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize ESV to zeros."""
        return torch.zeros(batch_size, self.emotion_dim, device=device, dtype=dtype)

    def forward(
        self,
        hidden_state: torch.Tensor,
        prev_esv: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
        """Update ESV and compute modulation factors.

        Args:
            hidden_state: (B, S, H) — current layer hidden states
            prev_esv: (B, emotion_dim) — previous emotion state

        Returns:
            new_esv: (B, emotion_dim)
            attention_scale: scalar
            temperature_mod: scalar
            category_logits: (B, num_categories)
        """
        # Aggregate hidden state across sequence
        h_agg = hidden_state.mean(dim=1)  # (B, H)
        context = self.context_proj(h_agg)  # (B, E)

        # GRU-like update
        combined = torch.cat([context, prev_esv], dim=-1)  # (B, 2*E)
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        candidate_input = torch.cat([context, r * prev_esv], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input))

        new_esv = (1 - z) * prev_esv + z * candidate

        # Compute modulation factors
        attention_scale = 1.0 + torch.tanh(self.attention_scale_proj(new_esv)).mean().item()
        temperature_mod = 1.0 + 0.5 * torch.tanh(self.temperature_proj(new_esv)).mean().item()

        # Category logits
        category_logits = self.category_head(new_esv)

        return new_esv, attention_scale, temperature_mod, category_logits

    def get_style_bias(self, esv: torch.Tensor) -> torch.Tensor:
        """Get style modulation vector for hidden state bias.

        Args:
            esv: (B, emotion_dim)
        Returns:
            (B, 1, hidden_dim) — additive bias for hidden states
        """
        return self.style_proj(esv).unsqueeze(1)
