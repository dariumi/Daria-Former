"""Audio Encoder — mel-spectrogram convolutional adapter."""

from __future__ import annotations

import torch
import torch.nn as nn

from daria_former.config import DariaFormerConfig


class AudioEncoder(nn.Module):
    """Converts audio mel-spectrograms into a sequence of embeddings.

    Architecture: mel-spec → 2x Conv1d → linear projection → hidden_dim

    Accepts pre-computed mel-spectrograms as input.
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        n_mels = config.audio_n_mels
        conv_ch = config.audio_conv_channels

        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, conv_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.proj = nn.Linear(conv_ch, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Learnable position embeddings
        max_tokens = config.audio_max_len // 2 + 1  # after stride-2 conv
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_tokens, config.hidden_dim) * 0.02
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: (B, n_mels, T) — mel-spectrogram

        Returns:
            audio_tokens: (B, T', hidden_dim) where T' ≈ T/2
        """
        # Conv layers: (B, n_mels, T) → (B, conv_ch, T')
        x = self.conv_layers(mel_spec)

        # Transpose and project: (B, T', conv_ch) → (B, T', hidden_dim)
        x = x.transpose(1, 2)
        x = self.proj(x)

        # Add position embeddings
        T_prime = x.shape[1]
        x = x + self.position_embedding[:, :T_prime, :]

        return self.norm(x)
