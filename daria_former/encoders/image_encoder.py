"""Image Encoder — patch embedding adapter for vision inputs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from daria_former.config import DariaFormerConfig


class ImageEncoder(nn.Module):
    """Converts images into a sequence of embeddings in the model's latent space.

    Uses a simple patch embedding approach (similar to ViT):
        image → non-overlapping patches → linear projection → hidden_dim

    Can also serve as an adapter for external vision encoders
    (e.g., CLIP) by projecting their output to hidden_dim.
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        self.patch_size = config.image_patch_size
        self.image_size = config.image_size
        self.hidden_dim = config.hidden_dim

        num_patches = (config.image_size // config.image_patch_size) ** 2
        patch_dim = config.image_channels * config.image_patch_size ** 2

        # Patch embedding via Conv2d (equivalent to linear on flattened patches)
        self.patch_embed = nn.Conv2d(
            in_channels=config.image_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.image_patch_size,
            stride=config.image_patch_size,
            bias=True,
        )

        # Learnable position embeddings for patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, config.hidden_dim) * 0.02
        )

        # LayerNorm after projection
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Optional projection for external encoder features
        self.external_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) — raw images

        Returns:
            image_tokens: (B, num_patches, hidden_dim)
        """
        # Patch embed: (B, C, H, W) → (B, hidden_dim, H', W')
        x = self.patch_embed(images)
        # Flatten spatial dims: (B, hidden_dim, H', W') → (B, num_patches, hidden_dim)
        x = rearrange(x, "b d h w -> b (h w) d")

        # Add position embeddings
        x = x + self.position_embedding[:, :x.shape[1], :]

        return self.norm(x)

    def from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project pre-extracted features (e.g., from CLIP) into latent space.

        Args:
            features: (B, num_tokens, feature_dim) — external encoder output

        Returns:
            (B, num_tokens, hidden_dim)
        """
        return self.norm(self.external_proj(features))
