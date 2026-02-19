"""Daria Autoregressive Core (DAC) Block.

AdaptiveNorm → Multi-Context Attention → Emotion-FFN → Residual Fusion → Rhythm Gate → Residual
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from daria_former.config import DariaFormerConfig
from daria_former.core.adaptive_norm import AdaptiveNorm
from daria_former.core.multi_context_attention import MultiContextAttention
from daria_former.core.emotion_ffn import EmotionModulatedFFN
from daria_former.core.residual_fusion import ResidualFusion
from daria_former.core.rhythm_gate import RhythmGate


class DACBlock(nn.Module):
    """A single Daria Autoregressive Core block.

    Pipeline::

        residual = x
        x = AdaptiveNorm(x, esv)
        attn_out = MCA(x, ...)
        ffn_out  = EmotionFFN(attn_out + residual, esv)
        fused    = ResidualFusion(attn_out, ffn_out, residual)
        output   = RhythmGate(fused) + residual
    """

    def __init__(self, config: DariaFormerConfig, has_modality: bool = False):
        super().__init__()
        self.config = config

        self.norm = AdaptiveNorm(config.hidden_dim, config.emotion_dim)

        self.attention = MultiContextAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            sliding_window_size=config.sliding_window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            modality_attention=has_modality,
        )

        self.ffn_norm = AdaptiveNorm(config.hidden_dim, config.emotion_dim)

        self.ffn = EmotionModulatedFFN(
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            emotion_dim=config.emotion_dim,
            activation=config.ffn_activation,
            dropout=config.dropout,
        )

        self.residual_fusion = ResidualFusion(config.hidden_dim)
        self.rhythm_gate = RhythmGate(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        esv: torch.Tensor | None = None,
        memory_kv: torch.Tensor | None = None,
        persona_kv: torch.Tensor | None = None,
        modality_kv: torch.Tensor | None = None,
        kv_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        emotion_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            output: (B, S, H)
            new_kv_cache: dict of KV caches per attention pathway
        """
        residual = x

        # Pre-norm with emotion conditioning
        h = self.norm(x, esv)

        # Multi-Context Attention
        attn_out, new_cache = self.attention(
            h,
            cos=cos,
            sin=sin,
            memory_kv=memory_kv,
            persona_kv=persona_kv,
            modality_kv=modality_kv,
            kv_cache=kv_cache,
            emotion_scale=emotion_scale,
        )

        # Emotion-Modulated FFN (with pre-norm)
        ffn_input = self.ffn_norm(attn_out + residual, esv)
        ffn_out = self.ffn(ffn_input, esv)

        # Residual Fusion: merge attn, ffn, skip
        fused = self.residual_fusion(attn_out, ffn_out, residual)

        # Rhythm Gate
        output = self.rhythm_gate(fused) + residual

        return output, new_cache
