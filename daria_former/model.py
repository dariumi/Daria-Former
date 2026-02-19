"""DariaFormerModel — top-level model assembling all components.

Pipeline:
    Input Encoders → Context Router → Memory Integration → N x DAC Blocks → Output Head

ESV is updated across layers for continuous emotion modulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from daria_former.config import DariaFormerConfig
from daria_former.core.positional import RotaryEmbedding
from daria_former.core.dac_block import DACBlock
from daria_former.context.context_router import ContextRouter
from daria_former.memory.memory_system import MemoryIntegrationSystem
from daria_former.modulation.emotion import EmotionExpressionLayer
from daria_former.modulation.rhythm import ConversationRhythmLayer
from daria_former.modulation.variability import ReactionVariabilitySystem
from daria_former.encoders.text_encoder import TextEncoder
from daria_former.encoders.image_encoder import ImageEncoder
from daria_former.encoders.audio_encoder import AudioEncoder


class DariaFormerModel(nn.Module):
    """Daria-Former: Autoregressive Multi-Context Transformer.

    Architectural formula (per step t)::

        h_t = DAC(x_t, LocalContext, LongContext, Memory_t, Persona, ESV_t)
        ESV_t = EmotionUpdate(h_t, ESV_{t-1})
        y_t = Softmax(Projection(h_t))
    """

    def __init__(self, config: DariaFormerConfig):
        super().__init__()
        self.config = config

        # ── Input Encoders ──────────────────────────────────────────────
        self.text_encoder = TextEncoder(config)

        self.image_encoder: Optional[ImageEncoder] = None
        if config.image_enabled:
            self.image_encoder = ImageEncoder(config)

        self.audio_encoder: Optional[AudioEncoder] = None
        if config.audio_enabled:
            self.audio_encoder = AudioEncoder(config)

        # ── Positional Encoding ─────────────────────────────────────────
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling_factor=config.rope_scaling_factor,
        )

        # ── Context Router ──────────────────────────────────────────────
        self.context_router = ContextRouter(
            hidden_dim=config.hidden_dim,
            sliding_window_size=config.sliding_window_size,
            num_global_tokens=config.num_global_tokens,
        )

        # ── Memory System ───────────────────────────────────────────────
        self.memory = MemoryIntegrationSystem(config)

        # ── Emotion System ──────────────────────────────────────────────
        self.emotion = EmotionExpressionLayer(
            hidden_dim=config.hidden_dim,
            emotion_dim=config.emotion_dim,
            num_categories=config.emotion_categories,
        )

        # ── DAC Blocks ──────────────────────────────────────────────────
        has_modality = config.image_enabled or config.audio_enabled
        self.layers = nn.ModuleList([
            DACBlock(config, has_modality=has_modality)
            for _ in range(config.num_layers)
        ])

        # ── Post-transformer ────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.rhythm = ConversationRhythmLayer(
            hidden_dim=config.hidden_dim,
            noise_std=config.rhythm_noise_std,
        )

        # ── Output Head ─────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.lm_head.weight = self.text_encoder.weight

        # ── Variability system (used during generation) ─────────────────
        self.variability = ReactionVariabilitySystem(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            ngram_size=config.variability_ngram_size,
            penalty_weight=config.variability_penalty_weight,
        )

        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        emotion_state: torch.Tensor | None = None,
        kv_cache: Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_emotion: bool = False,
    ) -> dict:
        """
        Args:
            input_ids: (B, S) — token IDs
            images: (B, C, H, W) or None
            audio: (B, n_mels, T) or None
            emotion_state: (B, emotion_dim) or None (initial ESV)
            kv_cache: list of per-layer KV caches (for generation)
            return_emotion: whether to return emotion state and logits

        Returns:
            dict with:
                "logits": (B, S, V)
                "emotion_state": (B, emotion_dim) — if return_emotion
                "emotion_logits": (B, num_categories) — if return_emotion
                "kv_cache": list of per-layer KV caches
                "hidden_states": (B, S, H) — last hidden state
        """
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.lm_head.weight.dtype

        # ── Encode inputs ───────────────────────────────────────────────
        x = self.text_encoder(input_ids)  # (B, S, H)

        # Multimodal tokens
        modality_kv = None
        modality_tokens = []
        if self.image_encoder is not None and images is not None:
            img_tokens = self.image_encoder(images)  # (B, P, H)
            modality_tokens.append(img_tokens)
        if self.audio_encoder is not None and audio is not None:
            aud_tokens = self.audio_encoder(audio)  # (B, T', H)
            modality_tokens.append(aud_tokens)
        if modality_tokens:
            modality_kv = torch.cat(modality_tokens, dim=1)  # (B, M, H)

        # ── RoPE ────────────────────────────────────────────────────────
        offset = kv_cache[0]["local"][0].shape[2] if kv_cache and "local" in kv_cache[0] else 0
        cos, sin = self.rope(S, offset=offset)

        # ── Memory ──────────────────────────────────────────────────────
        memory_kv = self.memory.get_memory_kv(x)        # (B, M_mem, H)
        persona_kv = self.memory.get_persona_kv(B)       # (B, M_per, H)

        # ── Emotion state init ──────────────────────────────────────────
        esv = emotion_state
        if esv is None:
            esv = self.emotion.init_state(B, device, dtype)

        # ── DAC layers ──────────────────────────────────────────────────
        new_kv_cache = []
        emotion_logits = None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache else None

            if self.config.gradient_checkpointing and self.training:
                def create_forward(layer_module, layer_cache_inner):
                    def custom_forward(x_inner, cos_inner, sin_inner, esv_inner, mem, pers, mod):
                        return layer_module(
                            x_inner,
                            cos=cos_inner, sin=sin_inner,
                            esv=esv_inner,
                            memory_kv=mem,
                            persona_kv=pers,
                            modality_kv=mod,
                            kv_cache=layer_cache_inner,
                            emotion_scale=1.0,
                        )
                    return custom_forward

                x, cache_i = gradient_checkpoint(
                    create_forward(layer, layer_cache),
                    x, cos, sin, esv, memory_kv, persona_kv, modality_kv,
                    use_reentrant=False,
                )
            else:
                x, cache_i = layer(
                    x,
                    cos=cos, sin=sin,
                    esv=esv,
                    memory_kv=memory_kv,
                    persona_kv=persona_kv,
                    modality_kv=modality_kv,
                    kv_cache=layer_cache,
                    emotion_scale=1.0,
                )

            new_kv_cache.append(cache_i)

            # Update emotion state every few layers (or every layer)
            if (i + 1) % max(1, self.config.num_layers // 4) == 0 or i == self.config.num_layers - 1:
                esv, attn_scale, temp_mod, emotion_logits = self.emotion(x, esv)

        # ── Post-processing ─────────────────────────────────────────────
        hidden_states = self.final_norm(x)
        hidden_states = self.rhythm(hidden_states, training=self.training)

        # ── Output head ─────────────────────────────────────────────────
        logits = self.lm_head(hidden_states)  # (B, S, V)

        # ── Update working memory ───────────────────────────────────────
        if self.training:
            self.memory.update_working_memory(hidden_states.detach())

        # ── Result ──────────────────────────────────────────────────────
        result = {
            "logits": logits,
            "kv_cache": new_kv_cache,
            "hidden_states": hidden_states,
        }
        if return_emotion:
            result["emotion_state"] = esv
            result["emotion_logits"] = emotion_logits

        return result

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
