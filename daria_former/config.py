from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Literal

import yaml


@dataclass
class DariaFormerConfig:
    """Full configuration for a Daria-Former model."""

    # ── Vocabulary & Embedding ──────────────────────────────────────────
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64  # if 0, computed as hidden_dim // num_heads

    # ── Positional / Context ────────────────────────────────────────────
    max_seq_len: int = 8192
    rope_base: float = 10000.0
    rope_scaling_factor: float = 1.0  # >1 enables dynamic NTK scaling
    sliding_window_size: int = 1024
    num_global_tokens: int = 64

    # ── FFN ──────────────────────────────────────────────────────────────
    ffn_hidden_dim: int = 0  # if 0, defaults to 4 * hidden_dim
    ffn_activation: Literal["swiglu", "gelu", "relu"] = "swiglu"

    # ── Memory ───────────────────────────────────────────────────────────
    working_memory_slots: int = 512
    episodic_memory_slots: int = 256
    persistent_memory_slots: int = 128
    persona_memory_slots: int = 64
    memory_key_dim: int = 0  # defaults to head_dim
    memory_top_k: int = 32

    # ── Emotion ──────────────────────────────────────────────────────────
    emotion_dim: int = 64
    emotion_categories: int = 8

    # ── Persona ──────────────────────────────────────────────────────────
    persona_dim: int = 64

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_rank: int = 0  # 0 = LoRA disabled
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # ── Modulation ───────────────────────────────────────────────────────
    rhythm_noise_std: float = 0.01
    variability_ngram_size: int = 4
    variability_penalty_weight: float = 0.1

    # ── Multimodal Encoders ──────────────────────────────────────────────
    image_enabled: bool = False
    image_patch_size: int = 16
    image_channels: int = 3
    image_size: int = 224

    audio_enabled: bool = False
    audio_n_mels: int = 80
    audio_max_len: int = 3000  # max mel-spec frames
    audio_conv_channels: int = 256

    # ── Training ─────────────────────────────────────────────────────────
    dropout: float = 0.0
    attention_dropout: float = 0.0
    weight_tying: bool = True
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False

    # ── Loss weights ─────────────────────────────────────────────────────
    emotion_loss_weight: float = 0.1
    repetition_loss_weight: float = 0.05
    rhythm_loss_weight: float = 0.05
    memory_loss_weight: float = 0.05

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.ffn_hidden_dim == 0:
            self.ffn_hidden_dim = 4 * self.hidden_dim
        if self.memory_key_dim == 0:
            self.memory_key_dim = self.head_dim

        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )

    # ── Serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> DariaFormerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> DariaFormerConfig:
        path = Path(path)
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                d = yaml.safe_load(f)
            else:
                d = json.load(f)
        return cls.from_dict(d)

    # ── Presets ──────────────────────────────────────────────────────────
    @classmethod
    def small(cls) -> DariaFormerConfig:
        """~125M parameters."""
        return cls(
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            max_seq_len=8192,
        )

    @classmethod
    def base(cls) -> DariaFormerConfig:
        """~350M parameters."""
        return cls(
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            head_dim=64,
            max_seq_len=16384,
        )

    @classmethod
    def large(cls) -> DariaFormerConfig:
        """~1.3B parameters."""
        return cls(
            hidden_dim=2048,
            num_layers=24,
            num_heads=32,
            head_dim=64,
            max_seq_len=32768,
        )

    @property
    def num_parameters_estimate(self) -> int:
        """Rough parameter estimate (embedding + transformer layers)."""
        emb = self.vocab_size * self.hidden_dim
        attn_per_layer = 4 * self.hidden_dim * self.hidden_dim  # Q/K/V/O
        ffn_per_layer = 2 * self.hidden_dim * self.ffn_hidden_dim
        if self.ffn_activation == "swiglu":
            ffn_per_layer = 3 * self.hidden_dim * self.ffn_hidden_dim
        layer_total = attn_per_layer + ffn_per_layer
        return emb + self.num_layers * layer_total
