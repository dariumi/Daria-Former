# Daria-Former

**Custom Autoregressive Multi-Context Transformer Framework**

Daria-Former — универсальный авторегрессионный каркас с динамической эмоциональной модуляцией, асинхронной многоуровневой памятью и нативной поддержкой LoRA. Это архитектурный стандарт, а не конкретная обученная модель.

## Architecture

```
Input Encoding → Context Router → Memory Integration → Daria Autoregressive Core → Dynamic Modulation → Output Head
```

### Core Pipeline (DAC Block)

```
AdaptiveNorm(x, ESV) → Multi-Context Attention → Emotion-FFN → Residual Fusion → Rhythm Gate → Residual
```

### Key Components

| Component | Description |
|---|---|
| **Multi-Context Attention (MCA)** | 5 параллельных attention-путей: Local (sliding window), Long (full causal), Memory, Persona, Modality — с dynamic gating |
| **Memory Integration System** | 4-уровневая память: Working, Episodic, Persistent, Persona — подключается как attention-источник, а не текст в prompt |
| **EmotionExpressionLayer** | Латентный GRU-модуль, ESV обновляется на каждом шаге авторегрессии и влияет на attention, FFN, temperature, стиль |
| **ConversationRhythmLayer** | Контроль ритма: variance projection, entropy regulation, structural noise injection |
| **ReactionVariabilitySystem** | Подавление шаблонности: n-gram detection, similarity penalty, dynamic temperature, anti-template bias |
| **Native LoRA** | Встроенная в архитектуру: multi-LoRA stacking, runtime switching, Persona-LoRA, Emotion-LoRA |

### Architectural Formula

```
h_t = DAC(x_t, LocalContext, LongContext, Memory_t, Persona, ESV_t)
ESV_t = EmotionUpdate(h_t, ESV_{t-1})
y_t = Softmax(Projection(h_t))
```

## Installation

```bash
git clone https://github.com/your-username/Daria-Former.git
cd Daria-Former
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional dependencies

```bash
pip install -e ".[wandb]"       # Weights & Biases logging
pip install -e ".[flash]"       # FlashAttention support
pip install torchaudio torchvision  # Multimodal encoders
```

## Quick Start

### Create a model

```python
from daria_former import DariaFormerConfig, DariaFormerModel

config = DariaFormerConfig.small()  # ~125M params
model = DariaFormerModel(config)

print(f"Parameters: {model.count_parameters():,}")
```

### Forward pass

```python
import torch

input_ids = torch.randint(0, config.vocab_size, (1, 512))
outputs = model(input_ids, return_emotion=True)

logits = outputs["logits"]           # (1, 512, vocab_size)
emotion_state = outputs["emotion_state"]  # (1, emotion_dim)
```

### With multimodal inputs

```python
config = DariaFormerConfig.small()
config.image_enabled = True
config.audio_enabled = True
model = DariaFormerModel(config)

outputs = model(
    input_ids=torch.randint(0, 256, (1, 128)),
    images=torch.randn(1, 3, 224, 224),
    audio=torch.randn(1, 80, 1000),
)
```

### LoRA injection

```python
from daria_former.lora import LoRAManager

manager = LoRAManager(model, rank=16, alpha=32, targets=["q_proj", "v_proj"])
manager.inject()
manager.freeze_base()

# Add persona adapter
manager.add_adapter("persona_style", rank=8)
manager.activate_adapter("persona_style")

# Runtime switching
manager.deactivate_adapter("persona_style")
manager.activate_adapter("emotion_intense", weight=0.5)
```

### Generation

```python
from daria_former.generation import DariaGenerator

generator = DariaGenerator(model)
output_ids = generator.generate(
    input_ids=torch.tensor([[1, 2, 3]]),
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

## Training

### CLI

```bash
daria-train \
    --config configs/daria_former_small.yaml \
    --train_data data/train.txt \
    --eval_data data/eval.txt \
    --tokenizer gpt2 \
    --lr 1e-4 \
    --batch_size 4 \
    --max_steps 100000 \
    --warmup_steps 1000 \
    --save_dir checkpoints \
    --use_amp \
    --wandb
```

### Distributed training

```bash
torchrun --nproc_per_node=4 -m cli.train \
    --config configs/daria_former_base.yaml \
    --train_data data/train.txt \
    --lr 3e-4 \
    --batch_size 2 \
    --grad_accum 8
```

### Python API

```python
from daria_former.training import Trainer, DariaFormerLoss, build_optimizer

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    lr=1e-4,
    max_steps=50000,
    use_amp=True,
)
trainer.train()
```

### Loss function

Base: `CrossEntropy(next_token)`

Auxiliary losses (configurable weights):
- **EmotionAlignmentLoss** — ESV smoothness + optional label alignment
- **RepetitionPenaltyLoss** — cosine similarity between consecutive hidden states
- **RhythmRegularizer** — entropy variance across positions
- **MemoryConsistencyLoss** — memory retrieval alignment

## Generation / Inference

```bash
daria-generate \
    --checkpoint checkpoints/checkpoint-10000 \
    --tokenizer gpt2 \
    --prompt "Hello, world" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.9
```

## Export

```bash
daria-export \
    --checkpoint checkpoints/checkpoint-10000 \
    --output exported_model \
    --merge_lora \
    --half
```

## Model Sizes

| Variant | Layers | Hidden | Heads | Context | ~Params |
|---------|--------|--------|-------|---------|---------|
| Small   | 12     | 768    | 12    | 8K      | 125M    |
| Base    | 24     | 1024   | 16    | 16K     | 350M    |
| Large   | 24     | 2048   | 32    | 32K     | 1.3B    |

The architecture scales to 70B+ via pipeline/tensor parallelism and MoE-compatible expansion.

## Project Structure

```
Daria-Former/
├── daria_former/
│   ├── config.py                  # DariaFormerConfig
│   ├── model.py                   # DariaFormerModel
│   ├── core/                      # DAC block, MCA, RoPE, FFN, norms, gates
│   ├── memory/                    # Memory banks, retrieval, MIS
│   ├── context/                   # Context router
│   ├── modulation/                # Emotion, rhythm, variability
│   ├── lora/                      # LoRA linear, manager
│   ├── encoders/                  # Text, image, audio encoders
│   ├── generation/                # Autoregressive generator
│   ├── training/                  # Trainer, losses, optimizer, scheduler
│   ├── data/                      # Datasets, tokenizer, collator
│   └── utils/                     # Logging, checkpoints
├── cli/                           # CLI entry points
├── configs/                       # YAML model configs
└── tests/                         # Unit & integration tests
```

## Optimization Features

- FlashAttention compatible interface
- Quantization friendly (4bit/8bit via standard PyTorch quantization)
- KV-cache for efficient autoregressive generation
- Gradient checkpointing support
- Mixed precision training (AMP)
- Speculative decoding compatible
- CPU inference support (small model variants)

## Tests

```bash
pytest tests/ -v
```

78 tests covering all components: attention, config, DAC blocks, emotion, encoders, generation, LoRA, memory, full model forward/backward, training pipeline.

## Architectural Principles

1. **Autoregression is fundamental** — pure next-token prediction as the base
2. **Multi-level asynchronous context** — local, long, memory, persona, modality
3. **Emotion as latent dynamic variable** — continuous, context-dependent, updated every step
4. **Memory as attention source** — not text in prompt, but separate KV cross-attention
5. **Anti-template architecture** — repetition suppressed at the model level
6. **Native LoRA** — built into the architecture, not bolted on
7. **Unlimited scale** — from 125M to 70B+ parameters

## License

Apache License 2.0  — see [LICENSE](LICENSE).
