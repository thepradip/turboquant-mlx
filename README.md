# TurboQuant

KV-cache compression for LLM inference on Apple Silicon (MLX).

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

## Install

```bash
pip install mlx mlx-lm scipy
pip install git+https://github.com/thepradip/turboquant-m2.git
```

## Usage

```python
import mlx_lm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from turboquant import compress_cache

# Load any HuggingFace model
model, tok = mlx_lm.load("Qwen/Qwen2.5-1.5B-Instruct")

# Prefill
text = tok.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — one line, auto-detects everything
result = compress_cache(cache, model=model, bits=4)
print(result)

# Generate with compressed cache
y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(200):
    logits = model(y.reshape(1, -1), cache=cache)
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tok.eos_token_id:
        break
    tokens.append(y.item())
print(tok.decode(tokens))
```

## Tested Models (all on M2 Pro 16GB)

### Qwen 3.5 2B — 5/5 correct (24L, 6 compressed, head_dim=256)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math (15*7) | 105 | 105 | PASS |
| QA (TCP) | Correct | Word-for-word identical | PASS |
| Code (add function) | Correct | Correct | PASS |
| Reasoning (logic) | "No" + correct reason | "No" + correct reason | PASS |
| Long QA (attention) | Formula + history | Formula + history | PASS |

### Gemma 3 4B 4-bit — 5/5 correct (34L, 34 compressed, head_dim=320)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math (15*7) | 105 | 105 | PASS |
| QA (TCP) | Correct | Correct | PASS |
| Code (add function) | Correct | Correct | PASS |
| Reasoning (logic) | "No" + explanation | "No" + explanation | PASS |
| Long QA (attention) | Formula + history | Formula + history | PASS |

### Qwen 2.5 1.5B — 4/5 (28L, 28 compressed, head_dim=128)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math | Correct | Correct | PASS |
| QA | Correct | Correct | PASS |
| Reasoning | Correct | Correct | PASS |
| Code generation | Correct | Starts OK, degrades | PARTIAL |

### Known limitations

- Code generation degrades on models where all layers are compressed (Qwen2.5)
- Hybrid models (Qwen3.5) work best — only full-attention layers are compressed
- QJL residual correction (Stage 2) is written but not integrated yet — needed for full reliability

## Architecture

```
src/turboquant/
├── __init__.py       # compress_cache, get_model_config
├── patch.py          # compress_cache() — compress KV in-place
├── compressor.py     # Stage 1: PolarQuant (rotation + Lloyd-Max)
├── codebook.py       # Lloyd-Max codebook builder
├── qjl.py           # Stage 2: QJL residual correction (written, not integrated)
├── attention.py      # Custom attention with QJL (written, not integrated)
└── cache.py          # TurboQuant cache structure (written, not integrated)
```
