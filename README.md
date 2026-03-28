# TurboQuant

KV-cache compression for LLM inference on Apple Silicon (MLX).

Directly accesses and compresses the KV cache at code level — reads real key/value tensors from the model, compresses them with 4-bit Lloyd-Max quantization, writes them back. The model continues generating from the compressed cache.

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

## How it works

```
Model forward pass → KV cache stored as MLX arrays
                          ↓
          cache[layer].keys   = (batch, kv_heads, seq_len, head_dim)
          cache[layer].values = (batch, kv_heads, seq_len, head_dim)
                          ↓
      TurboQuant reads these tensors directly
                          ↓
      1. Normalize to unit sphere (store norms separately)
      2. Apply random orthogonal rotation
      3. Quantize each coordinate with Lloyd-Max optimal codebook
      4. Dequantize → inverse rotate → rescale by norms
      5. Write compressed values back into cache
                          ↓
      Model generates next tokens using compressed cache
```

No custom attention kernels. No model modification. Just compress the cache arrays in-place.

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

# Load any HuggingFace model via MLX
model, tok = mlx_lm.load("Qwen/Qwen3.5-2B")

# Prefill — fills the KV cache
text = tok.apply_chat_template(
    [{"role": "user", "content": "What is the attention formula?"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — one line, auto-detects model config
result = compress_cache(cache, model=model, bits=4)
print(result)
# {'cosine': 0.9954, 'layers_compressed': 6, 'ratio': 3.9, ...}

# Generate with compressed cache — standard MLX generation
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

## Tested Models

All results measured live on Apple M2 Pro 16GB. Every output verified correct.

### Qwen 3.5 2B — 5/5 (24L, 6 compressed, head_dim=256)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math (15×7) | 105 | 105 | PASS |
| QA (TCP) | Correct | Word-for-word identical | PASS |
| Code (add function) | Correct + tests | Correct + output | PASS |
| Logic reasoning | "No" + correct reason | "No" + correct reason | PASS |
| Long QA (attention) | Formula + history | Formula + history | PASS |

### Gemma 3 4B 4-bit — 5/5 (34L, 34 compressed, head_dim=320)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math (15×7) | 105 | 105 | PASS |
| QA (TCP) | Correct | Correct | PASS |
| Code (add function) | Correct + docstring | Correct + docstring | PASS |
| Logic reasoning | "No" + explanation | "No" + explanation | PASS |
| Long QA (attention) | Formula + history | Formula + history | PASS |

### Qwen 2.5 1.5B — 4/5 (28L, 28 compressed, head_dim=128)

| Test | FP16 | TurboQuant 4-bit | Result |
|------|------|:---:|:---:|
| Math | Correct | Correct | PASS |
| QA | Correct | Correct | PASS |
| Code | Correct | Starts OK, degrades | PARTIAL |
| Reasoning | Correct | Correct | PASS |

## Why MLX

MLX stores KV cache as regular `mx.array` tensors accessible from Python:

```python
cache[0].keys     # real key tensor — we read and write this directly
cache[0].values   # real value tensor
cache[0].offset   # number of cached tokens
```

Ollama/llama.cpp store KV cache in C++ memory — no Python access. That's why TurboQuant targets MLX.

## Configuration

```python
# Auto-detect model architecture (works with any HuggingFace model on MLX)
from turboquant import get_model_config
config = get_model_config(model)
# {'head_dim': 256, 'num_layers': 24, 'num_kv_heads': 2, ...}

# Compress with options
compress_cache(
    cache,
    model=model,          # auto-detect head_dim
    bits=4,               # 2, 3, or 4 bit quantization
    window_size=16,       # keep last N tokens in FP16 (0 = compress all)
    min_context=0,        # skip compression below this context length
)
```

## Known Limitations

- **Code generation**: Degrades on models where ALL layers are compressed (Qwen2.5). Works fine on hybrid models (Qwen3.5) where only full-attention layers are compressed.
- **Apple Silicon only**: Uses MLX. Does not run on CUDA/CPU-only machines.
- **Metal buffer limit**: Contexts above ~24K tokens may hit Metal's 8GB single-allocation limit.
- **Stage 2 (QJL)**: Written (`qjl.py`, `attention.py`, `cache.py`) but not integrated into the generation pipeline yet. Needed for full reliability on all models.

## Architecture

```
src/turboquant/
├── __init__.py       # compress_cache, get_model_config
├── patch.py          # compress_cache() — reads/writes KV cache in-place
├── compressor.py     # PolarQuant: rotation + Lloyd-Max quantization
├── codebook.py       # Lloyd-Max codebook builder (numpy/scipy)
├── qjl.py           # QJL residual correction (written, not integrated)
├── attention.py      # Custom attention with QJL (written, not integrated)
└── cache.py          # TurboQuant cache structure (written, not integrated)
```
