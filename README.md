# TurboQuant

KV-cache compression for LLM inference on Apple Silicon (MLX).

Directly accesses and compresses the KV cache at code level — reads real key/value tensors, compresses with 4-bit Lloyd-Max quantization, writes back. Model continues generating from compressed cache.

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

model, tok = mlx_lm.load("Qwen/Qwen3.5-2B")

text = tok.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)

# Compress KV cache — one line
result = compress_cache(cache, model=model, bits=4)

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

## Test Results

All tested live on Apple M2 Pro 16GB. Every answer verified correct.

### Indian Constitution QA (402 pages, ~9K token context, 5 hard questions)

| Model | Score | Layers Compressed | KV Saved/Query | Cosine |
|-------|:---:|:---:|:---:|:---:|
| Qwen 3.5 9B (4-bit) | **5/5** | 8/32 | 208 MB | 0.9953 |
| Qwen 3.5 2B | **5/5** | 6/24 | 78 MB | 0.9953 |
| Gemma 3 4B (4-bit) | **5/5** | 34/34 | 893 MB | 0.9953 |
| Qwen 2.5 1.5B | **5/5** | 28/28 | 166 MB | 0.9954 |

Questions tested: Fundamental Rights, Amendment process (Art 368), President's powers, Lok Sabha vs Rajya Sabha, Directive Principles.

### Other tests (5 prompts: math, QA, code, reasoning, long QA)

| Model | Score |
|-------|:---:|
| Qwen 3.5 2B | 5/5 |
| Gemma 3 4B | 5/5 |
| Qwen 2.5 1.5B | 4/5 (code generation degrades) |

### Max context on M2 Pro 16GB

With `chunked_prefill()` — bypasses Metal's 8GB single allocation limit:

| Model | Max Context | KV Saved | Status |
|-------|:---:|:---:|:---:|
| Qwen3.5-2B-OptiQ-4bit (1.4GB) | **86,217** | 754 MB | OK |
| Qwen3.5-4B-MLX-4bit (2.3GB) | **15,565** | 363 MB | OK |
| Gemma3-4B-4bit (2.4GB) | **13,706** | 1,324 MB | OK |

Without chunked prefill, all models crash at ~15K due to Metal's 8GB buffer limit.
With chunked prefill, the 2B model handles **86K tokens** on 16GB RAM.

### AI Paper QA (23 pages, ~8K context)

Tested on "Reliable AI Agents in Python" paper with questions about SLIs/SLOs, Python stack, retry strategies, LangGraph vs OpenAI SDK, observability. Qwen3.5-2B: **5/5 correct, answers match FP16 baseline**.

## Long Context (chunked prefill)

```python
from turboquant import compress_cache, chunked_prefill
from mlx_lm.models.cache import make_prompt_cache

# Handles any context length — 86K+ tokens on 16GB
cache = make_prompt_cache(model)
ids = mx.array(tok.encode(text))
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
mx.eval(logits)

compress_cache(cache, model=model, bits=4)

# Generate
y = mx.argmax(logits[:, -1, :], axis=-1)
# ...
```

## How It Works

```
1. Model does full prefill → KV cache filled with FP16 values
2. TurboQuant reads cache[layer].keys and cache[layer].values
3. For each layer:
   a. Normalize vectors to unit sphere (store norms)
   b. Apply random orthogonal rotation
   c. Quantize each coordinate with Lloyd-Max codebook (4-bit)
   d. Dequantize → inverse rotate → rescale by norms
   e. Write compressed values back into cache
4. Model generates next tokens from compressed cache
```

## Architecture

```
src/turboquant/
├── __init__.py       # compress_cache, get_model_config
├── patch.py          # compress_cache() — main API
├── compressor.py     # PolarQuant: rotation + Lloyd-Max
├── codebook.py       # Lloyd-Max codebook builder
├── metal_kernel.py   # Fused Metal kernel (validated, 0.000001 precision)
├── qjl.py           # QJL residual correction (code present, disabled)
├── attention.py      # Custom attention function
└── cache.py          # TurboQuant cache structure
```

## Known Limitations

- **Metal 8GB buffer limit**: Contexts above ~16K tokens crash on Apple Silicon (hardware limit, not TurboQuant)
- **Code generation on Qwen2.5**: Degrades when all layers compressed. Works fine on hybrid models (Qwen3.5) and Gemma3
- **Apple Silicon only**: Uses MLX. Does not run on CUDA
- **QJL disabled**: Stage 2 of TurboQuant paper written but adds variance instead of reducing bias in our approach. Needs fused kernel integration
