# TurboQuant v0.5.0

KV-cache compression for LLM inference on Apple Silicon (MLX).

Compresses key/value tensors to **uint8 indices + float16 norms** using 4-bit Lloyd-Max quantization with random orthogonal rotation. Achieves **2.0x real memory reduction** with cosine similarity 0.9953.

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

## Install

```bash
pip install mlx mlx-lm scipy
pip install git+https://github.com/thepradip/turboquant-m2.git
```

## Quick Start

```python
import mlx_lm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from turboquant import compress_cache, compact_cache, restore_cache, chunked_prefill

model, tok = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")

text = tok.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False, add_generation_prompt=True,
)
ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)

# For long prompts (>4K tokens), use chunked prefill
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
mx.eval(logits)

# Compress KV cache — one line
result = compress_cache(cache, model=model, bits=4)

# Generate from compressed cache (standard attention, no code changes)
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

### Real Memory Savings

After `compress_cache()`, call `compact_cache()` to free the FP16 tensors and keep only the compressed indices+norms:

```python
result = compress_cache(cache, model=model, bits=4)

# Free FP16, keep uint8 indices + float16 norms (2.0x smaller)
savings = compact_cache(cache)
print(f"Freed {savings['freed_mb']} MB, retained {savings['retained_mb']} MB")

# Before each generation step, reconstruct FP16
restore_cache(cache)
logits = model(y.reshape(1, -1), cache=cache)
```

### Saving Experiment Results

```python
from turboquant import save_experiment, list_experiments, load_experiment

save_experiment(
    model_name="Qwen3.5-4B-MLX-4bit",
    compress_result=result,
    model=model,
    context_tokens=8192,
    gen_tps=26.9,
    passed=True,
)

for e in list_experiments():
    print(e["filename"], e["model_name"], e["cosine"])
```

## Confirmed Results (v0.5.0)

All numbers from real runs on Apple M2 Pro 16GB with Qwen3.5-4B-MLX-4bit.

### Memory and Speed

| Context | FP16 KV | Compressed | Saved | Ratio | Compress | BL tps | TQ tps |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1,024 | 32.0 MB | 16.1 MB | 15.9 MB | **2.0x** | 0.19s | 22.2 | **46.8** |
| 2,048 | 64.0 MB | 32.2 MB | 31.8 MB | **2.0x** | 0.13s | 22.6 | **35.1** |
| 4,096 | 128.0 MB | 64.5 MB | 63.5 MB | **2.0x** | 0.22s | 2.8 | **17.4** |
| 8,192 | 256.0 MB | 129.0 MB | 127.0 MB | **2.0x** | 0.71s | 22.4 | **26.9** |

- **Compressed** = real in-memory size (uint8 indices + float16 norms), measured by `nbytes`
- **Ratio** = FP16 bytes / compressed bytes — real, not theoretical
- **Compress time** scales linearly (per-layer `mx.eval()` prevents graph explosion)
- TQ is **faster than baseline** at all tested context lengths

### Multi-Model Results (v0.4.0 experiment_report.json)

These results are from an earlier experiment script (not in repo). The baseline tps numbers reflect total time including prefill, not pure generation throughput.

| Model | Layers Compressed | Context | KV Saved | Baseline tps | TQ tps |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen3.5-4B-MLX-4bit | 8/32 | 8,912 | 207.8 MB | 3.2 | 10.7 |
| Gemma3-4B-4bit | 34/34 | 4,347 | 430.7 MB | 7.1 | 45.7 |
| Qwen3.5-2B-OptiQ-4bit | 6/24 | 8,914 | 77.9 MB | 8.1 | 32.8 |
| Qwen3.5-2B | 6/24 | 4,270 | 37.3 MB | 17.1 | 35.2 |

## How It Works

### Compression Pipeline

```
Input: KV cache with FP16 keys/values after prefill

For each compressible layer:
  1. Normalize K,V to unit sphere → store float16 norms
  2. Rotate by cached random orthogonal matrix R
  3. Quantize per-coordinate via boundary comparisons (4-bit, 15 midpoints)
  4. Store uint8 indices + float16 norms on cache object
  5. Dequantize and write FP16 back for standard attention
  6. mx.eval() per layer (prevents memory graph accumulation)

Output: Cache has both FP16 (for generation) and compressed (for compact_cache)
```

### Key Algorithms

**PolarQuant (Stage 1)**: Rotate → quantize → dequantize → inverse rotate

```
x̂ = x / ||x||                    # normalize to unit sphere
y = x̂ @ R^T                      # random orthogonal rotation
indices = cumsum(y >= boundaries)  # binary search on sorted centroids
ŷ = centroids[indices]             # lookup
x_hat = (ŷ @ R) * ||x||           # inverse rotate + rescale
```

**Boundary-Search Quantization (v0.5.0)**: Replaces brute-force argmin over all 16 centroids with 15 boundary comparisons. 29x faster, identical output (3/33M off-by-one at exact midpoints).

**QJL (Stage 2, disabled)**: Residual correction via random projection. Written but disabled — adds variance at 4-bit where MSE is already good (cosine 0.9953). May help at 2-bit.

## API Reference

### Core Functions

| Function | Purpose |
|:---|:---|
| `compress_cache(cache, model, bits=4)` | Compress KV cache in-place. Returns metrics dict. |
| `compact_cache(cache)` | Free FP16 tensors, keep indices+norms. Returns savings dict. |
| `restore_cache(cache)` | Reconstruct FP16 from compressed data. |
| `chunked_prefill(model, ids, cache, chunk_size=2048)` | Process long prompts in chunks (bypasses Metal 8GB limit). |

### Model Utilities

| Function | Purpose |
|:---|:---|
| `get_head_dim(model)` | Auto-detect head dimension from any MLX model. |
| `get_num_layers(model)` | Auto-detect number of transformer layers. |
| `get_model_config(model)` | Extract full config (head_dim, layers, kv_heads, etc.). |

### Experiment Tracking

| Function | Purpose |
|:---|:---|
| `save_experiment(model_name, compress_result, ...)` | Save results to timestamped JSON. |
| `list_experiments(model_filter=None)` | List all saved results. |
| `load_experiment(filename)` | Load specific result file. |

### Advanced (QJL Integration)

| Function | Purpose |
|:---|:---|
| `patch_model(model, bits=4)` | Monkey-patch attention for QJL-corrected scores. |
| `make_turboquant_cache(model, bits=4)` | Create TurboQuantCache instances for all layers. |
| `TurboQuantCache(head_dim, key_bits, value_bits)` | Drop-in KVCache with on-insert compression. |
| `turboquant_sdpa(queries, keys, values, cache, scale)` | Scaled dot-product attention with optional QJL. |
| `PolarQuantMLX(head_dim, bits, seed)` | Low-level: quantize/dequantize unit-sphere vectors. |
| `QJLMLX(head_dim, m, seed)` | Low-level: QJL residual correction (disabled). |

## Architecture

```
src/turboquant/
├── __init__.py       # Public API: compress_cache, compact_cache, restore_cache, ...
├── patch.py          # Main API: compress, compact, restore, chunked_prefill
├── compressor.py     # PolarQuant: boundary-search quantization (v0.5.0)
├── codebook.py       # Lloyd-Max codebook + rotation matrix builders
├── cache.py          # TurboQuantCache: stores uint8 indices + float16 norms
├── attention.py      # Custom SDPA with optional QJL correction
├── results.py        # Experiment save/list/load
├── qjl.py            # QJL residual correction (written, disabled)
├── metal_kernel.py   # Fused Metal kernel (validated, not in production path)
├── torch_backend.py  # PyTorch backend (CPU/CUDA/MPS)
└── mlx_native.py     # Pure MLX implementation (alternative)
```

## What's New in v0.5.0

- **Real memory savings**: Stores uint8 indices + float16 norms (2.0x compression, measured by `nbytes`)
- **29x faster quantization**: Boundary comparisons replace brute-force argmin, eliminates 536 MB/layer intermediate tensor
- **Per-layer eval**: Fixes 143s→0.7s compression at 8K by preventing memory graph accumulation
- **compact_cache/restore_cache**: New API for on-demand FP16 reconstruction
- **QJL off by default**: Saves memory (QJL signs were larger than the compression savings)
- **Known cosine lookup**: Skips redundant 67 MB/layer computation for deterministic value

## Known Issues

- **Hybrid attention models**: Qwen3.5 models only compress 8/32 layers (hybrid attention), limiting benefit vs fully-compressible models like Gemma3 (34/34).
- **QJL disabled**: Stage 2 adds variance at 4-bit. May work at 2-bit where residual is larger.
- **Apple Silicon only**: Uses MLX. PyTorch backend exists but is untested in production.
- **compact/restore overhead**: Each generation step requires `restore_cache()` to reconstruct FP16. For continuous generation, use `compress_cache()` alone (keeps FP16 in cache).
