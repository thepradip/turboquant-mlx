# TurboQuant v0.5.0

KV-cache compression for LLM inference on Apple Silicon (MLX).

Compresses key/value tensors to **uint8 indices + float16 norms** using 4-bit Lloyd-Max quantization with random orthogonal rotation. Achieves **2.0x real memory reduction** with cosine similarity 0.9953.

Based on: [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)

---

## Table of Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Confirmed Results](#confirmed-results-v050)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Configuration Options](#configuration-options)
- [Architecture](#architecture)
- [Tested Models](#tested-models)
- [What's New in v0.5.0](#whats-new-in-v050)
- [Known Issues & Limitations](#known-issues--limitations)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Hardware**: Apple Silicon Mac (M1, M2, M3, M4 — any variant)
- **RAM**: 8 GB minimum (16 GB recommended for 4B models)
- **Python**: >= 3.9
- **OS**: macOS Ventura or later

### Dependencies

| Package | Version | Purpose |
|:---|:---|:---|
| `mlx` | >= 0.20.0 | Apple Silicon ML framework |
| `mlx-lm` | >= 0.15.0 | Model loading, tokenization, KVCache |
| `numpy` | >= 1.24.0 | Codebook + rotation matrix construction |
| `scipy` | >= 1.10.0 | Beta distribution for Lloyd-Max optimization |

## Install

### From GitHub (recommended)

```bash
pip install mlx mlx-lm scipy
pip install git+https://github.com/thepradip/turboquant-mlx.git
```

### From Source (for development)

```bash
git clone https://github.com/thepradip/turboquant-mlx.git
cd turboquant-mlx
pip install -e ".[dev]"
```

### Verify Installation

```python
import turboquant
print(turboquant.__version__)  # 0.5.0
```

---

## Quick Start

```python
import mlx_lm
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from turboquant import compress_cache, chunked_prefill

# 1. Load any MLX model
model, tok = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")

# 2. Prepare prompt
text = tok.apply_chat_template(
    [{"role": "user", "content": "Explain attention in transformers."}],
    tokenize=False, add_generation_prompt=True,
)
ids = mx.array(tok.encode(text))

# 3. Prefill + compress
cache = make_prompt_cache(model)
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
result = compress_cache(cache, model=model, bits=4, compact=False)

# 4. Generate (standard loop — compact=False keeps FP16 for compatibility)
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

**Memory-saving mode** (uses `generate_step` for real memory savings):

```python
from turboquant import compress_cache, generate_step, chunked_prefill

cache = make_prompt_cache(model)
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
compress_cache(cache, model=model, bits=4)  # compact=True (default), frees FP16

y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(200):
    logits = generate_step(model, y[:, None], cache)  # decompress → generate → recompress
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tok.eos_token_id:
        break
    tokens.append(y.item())
print(tok.decode(tokens))
```

---

## Step-by-Step Guide

### Step 1: Load a Model

TurboQuant works with any MLX model loaded via `mlx_lm.load()`. Config is auto-detected.

```python
import mlx_lm

model, tok = mlx_lm.load("mlx-community/Qwen3.5-4B-MLX-4bit")

# Verify what TurboQuant sees
from turboquant import get_model_config
config = get_model_config(model)
print(config)
# {'head_dim': 256, 'num_layers': 32, 'num_kv_heads': 4, ...}
```

### Step 2: Prefill the KV Cache

For short prompts (<4K tokens), use standard prefill:

```python
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

ids = mx.array(tok.encode(text))
cache = make_prompt_cache(model)
logits = model(ids[None], cache=cache)
mx.eval(logits)
```

For long prompts (>4K tokens), use `chunked_prefill` to avoid Metal's 8 GB buffer limit:

```python
from turboquant import chunked_prefill

cache = make_prompt_cache(model)
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
mx.eval(logits)
```

`chunked_prefill` processes the prompt in 2048-token chunks. Each chunk's attention matrix is only `(n_heads x 2048 x 2048 x 4 bytes)` = ~128 MB instead of the full `(n_heads x seq x seq x 4)` which crashes at ~15K tokens.

Tested up to 86K tokens on M2 Pro 16 GB.

### Step 3: Compress the KV Cache

```python
from turboquant import compress_cache

result = compress_cache(cache, model=model, bits=4)
print(result)
# {
#   'cosine': 0.9953,
#   'compress_ms': 710,
#   'layers_compressed': 8,
#   'original_mb': 256.0,
#   'compressed_mb': 65.0,
#   'saved_mb': 191.0,
#   'ratio': 3.9
# }
```

What happens inside:
1. For each compressible layer, normalizes K/V to unit sphere and stores float16 norms
2. Rotates by a cached random orthogonal matrix
3. Quantizes each coordinate to the nearest Lloyd-Max centroid (4-bit = 16 levels)
4. Stores uint8 indices + float16 norms on the cache object
5. Writes dequantized FP16 back for standard attention compatibility
6. Runs `mx.eval()` per layer to prevent memory graph accumulation

After this call, the cache works exactly like before — no downstream code changes needed.

### Step 4: Generate Tokens

Standard generation loop. TurboQuant does not change the generation API:

```python
y = mx.argmax(logits[:, -1, :], axis=-1)
tokens = []
for _ in range(500):
    logits = model(y.reshape(1, -1), cache=cache)
    mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    if y.item() == tok.eos_token_id:
        break
    tokens.append(y.item())
print(tok.decode(tokens))
```

### Step 5 (Optional): Compact for Real Memory Savings

After `compress_cache()`, the cache holds both FP16 tensors (for generation) and compressed indices+norms. To actually free the FP16 memory:

```python
from turboquant import compact_cache, restore_cache

# Free FP16 tensors, keep only indices+norms
savings = compact_cache(cache)
print(savings)
# {'layers_compacted': 8, 'freed_mb': 256.0, 'retained_mb': 129.0, 'actual_saved_mb': 127.0}

# Before next generation step, reconstruct FP16
restore_cache(cache)
logits = model(y.reshape(1, -1), cache=cache)
```

Use this when memory is tight (e.g., running multiple conversations, 32K+ context).
For simple single-conversation generation, `compress_cache()` alone is sufficient.

### Step 6 (Optional): Save Results

```python
from turboquant import save_experiment

save_experiment(
    model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
    compress_result=result,
    model=model,
    context_tokens=len(ids),
    gen_tokens=len(tokens),
    gen_tps=35.0,
    passed=True,
    notes="8K context benchmark",
)
```

Results are saved to `results/` as timestamped JSON with auto-detected hardware info:

```
results/qwen3.5-4b-mlx-4bit_8192tok_20260329_120000.json
```

Review past experiments:

```python
from turboquant import list_experiments, load_experiment

for e in list_experiments():
    print(f"{e['filename']:50s} ctx={e['context_tokens']} cos={e['cosine']} tps={e['gen_tps']}")

data = load_experiment("qwen3.5-4b-mlx-4bit_8192tok_20260329_120000.json")
```

### Step 7 (Optional): Run Benchmark

The included benchmark script tests baseline vs TurboQuant across multiple context lengths:

```bash
cd turboquant-mlx
python examples/benchmark.py --model mlx-community/Qwen3.5-4B-MLX-4bit
```

Tests 1K, 2K, 4K, 8K, 16K, 32K, 64K contexts in three modes: baseline (FP16), mlx_quantized (built-in 4-bit group-wise), and turboquant (Lloyd-Max 4-bit). Results saved to `results/`.

---

## Confirmed Results (v0.5.0)

All numbers from real runs on Apple M2 Pro 16 GB with Qwen3.5-4B-MLX-4bit.

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

### Multi-Model Results (v0.4.0)

From `experiment_report.json` — 4 models, 4 context lengths, 16 runs total on M2 Pro 16 GB.

| Model | Layers Compressed | Context | KV Saved | Cosine | TQ tps |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen3.5-4B-MLX-4bit | 8/32 | 8,912 | 207.8 MB | 0.9953 | 10.7 |
| Gemma3-4B-4bit | 34/34 | 4,347 | 430.7 MB | 0.9953 | 45.7 |
| Qwen3.5-2B-OptiQ-4bit | 6/24 | 8,914 | 77.9 MB | 0.9953 | 32.8 |
| Qwen3.5-2B | 6/24 | 4,270 | 37.3 MB | 0.9953 | 35.2 |

### Memory Budget (16 GB Mac)

| Context | Model Weight | FP16 KV | After Compact | Total | Fits? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1K | 2,400 MB | 32 MB | 16 MB | 2.4 GB | Yes |
| 8K | 2,400 MB | 256 MB | 129 MB | 2.5 GB | Yes |
| 16K | 2,400 MB | 512 MB | ~258 MB | 2.7 GB | Yes |
| 32K | 2,400 MB | 1,024 MB | ~516 MB | 2.9 GB | Yes |
| 65K | 2,400 MB | 2,048 MB | ~1,030 MB | 3.4 GB | Yes |

---

## How It Works

### The Problem

LLMs store key/value tensors for every previous token during generation. At long context lengths, this KV cache dominates memory:

```
KV memory = 2 × num_kv_heads × head_dim × seq_len × num_layers × 2 bytes (FP16)

Qwen3.5-4B at 32K tokens:
  2 × 4 × 256 × 32768 × 32 × 2 = 1,024 MB
```

### The Solution

TurboQuant compresses the KV cache from FP16 (2 bytes/element) to uint8 indices (1 byte/element) + small float16 norms:

```
Compressed = seq_len × head_dim × 1 byte (indices)
           + seq_len × 2 bytes (norms per token)
           ≈ 50% of FP16
```

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

Output: Cache has FP16 (for generation) + compressed (for compact_cache)
```

### Key Algorithms

**PolarQuant (Stage 1)** — the core compression:

```
x̂ = x / ||x||                    # 1. normalize to unit sphere
y = x̂ @ R^T                      # 2. random orthogonal rotation
indices = cumsum(y >= boundaries)  # 3. quantize (boundary search)
ŷ = centroids[indices]             # 4. lookup centroid values
x_hat = (ŷ @ R) * ||x||           # 5. inverse rotate + rescale
```

Why rotation? After rotating a unit-sphere vector by a random orthogonal matrix, each coordinate becomes approximately i.i.d. with a known Beta distribution. This makes scalar quantization near-optimal — the Lloyd-Max codebook minimizes MSE for this distribution.

**Boundary-Search Quantization (v0.5.0)** — replaces brute-force argmin:

```python
# Old (v0.4.0): builds (..., head_dim, 16) tensor — 536 MB/layer at 32K
dists = mx.abs(mx.expand_dims(y, axis=-1) - centroids)
indices = mx.argmin(dists, axis=-1)

# New (v0.5.0): 15 scalar comparisons — O(input_size) memory
boundaries = (centroids[:-1] + centroids[1:]) / 2  # precomputed midpoints
indices = zeros(y.shape)
for b in boundaries:
    indices += (y >= b)
```

29x faster, identical output (3 out of 33.5M values differ by 1 at exact midpoints, cosine identical to 6 decimal places).

**QJL (Stage 2, disabled)** — residual correction via random projection:

```
r = x - x_mse                          # quantization residual
signs = sign(r @ S^T)                   # random Gaussian projection
correction = ||r|| * sqrt(π/2)/m * <S@q, signs>  # unbiased estimator
```

Written and validated but disabled — adds variance at 4-bit where MSE is already good (cosine 0.9953). May help at 2-bit where residual is larger.

### Why Only Some Layers Are Compressed

Qwen3.5 models use **hybrid attention** — some layers have standard KV cache, others use sliding window or no cache. TurboQuant detects and compresses only the layers with standard KVCache:

| Model | Total Layers | Compressible | Reason |
|:---|:---:|:---:|:---|
| Qwen3.5-4B | 32 | 8 | Hybrid attention (24 sliding window) |
| Qwen3.5-2B | 24 | 6 | Hybrid attention (18 sliding window) |
| Gemma3-4B | 34 | 34 | All standard attention |

---

## API Reference

### Core Functions

#### `compress_cache(cache, model=None, head_dim=None, bits=4, window_size=0, min_context=0)`

Compress KV cache in-place using 4-bit Lloyd-Max quantization.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `cache` | `List[KVCache]` | required | KV cache from `make_prompt_cache(model)` |
| `model` | MLX model | `None` | For auto-detecting head_dim |
| `head_dim` | `int` | `None` | Override head_dim (if model not provided) |
| `bits` | `int` | `4` | Quantization bits: 2, 3, or 4 |
| `window_size` | `int` | `0` | Keep this many recent tokens uncompressed |
| `min_context` | `int` | `0` | Skip compression if context shorter than this |

**Returns**: `Dict` with keys: `cosine`, `compress_ms`, `layers_compressed`, `original_mb`, `compressed_mb`, `saved_mb`, `ratio`

#### `compact_cache(cache)`

Free FP16 keys/values, keep only uint8 indices + float16 norms. Call after `compress_cache()`.

**Returns**: `Dict` with keys: `layers_compacted`, `freed_mb`, `retained_mb`, `actual_saved_mb`

#### `restore_cache(cache)`

Reconstruct FP16 keys/values from compressed indices+norms. Call before model forward pass if `compact_cache()` was used.

#### `chunked_prefill(model, ids, cache, chunk_size=2048)`

Process long prompts in chunks to bypass Metal's 8 GB buffer limit.

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `model` | MLX model | required | The loaded model |
| `ids` | `mx.array` | required | Token IDs as 1D array (not batched) |
| `cache` | `List[KVCache]` | required | KV cache from `make_prompt_cache(model)` |
| `chunk_size` | `int` | `2048` | Tokens per chunk. 2048 is safe for all models |

**Returns**: `mx.array` — logits from the last chunk

### Model Utilities

| Function | Signature | Description |
|:---|:---|:---|
| `get_head_dim(model)` | `(model) -> int` | Auto-detect head dimension |
| `get_num_layers(model)` | `(model) -> int` | Auto-detect number of layers |
| `get_model_config(model)` | `(model) -> Dict` | Full config: head_dim, num_layers, num_kv_heads, etc. |

### Experiment Tracking

| Function | Description |
|:---|:---|
| `save_experiment(model_name, compress_result, model, context_tokens, gen_tps, ...)` | Save results to `results/` as timestamped JSON with hardware auto-detection |
| `list_experiments(model_filter=None)` | List all saved results, sorted newest first |
| `load_experiment(filename)` | Load specific result file by filename |

### Advanced / Experimental

| Function | Description |
|:---|:---|
| `patch_model(model, bits=4)` | Monkey-patch attention (experimental, not production-tested) |
| `make_turboquant_cache(model, bits=4)` | Create TurboQuantCache instances for all layers |
| `TurboQuantCache(head_dim, key_bits, value_bits)` | Drop-in KVCache with on-insert compression |
| `turboquant_sdpa(queries, keys, values, cache, scale)` | Standard SDPA with dequantized KV cache |
| `PolarQuantMLX(head_dim, bits, seed)` | Low-level: quantize/dequantize unit-sphere vectors |

> **Note**: QJL residual correction (`qjl.py`) is implemented but disabled — 4-bit MSE provides sufficient quality (0.991+ cosine) without the added variance. The code is kept for future research at lower bit widths.

---

## Configuration Options

### Bit Width

| Bits | Centroids | Cosine (256-dim) | Ratio | Use Case |
|:---:|:---:|:---:|:---:|:---|
| 4 | 16 | 0.9953 | 2.0x | Default. Best quality-compression tradeoff |
| 3 | 8 | 0.9870 | ~2.6x | More aggressive. Test quality before deploying |
| 2 | 4 | 0.9650 | ~4.0x | Experimental. Noticeable quality loss |

### Window Size

Keep recent tokens in full FP16 for better quality on local attention:

```python
compress_cache(cache, model=model, bits=4, window_size=512, compact=False)
# Compresses all tokens EXCEPT the most recent 512
# NOTE: window_size > 0 requires compact=False (standard generation loop)
# generate_step() only works with window_size=0
```

### Minimum Context

Skip compression for short contexts where overhead isn't worth it:

```python
compress_cache(cache, model=model, bits=4, min_context=1024)
# Only compresses if context > 1024 tokens
```

---

## Architecture

```
src/turboquant/
├── __init__.py       # Public API exports
├── patch.py          # Core: compress_cache, generate_step, chunked_prefill
├── compressor.py     # PolarQuant: polar quantization (2/3/4-bit)
├── codebook.py       # Lloyd-Max codebook + rotation matrix builders
├── cache.py          # TurboQuantCache: uint8 indices + float16 norms
├── attention.py      # Standard SDPA with dequantized KV cache
├── results.py        # Experiment save/list/load
├── qjl.py            # QJL residual correction (disabled, kept for research)
├── metal_kernel.py   # Fused Metal kernel (used by bonsai_loader)
└── bonsai_loader.py  # 1-bit Bonsai model loader

benchmarks/
├── tq_eval.py                # Unified eval suite (65 questions, LLM-as-judge)
├── tq_eval_report.py         # HTML report generator
└── tq_eval_65_questions.json # Test dataset with reference answers

tests/
└── test_core.py      # 55 tests: round-trip, pack/unpack, compress, generate_step
```

---

## Tested Models

| Model | Size | Compressible Layers | Tested Context | Status |
|:---|:---|:---:|:---:|:---:|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 2.4 GB | 8/32 | 1K–65K | Confirmed v0.5.0 |
| `mlx-community/gemma-3-4b-it-4bit` | 2.5 GB | 34/34 | 2K–13K | Confirmed v0.4.0 |
| `mlx-community/Qwen3.5-2B-OptiQ-4bit` | 1.0 GB | 6/24 | 2K–86K | Confirmed v0.4.0 |
| `Qwen/Qwen3.5-2B` | 5.0 GB | 6/24 | 2K–13K | Confirmed v0.4.0 |

Any model loaded via `mlx_lm.load()` should work. Config is auto-detected. Models with hybrid attention (Qwen3.5) will have fewer compressible layers.

---

## What's New in v0.5.0

| Feature | v0.4.0 | v0.5.0 |
|:---|:---|:---|
| Memory savings | None (wrote FP16 back) | **Real savings** via `generate_step()` |
| Quantization speed | Brute-force argmin | **29x faster** (boundary comparisons) |
| 8K compression | ~5.3s | **0.71s** (per-layer eval) |
| 32K compression | 143s (graph explosion) | Linear scaling restored |
| Eval suite | None | **65 questions, LLM-as-judge, 3-model comparison** |
| New API | — | `generate_step()`, `compact_cache()`, `restore_cache()` |

---

## Known Issues & Limitations

- **Hybrid attention**: Qwen3.5 models only compress 8/32 layers. Gemma compresses all layers and benefits more.
- **Apple Silicon only**: Uses MLX. No PyTorch backend.
- **generate_step overhead**: `restore_cache()` reconstructs full FP16 before each forward pass. Peak memory during a step = full FP16. Savings are between steps, not during.
- **window_size > 0**: Not compatible with `generate_step()`. Use `compact=False` with standard generation loop instead.
- **QJL disabled**: Residual correction implemented (`qjl.py`) but disabled — 4-bit MSE provides 0.991+ cosine without it. Kept for future research at lower bit widths.

---

## Troubleshooting

### "Metal's 8 GB buffer limit" crash at >15K tokens

Use `chunked_prefill()` instead of direct `model(ids, cache=cache)`:

```python
logits = chunked_prefill(model, ids, cache, chunk_size=2048)
```

### Slow compression at 32K+ context

Make sure you're on v0.5.0. v0.4.0 had a batch-eval bug that caused 143s at 32K. v0.5.0 uses per-layer eval.

```python
import turboquant
print(turboquant.__version__)  # Should be 0.5.0
```

### "Provide head_dim or model" error

Pass the model object to `compress_cache()`:

```python
result = compress_cache(cache, model=model, bits=4)  # not just compress_cache(cache)
```

### No memory savings visible

`compress_cache()` keeps FP16 in the cache for generation compatibility. To see real savings:

```python
result = compress_cache(cache, model=model, bits=4)
savings = compact_cache(cache)  # This actually frees the FP16 memory
print(savings['actual_saved_mb'])
```

### Model generates garbage after compression

This should not happen — compression preserves cosine 0.991+. If it does:
1. Check `result['layers_compressed']` — should be >0
2. Ensure you use `compact=False` with standard generation, or `generate_step()` with `compact=True`
3. Try `bits=4` (not 2 or 3)

### Import error: "No module named turboquant"

Install from source:

```bash
pip install git+https://github.com/thepradip/turboquant-mlx.git
```

Or if developing locally:

```bash
cd turboquant-mlx
pip install -e .
```

---

## License

Apache-2.0

## Author

Pradip Tivhale — [github.com/thepradip](https://github.com/thepradip)
