"""
TurboQuant for MLX — compress KV cache, extend context length.

Main API:
  1. compress_cache() — compress KV after prefill
  2. chunked_prefill() — process long prompts in chunks (bypasses Metal 8GB limit)

Usage:
    from turboquant import compress_cache, chunked_prefill

    cache = make_prompt_cache(model)
    logits = chunked_prefill(model, ids, cache)  # handles any context length
    compress_cache(cache, model=model, bits=4)    # compress KV
    # generate from compressed cache...
"""

import time
import mlx.core as mx
from typing import Any, Dict, List, Optional, Tuple

from .compressor import PolarQuantMLX


def get_head_dim(model) -> int:
    """Auto-detect head_dim from any MLX model."""
    args = model.args
    if hasattr(args, 'text_config'):
        tc = args.text_config
        return tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])
    return getattr(args, "head_dim", args.hidden_size // args.num_attention_heads)


def get_num_layers(model) -> int:
    """Auto-detect number of layers."""
    args = model.args
    if hasattr(args, 'text_config'):
        return args.text_config["num_hidden_layers"]
    return getattr(args, "num_hidden_layers", len(model.layers))


def get_model_config(model) -> Dict:
    """Auto-detect full model config."""
    args = model.args
    if hasattr(args, 'text_config'):
        tc = args.text_config
        return {
            "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            "num_layers": tc["num_hidden_layers"],
            "num_kv_heads": tc["num_key_value_heads"],
            "num_attention_heads": tc["num_attention_heads"],
            "hidden_size": tc["hidden_size"],
        }
    hidden = getattr(args, "hidden_size", 0)
    n_heads = getattr(args, "num_attention_heads", 1)
    return {
        "head_dim": getattr(args, "head_dim", hidden // n_heads if n_heads else 128),
        "num_layers": getattr(args, "num_hidden_layers", len(model.layers)),
        "num_kv_heads": getattr(args, "num_key_value_heads", n_heads),
        "num_attention_heads": n_heads,
        "hidden_size": hidden,
    }


def chunked_prefill(
    model: Any,
    ids: "mx.array",
    cache: List[Any],
    chunk_size: int = 2048,
) -> "mx.array":
    """
    Process long prompts in chunks to bypass Metal's 8GB buffer limit.

    Standard prefill crashes at ~15K tokens because the attention matrix
    (n_heads × seq × seq × 4 bytes) exceeds Metal's single allocation limit.
    Chunked prefill processes 2048 tokens at a time, so the max attention
    matrix is only (n_heads × 2048 × 2048 × 4) = ~128MB.

    Tested: 86K tokens on M2 Pro 16GB with Qwen3.5-2B-OptiQ-4bit.

    Args:
        model: MLX model.
        ids: Token IDs as 1D mx.array (not batched).
        cache: KV cache from make_prompt_cache(model).
        chunk_size: Tokens per chunk. 2048 is safe for all models.

    Returns:
        logits from the last chunk (use for first token generation).

    Example::

        cache = make_prompt_cache(model)
        logits = chunked_prefill(model, ids, cache)
        compress_cache(cache, model=model, bits=4)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        # continue generation...
    """
    total = len(ids)
    logits = None

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = ids[start:end]
        logits = model(chunk[None], cache=cache)
        mx.eval(logits)

    return logits


def compress_cache(
    cache: List[Any],
    model: Any = None,
    head_dim: int = None,
    bits: int = 4,
    window_size: int = 0,
    min_context: int = 0,
) -> Dict:
    """
    Compress KV cache in-place using TurboQuant.

    Args:
        cache: List of MLX KVCache objects from make_prompt_cache().
        model: MLX model (for auto-detecting head_dim).
        head_dim: Override head_dim if model not provided.
        bits: Quantization bits (2, 3, or 4). Default: 4.
        window_size: Keep this many recent tokens uncompressed. Default: 0 (compress all).
        min_context: Skip compression if context shorter than this. Default: 0.

    Returns:
        Dict with cosine, compress_ms, layers_compressed, memory stats.
    """
    if head_dim is None and model is not None:
        head_dim = get_head_dim(model)
    if head_dim is None:
        raise ValueError("Provide head_dim or model")

    t0 = time.time()
    cos_scores = []
    total_orig = 0
    total_comp = 0

    for li in range(len(cache)):
        c = cache[li]
        if not hasattr(c, 'keys') or c.keys is None:
            continue
        seq = c.offset
        if seq <= min_context:
            continue

        # Determine range to compress
        compress_end = seq - window_size if window_size > 0 else seq
        if compress_end <= 0:
            continue

        k = c.keys[:, :, :compress_end, :]
        v = c.values[:, :, :compress_end, :]
        mx.eval(k, v)

        kd = k.shape[-1]
        if kd < 2:
            continue

        # Compress keys
        k_mse = PolarQuantMLX(kd, bits=bits, seed=42 + li)
        k_f = k.astype(mx.float32)
        k_norms = mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True))
        k_norms = mx.maximum(k_norms, 1e-8)
        k_unit = k_f / k_norms
        k_hat_unit = k_mse.dequantize(k_mse.quantize(k_unit))
        k_hat = (k_hat_unit * k_norms).astype(k.dtype)

        # Compress values
        v_mse = PolarQuantMLX(kd, bits=bits, seed=1000 + li)
        v_f = v.astype(mx.float32)
        v_norms = mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True))
        v_norms = mx.maximum(v_norms, 1e-8)
        v_unit = v_f / v_norms
        v_hat_unit = v_mse.dequantize(v_mse.quantize(v_unit))
        v_hat = (v_hat_unit * v_norms).astype(v.dtype)

        # Cosine similarity
        flat_k = mx.reshape(k_f, (-1, kd))
        flat_r = mx.reshape(k_hat.astype(mx.float32), (-1, kd))
        dot = mx.sum(flat_k * flat_r, axis=-1)
        nk = mx.sqrt(mx.sum(flat_k * flat_k, axis=-1))
        nr = mx.sqrt(mx.sum(flat_r * flat_r, axis=-1))
        cos = mx.mean(dot / (nk * nr + 1e-8))
        mx.eval(cos)
        cos_scores.append(cos.item())

        # Memory tracking
        n_elements = k.size
        total_orig += n_elements * 2 * 2  # K + V, 2 bytes each (FP16)
        total_comp += (n_elements * bits / 8 + k.shape[2] * k.shape[1] * 2) * 2  # indices + norms, K+V

        # Write back in-place
        c.keys[:, :, :compress_end, :] = k_hat
        c.values[:, :, :compress_end, :] = v_hat
        mx.eval(c.keys, c.values)

    elapsed_ms = (time.time() - t0) * 1000
    avg_cos = sum(cos_scores) / len(cos_scores) if cos_scores else 0
    ratio = total_orig / total_comp if total_comp > 0 else 0

    return {
        "cosine": round(avg_cos, 4),
        "compress_ms": round(elapsed_ms, 0),
        "layers_compressed": len(cos_scores),
        "original_mb": round(total_orig / 1024 / 1024, 1),
        "compressed_mb": round(total_comp / 1024 / 1024, 1),
        "saved_mb": round((total_orig - total_comp) / 1024 / 1024, 1),
        "ratio": round(ratio, 1),
    }


# ═══════════════════════════════════════════════════════
#  Full QJL Integration — patch model attention
# ═══════════════════════════════════════════════════════

def make_turboquant_cache(model, bits: int = 4, value_bits: int = None) -> List:
    """Create TurboQuant caches for all KVCache layers, default for others."""
    from mlx_lm.models.cache import make_prompt_cache, KVCache
    from .cache import TurboQuantCache

    head_dim = get_head_dim(model)
    vb = value_bits or bits
    default = make_prompt_cache(model)

    caches = []
    for i in range(len(default)):
        if isinstance(default[i], KVCache):
            caches.append(TurboQuantCache(head_dim, key_bits=bits, value_bits=vb, layer_idx=i))
        else:
            caches.append(default[i])
    return caches


def patch_model(model, bits: int = 4):
    """
    Monkey-patch model attention to use QJL-corrected scores with TurboQuantCache.

    After patching, use make_turboquant_cache() to create the cache, then
    model(ids, cache=cache) automatically compresses KV and applies QJL correction.
    """
    import types
    from .cache import TurboQuantCache
    from .attention import turboquant_sdpa

    head_dim = get_head_dim(model)
    patched = 0

    for i, layer in enumerate(model.layers):
        attn = None
        for name in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, name):
                attn = getattr(layer, name)
                break
        if attn is None or not hasattr(attn, 'q_proj'):
            continue

        def make_patched(original_attn):
            def patched_call(self, x, mask=None, cache=None):
                B, L, D = x.shape
                queries = self.q_proj(x)
                keys = self.k_proj(x)
                values = self.v_proj(x)

                queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = self.rope(queries, offset=cache.offset)
                    keys = self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)

                    if isinstance(cache, TurboQuantCache):
                        output = turboquant_sdpa(
                            queries, keys, values,
                            cache=cache, scale=self.scale, mask=mask,
                        )
                    else:
                        from mlx_lm.models.base import scaled_dot_product_attention
                        output = scaled_dot_product_attention(
                            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
                        )
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)
                    from mlx_lm.models.base import scaled_dot_product_attention
                    output = scaled_dot_product_attention(
                        queries, keys, values, scale=self.scale, mask=mask,
                    )

                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)
            return patched_call

        attn.__call__ = types.MethodType(make_patched(attn), attn)
        patched += 1

    return patched
