"""
TurboQuant for MLX — compress KV cache, extend context length.

Main API:
  1. compress_cache() — compress KV after prefill (stores indices + norms)
  2. chunked_prefill() — process long prompts in chunks (bypasses Metal 8GB limit)
  3. decompress_cache() — restore FP16 from compressed representation

v0.5.0 — Real memory savings: stores uint8 indices + float16 norms instead
of writing dequantized FP16 back. Per-layer eval to fix 32K speed.

Usage:
    from turboquant import compress_cache, chunked_prefill

    cache = make_prompt_cache(model)
    logits = chunked_prefill(model, ids, cache)  # handles any context length
    result = compress_cache(cache, model=model, bits=4)  # compress KV
    # generate from compressed cache...
"""

import time
import mlx.core as mx
from typing import Any, Dict, List, Optional, Tuple

from .compressor import PolarQuantMLX, pack_indices, unpack_indices


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
    compact: bool = True,
) -> Dict:
    """
    Compress KV cache in-place using TurboQuant polar quantization.

    Quantizes FP16 keys/values to uint8 indices + float16 norms.
    When compact=True (default), the FP16 tensors are freed and only
    the compressed representation is kept — real memory savings.

    After compression, use generate_step() for token generation:
        result = compress_cache(cache, model=model, bits=4)
        for step in range(max_tokens):
            logits, cache = generate_step(model, token, cache)

    Or for compatibility with existing code (no memory savings):
        result = compress_cache(cache, model=model, bits=4, compact=False)
        # cache still has dequantized FP16, use standard generation

    Args:
        cache: List of MLX KVCache objects from make_prompt_cache().
        model: MLX model (for auto-detecting head_dim).
        head_dim: Override head_dim if model not provided.
        bits: Quantization bits (2, 3, or 4). Default: 4.
        window_size: Keep this many recent tokens uncompressed. Default: 0.
        min_context: Skip compression if context shorter than this. Default: 0.
        compact: If True (default), free FP16 tensors for real memory savings.
                 Use generate_step() for generation. If False, write dequantized
                 FP16 back (no memory savings, but works with standard generation).

    Returns:
        Dict with cosine, compress_ms, layers_compressed, memory stats.
    """
    if head_dim is None and model is not None:
        head_dim = get_head_dim(model)
    if head_dim is None:
        raise ValueError("Provide head_dim or model")

    t0 = time.time()
    total_orig = 0
    total_comp = 0
    layers_compressed = 0
    cosine_sum = 0.0
    cosine_count = 0

    # Pre-build all compressors (codebook cached, rotation cached)
    compressors = {}
    for li in range(len(cache)):
        c = cache[li]
        if not hasattr(c, 'keys') or c.keys is None:
            continue
        kd = c.keys.shape[-1]
        if kd < 2:
            continue
        compressors[li] = (
            PolarQuantMLX(kd, bits=bits, seed=42 + li),
            PolarQuantMLX(kd, bits=bits, seed=1000 + li),
        )

    for li, (k_mse, v_mse) in compressors.items():
        c = cache[li]
        seq = c.offset
        if seq <= min_context:
            continue

        compress_end = seq - window_size if window_size > 0 else seq
        if compress_end <= 0:
            continue

        k = c.keys[:, :, :compress_end, :]
        v = c.values[:, :, :compress_end, :]
        kd = k.shape[-1]

        # ── Compress keys: normalize → quantize → store indices + norms ──
        k_f = k.astype(mx.float32)
        k_norms = mx.maximum(mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True)), 1e-8)
        k_unit = k_f / k_norms
        k_indices = k_mse.quantize(k_unit)

        # ── Compress values: normalize → quantize → store indices + norms ──
        v_f = v.astype(mx.float32)
        v_norms = mx.maximum(mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True)), 1e-8)
        v_unit = v_f / v_norms
        v_indices = v_mse.quantize(v_unit)

        # ── Measure real cosine BEFORE overwriting cache ──
        # Compare original unit vectors vs dequantized unit vectors
        k_recon_unit = k_mse.dequantize(k_indices)
        layer_cos = mx.mean(mx.sum(k_unit * k_recon_unit, axis=-1))

        # ── Per-layer eval: prevents memory graph accumulation across layers ──
        mx.eval(k_indices, k_norms, v_indices, v_norms, layer_cos)
        cosine_sum += layer_cos.item()
        cosine_count += 1

        # ── Pack indices for real memory savings ──
        # 4-bit: 2 per byte (4x vs FP16), 3-bit: 8 per 3 bytes (5.3x vs FP16)
        k_packed = pack_indices(k_indices, bits)
        v_packed = pack_indices(v_indices, bits)
        mx.eval(k_packed, v_packed)

        # ── Store compressed representation on the cache object ──
        c._tq_k_indices = k_packed                                     # packed bits
        c._tq_k_norms = k_norms.astype(mx.float16)                    # float16
        c._tq_v_indices = v_packed                                     # packed bits
        c._tq_v_norms = v_norms.astype(mx.float16)                    # float16
        c._tq_head_dim = kd                                            # for unpack
        c._tq_k_compressor = k_mse                                    # for dequantize
        c._tq_v_compressor = v_mse                                    # for dequantize
        c._tq_compress_end = compress_end                              # how many tokens compressed
        c._tq_bits = bits

        # ── Dequantize and write back for immediate use ──
        # This ensures the cache works with standard attention without any
        # code changes downstream. The compressed data is also stored above
        # for real memory savings when decompress_cache() is used.
        k_hat = (k_recon_unit * k_norms).astype(k.dtype)
        v_hat = (v_mse.dequantize(v_indices) * v_norms).astype(v.dtype)
        c.keys[:, :, :compress_end, :] = k_hat
        c.values[:, :, :compress_end, :] = v_hat
        mx.eval(c.keys, c.values)

        layers_compressed += 1

    # ── Measure FP16 memory BEFORE compacting (used portion only) ──
    original_bytes = 0
    for c in cache:
        if hasattr(c, 'keys') and c.keys is not None:
            used = getattr(c, 'offset', c.keys.shape[2])
            original_bytes += c.keys[:, :, :used, :].nbytes + c.values[:, :, :used, :].nbytes

    # ── Compact or restore ──
    if compact and layers_compressed > 0:
        # Free FP16 tensors — real memory savings.
        # Use generate_step() for generation (decompresses on-demand per step).
        compact_cache(cache)
    elif not compact and layers_compressed > 0:
        # Write dequantized FP16 back — no memory savings, but compatible
        # with standard generation loop (model(token, cache=cache)).
        pass  # FP16 already written back in the per-layer loop above

    elapsed_ms = (time.time() - t0) * 1000

    # Real cosine: averaged across all compressed layers (measured before overwrite)
    cos = cosine_sum / cosine_count if cosine_count > 0 else 0.0

    # ── Measure compressed representation (indices + norms) ──
    compressed_bytes = 0
    for c in cache:
        if hasattr(c, '_tq_k_indices') and c._tq_k_indices is not None:
            compressed_bytes += (
                c._tq_k_indices.nbytes + c._tq_k_norms.nbytes +
                c._tq_v_indices.nbytes + c._tq_v_norms.nbytes
            )

    original_mb = original_bytes / 1024 / 1024
    compressed_mb = compressed_bytes / 1024 / 1024
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0

    return {
        "cosine": round(cos, 4),
        "compress_ms": round(elapsed_ms, 0),
        "layers_compressed": layers_compressed,
        "original_mb": round(original_mb, 1),
        "compressed_mb": round(compressed_mb, 1),
        "ratio": round(ratio, 1),
    }


def compact_cache(cache: List[Any]) -> Dict:
    """
    Replace FP16 keys/values with indices+norms for real memory savings.

    Call this AFTER compress_cache() to free the FP16 tensors and keep only
    the compressed representation. Generation still works because
    restore_cache() reconstructs FP16 on-demand before each forward pass.

    Args:
        cache: List of KVCache objects (already compressed via compress_cache).

    Returns:
        Dict with actual memory saved.

    Example::

        compress_cache(cache, model=model, bits=4)
        savings = compact_cache(cache)  # free FP16, keep indices+norms
        # Before each generation step, call restore_cache(cache)
    """
    freed_bytes = 0
    compacted = 0

    for c in cache:
        if not hasattr(c, '_tq_k_indices'):
            continue

        end = c._tq_compress_end

        # Save the uncompressed window (if any)
        if end < c.offset:
            c._tq_window_keys = mx.array(c.keys[:, :, end:, :])
            c._tq_window_values = mx.array(c.values[:, :, end:, :])
            mx.eval(c._tq_window_keys, c._tq_window_values)

        # Measure what we're freeing
        freed_bytes += c.keys.nbytes + c.values.nbytes

        # Drop FP16 tensors
        c.keys = None
        c.values = None
        c._tq_compacted = True

        compacted += 1

    # Measure what remains (indices + norms)
    retained_bytes = 0
    for c in cache:
        if hasattr(c, '_tq_k_indices') and c._tq_k_indices is not None:
            retained_bytes += c._tq_k_indices.nbytes + c._tq_k_norms.nbytes
            retained_bytes += c._tq_v_indices.nbytes + c._tq_v_norms.nbytes
            if hasattr(c, '_tq_window_keys') and c._tq_window_keys is not None:
                retained_bytes += c._tq_window_keys.nbytes + c._tq_window_values.nbytes

    return {
        "layers_compacted": compacted,
        "freed_mb": round(freed_bytes / 1024 / 1024, 1),
        "retained_mb": round(retained_bytes / 1024 / 1024, 1),
        "actual_saved_mb": round((freed_bytes - retained_bytes) / 1024 / 1024, 1),
    }


def restore_cache(cache: List[Any]) -> None:
    """
    Reconstruct FP16 keys/values from stored indices+norms.

    Call this before model forward pass if compact_cache() was used.
    This is the decompress-on-demand step.

    Args:
        cache: List of KVCache objects (compacted via compact_cache).
    """
    for c in cache:
        if not getattr(c, '_tq_compacted', False):
            continue

        end = c._tq_compress_end
        k_mse = c._tq_k_compressor
        v_mse = c._tq_v_compressor

        # Unpack packed indices back to uint8 for dequantization
        bits = c._tq_bits
        hd = c._tq_head_dim
        k_indices = unpack_indices(c._tq_k_indices, bits, hd)
        v_indices = unpack_indices(c._tq_v_indices, bits, hd)

        # Dequantize compressed region
        k_hat = k_mse.dequantize(k_indices)
        k_hat = (k_hat * c._tq_k_norms.astype(mx.float32)).astype(mx.float16)

        v_hat = v_mse.dequantize(v_indices)
        v_hat = (v_hat * c._tq_v_norms.astype(mx.float32)).astype(mx.float16)

        # Append uncompressed window if present
        if hasattr(c, '_tq_window_keys') and c._tq_window_keys is not None:
            k_hat = mx.concatenate([k_hat, c._tq_window_keys], axis=2)
            v_hat = mx.concatenate([v_hat, c._tq_window_values], axis=2)

        c.keys = k_hat
        c.values = v_hat
        mx.eval(c.keys, c.values)

        c._tq_compacted = False


# ═══════════════════════════════════════════════════════
#  Production generation with compressed cache
# ═══════════════════════════════════════════════════════

def generate_step(model, token_id, cache):
    """
    One generation step with compressed KV cache.

    Decompresses all layers → model forward → quantizes new token → recompacts.

    Memory behavior: during the forward pass, full FP16 KV is temporarily in
    memory (same as standard generation). After the step, FP16 is freed and
    only compressed indices remain. Net savings = between steps, not during.

    NOTE: Only works with window_size=0 (default). If compress_cache was called
    with window_size > 0, use compact=False instead and generate normally.

    Args:
        model: MLX model.
        token_id: mx.array of shape (1, 1) — the token to process.
        cache: List of KVCache objects (compacted via compress_cache with compact=True).

    Returns:
        logits: mx.array — logits for next token prediction.

    Example::

        cache = make_prompt_cache(model)
        logits = chunked_prefill(model, ids, cache)
        compress_cache(cache, model=model, bits=4)  # compact=True, window_size=0

        y = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(max_tokens):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            logits = generate_step(model, y[:, None], cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
    """
    if isinstance(token_id, int):
        token_id = mx.array([[token_id]])

    # window_size stores an uncompressed tail separately. That layout is not
    # compatible with the incremental append logic below.
    for c in cache:
        if getattr(c, '_tq_compacted', False) and getattr(c, '_tq_window_keys', None) is not None:
            raise ValueError(
                "generate_step only supports window_size=0. "
                "Use compress_cache(..., compact=False) with the standard "
                "generation loop when window_size > 0."
            )

    # Restore FP16 from compressed indices (temporary, for this forward pass)
    any_compacted = any(getattr(c, '_tq_compacted', False) for c in cache)
    if any_compacted:
        restore_cache(cache)

    # Forward pass — standard attention with dequantized FP16
    logits = model(token_id, cache=cache)
    mx.eval(logits)

    # Recompress: quantize new token, append to compressed storage, free FP16.
    if any_compacted:
        for c in cache:
            if not hasattr(c, '_tq_k_compressor'):
                continue

            k_mse = c._tq_k_compressor
            v_mse = c._tq_v_compressor
            bits = c._tq_bits

            # Read new token at exact offset position (NOT -1, because KVCache pre-allocates)
            pos = c.offset - 1
            new_k = c.keys[:, :, pos:pos+1, :].astype(mx.float32)
            new_v = c.values[:, :, pos:pos+1, :].astype(mx.float32)

            k_norm = mx.maximum(mx.sqrt(mx.sum(new_k * new_k, axis=-1, keepdims=True)), 1e-8)
            k_idx = k_mse.quantize(new_k / k_norm)
            k_packed = pack_indices(k_idx, bits)

            v_norm = mx.maximum(mx.sqrt(mx.sum(new_v * new_v, axis=-1, keepdims=True)), 1e-8)
            v_idx = v_mse.quantize(new_v / v_norm)
            v_packed = pack_indices(v_idx, bits)

            # Append to compressed storage
            c._tq_k_indices = mx.concatenate([c._tq_k_indices, k_packed], axis=2)
            c._tq_k_norms = mx.concatenate([c._tq_k_norms, k_norm.astype(mx.float16)], axis=2)
            c._tq_v_indices = mx.concatenate([c._tq_v_indices, v_packed], axis=2)
            c._tq_v_norms = mx.concatenate([c._tq_v_norms, v_norm.astype(mx.float16)], axis=2)
            c._tq_compress_end = c.offset

            # Clear window buffer (no longer valid after new tokens appended)
            c._tq_window_keys = None
            c._tq_window_values = None

            mx.eval(c._tq_k_indices, c._tq_k_norms, c._tq_v_indices, c._tq_v_norms)

            # Free FP16 — back to compressed only
            c.keys = None
            c.values = None
            c._tq_compacted = True

    return logits


# ═══════════════════════════════════════════════════════
#  Fused generation — no FP16 materialization
# ═══════════════════════════════════════════════════════

def generate_step_fused(model, token_id, cache):
    """Generate one token using fused attention on compressed KV cache.

    Unlike generate_step(), this NEVER materializes full FP16 K/V.
    Attention scores are computed directly on compressed indices via
    pre-rotated queries and centroid lookups.

    Memory: no FP16 spike. Only transient centroid gather per layer.
    Speed: eliminates decompress/recompress cycle entirely.

    Args:
        model: MLX model (must be patched via patch_model_fused first).
        token_id: mx.array of shape (1, 1).
        cache: List of KVCache objects (compacted via compress_cache with compact=True).

    Returns:
        logits: mx.array

    Example::
        cache = make_prompt_cache(model)
        logits = chunked_prefill(model, ids, cache)
        compress_cache(cache, model=model, bits=4)
        patch_model_fused(model)

        y = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(max_tokens):
            tok = y.item()
            if tok == tokenizer.eos_token_id:
                break
            logits = generate_step_fused(model, y[:, None], cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
    """
    if isinstance(token_id, int):
        token_id = mx.array([[token_id]])

    # Forward pass — patched attention computes directly on compressed indices
    logits = model(token_id, cache=cache)
    mx.eval(logits)

    # After forward, new K/V token was appended to FP16 cache by the model.
    # Compress it and append to packed indices, then free FP16.
    for c in cache:
        if not hasattr(c, '_tq_k_compressor'):
            continue

        k_mse = c._tq_k_compressor
        v_mse = c._tq_v_compressor
        bits = c._tq_bits

        # New token at exact offset position
        pos = c.offset - 1
        new_k = c.keys[:, :, pos:pos+1, :].astype(mx.float32)
        new_v = c.values[:, :, pos:pos+1, :].astype(mx.float32)

        # Quantize new K
        k_norm = mx.maximum(mx.sqrt(mx.sum(new_k * new_k, axis=-1, keepdims=True)), 1e-8)
        k_idx = k_mse.quantize(new_k / k_norm)
        k_packed = pack_indices(k_idx, bits)

        # Quantize new V
        v_norm = mx.maximum(mx.sqrt(mx.sum(new_v * new_v, axis=-1, keepdims=True)), 1e-8)
        v_idx = v_mse.quantize(new_v / v_norm)
        v_packed = pack_indices(v_idx, bits)

        # Append to compressed storage
        c._tq_k_indices = mx.concatenate([c._tq_k_indices, k_packed], axis=2)
        c._tq_k_norms = mx.concatenate([c._tq_k_norms, k_norm.astype(mx.float16)], axis=2)
        c._tq_v_indices = mx.concatenate([c._tq_v_indices, v_packed], axis=2)
        c._tq_v_norms = mx.concatenate([c._tq_v_norms, v_norm.astype(mx.float16)], axis=2)
        c._tq_compress_end = c.offset

        mx.eval(c._tq_k_indices, c._tq_k_norms, c._tq_v_indices, c._tq_v_norms)

        # Free FP16
        c.keys = None
        c.values = None
        c._tq_compacted = True

    return logits


def patch_model_fused(model):
    """Patch model's SDPA function to use fused compressed attention.

    Intercepts at the scaled_dot_product_attention level — works with any
    model architecture (Qwen, Gemma, Llama, etc.) without knowing attribute names.

    After patching, any attention call that receives a cache with compressed
    data will use fused_turboquant_sdpa instead of standard SDPA.

    Call once after compress_cache(), before generation loop.
    """
    import sys
    from .fused_attention import fused_turboquant_sdpa

    # The replacement SDPA that checks for compressed cache
    _original_sdpa = None

    def _patched_sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kwargs):
        # Check if cache has compressed data
        if cache is not None and hasattr(cache, '_tq_k_indices') and cache._tq_k_indices is not None:
            # Split: compressed portion vs new FP16 tokens
            n_compressed = cache._tq_compress_end
            n_total = keys.shape[2]

            if n_total > n_compressed:
                new_k = keys[:, :, n_compressed:, :]
                new_v = values[:, :, n_compressed:, :]
            else:
                new_k = None
                new_v = None

            return fused_turboquant_sdpa(
                queries, cache, scale=scale, mask=mask,
                new_keys=new_k, new_values=new_v)

        # No compressed data — use original SDPA
        return _original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, **kwargs)

    # Patch scaled_dot_product_attention in mlx_lm.models.base
    import mlx_lm.models.base as base_module
    _original_sdpa = base_module.scaled_dot_product_attention
    base_module.scaled_dot_product_attention = _patched_sdpa

    # Python's `from X import Y` creates local bindings — patch all loaded model modules
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", _patched_sdpa)

    model._tq_fused_patched = True
    model._tq_original_sdpa = _original_sdpa
    return sum(1 for _ in model.layers)


# ═══════════════════════════════════════════════════════
#  Patched model attention (experimental, non-fused)
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
    Monkey-patch model attention to use TurboQuantCache with SDPA.

    NOTE: This is experimental. The recommended path is:
      cache = make_prompt_cache(model)
      compress_cache(cache, model=model, bits=bits)
    which works without patching.

    After patching, use make_turboquant_cache() to create the cache, then
    model(ids, cache=cache) automatically compresses KV.
    """
    import types
    from .cache import TurboQuantCache
    from .attention import turboquant_sdpa

    head_dim = get_head_dim(model)
    patched = 0

    for i, layer in enumerate(model.layers):
        attn = None
        attn_name = None
        for name in ['self_attn', 'attention', 'attn']:
            if hasattr(layer, name):
                attn = getattr(layer, name)
                attn_name = name
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

        # Patch the class method on this specific instance's class to ensure
        # Python's method resolution finds it. Create a per-instance subclass.
        AttnClass = type(attn)
        patched_cls = type(f"Patched{AttnClass.__name__}", (AttnClass,), {
            "__call__": make_patched(attn),
        })
        attn.__class__ = patched_cls
        patched += 1

    return patched
