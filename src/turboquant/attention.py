"""
TurboQuant Attention — scaled dot-product attention for dequantized KV cache.

Current path: standard SDPA with MSE-dequantized keys/values.
QJL correction is implemented in qjl.py but disabled — 4-bit MSE provides
sufficient quality (0.991+ cosine) without the added variance.
"""

import math
import mlx.core as mx


def turboquant_sdpa(queries, keys, values, cache, scale, mask=None):
    """
    Attention with dequantized KV cache.

    Args:
        queries: (B, n_heads, seq_q, head_dim)
        keys: (B, n_kv_heads, seq_kv, head_dim) — MSE-dequantized
        values: (B, n_kv_heads, seq_kv, head_dim) — MSE-dequantized
        cache: TurboQuantCache
        scale: 1/sqrt(head_dim)
        mask: attention mask
    """
    B, n_heads, seq_q, d = queries.shape
    _, n_kv_heads, seq_kv, _ = keys.shape
    n_rep = n_heads // n_kv_heads

    return _python_sdpa(queries, keys, values, n_rep, scale, mask)


def _python_sdpa(queries, keys, values, n_rep, scale, mask):
    """Standard scaled dot-product attention."""
    if n_rep > 1:
        keys = mx.repeat(keys, n_rep, axis=1)
        values = mx.repeat(values, n_rep, axis=1)

    scores = (queries.astype(mx.float32) @ mx.transpose(keys.astype(mx.float32), (0, 1, 3, 2))) * scale

    if mask is not None:
        if isinstance(mask, mx.array):
            scores = scores + mask.astype(scores.dtype)

    weights = mx.softmax(scores, axis=-1)
    output = weights @ values.astype(mx.float32)
    return output.astype(queries.dtype)
