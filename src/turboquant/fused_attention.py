"""
Fused TurboQuant Attention — compute Q@K^T directly on compressed indices.

Never materializes full FP16 K/V for compressed tokens.

Math (from TurboQuant paper, ICLR 2026):
  score_ij = Q_i @ K_hat_j
           = Q_i @ (R^T @ centroids[k_idx_j]) * k_norm_j
           = (Q_i @ R^T) @ centroids[k_idx_j] * k_norm_j

  Pre-rotate Q once per layer: Q_rot = Q @ R_k^T
  Then per K position: centroid gather + dot product (no rotation on K)

  Value aggregation in rotated space:
    output_rot = weights @ (centroids[v_idx] * v_norms)
    output = output_rot @ R_v   (rotate back once)
"""

import mlx.core as mx
from .compressor import unpack_indices


def prerotate_queries(queries, rotation_t):
    """Pre-rotate queries by R^T. Done once per layer per step.

    Args:
        queries: (B, n_heads, seq_q, head_dim) float32
        rotation_t: (head_dim, head_dim) float32 — transpose of rotation matrix

    Returns:
        (B, n_heads, seq_q, head_dim) float32
    """
    return queries.astype(mx.float32) @ rotation_t


def compressed_scores(q_rot, packed_k_indices, k_norms, centroids, bits, head_dim, n_rep, scale):
    """Compute attention scores for compressed K positions.

    No rotation on K side — Q was pre-rotated instead.

    Args:
        q_rot: (B, n_heads, seq_q, head_dim) float32 — pre-rotated queries
        packed_k_indices: (B, n_kv, seq_compressed, packed_dim) uint8
        k_norms: (B, n_kv, seq_compressed, 1) float16
        centroids: (n_centroids,) float32
        bits: int (2, 3, or 4)
        head_dim: int
        n_rep: int — GQA factor (n_heads // n_kv_heads)
        scale: float — 1/sqrt(head_dim)

    Returns:
        (B, n_heads, seq_q, seq_compressed) float32
    """
    # Unpack bit-packed indices
    k_idx = unpack_indices(packed_k_indices, bits, head_dim)

    # Gather centroids — each element becomes one of 2^bits float32 values
    # Shape: (B, n_kv, seq_compressed, head_dim) float32
    k_centroid_vals = centroids[k_idx]

    # Scale by norms: (B, n_kv, seq_compressed, head_dim) * (B, n_kv, seq_compressed, 1)
    k_scaled = k_centroid_vals * k_norms.astype(mx.float32)

    # GQA: expand KV heads to match query heads
    if n_rep > 1:
        k_scaled = mx.repeat(k_scaled, n_rep, axis=1)

    # Scores: Q_rot @ K_scaled^T
    # (B, n_heads, seq_q, head_dim) @ (B, n_heads, head_dim, seq_compressed)
    scores = (q_rot @ mx.transpose(k_scaled, (0, 1, 3, 2))) * scale

    return scores


def compressed_value_aggregate(weights, packed_v_indices, v_norms, centroids_v,
                                rotation_v, bits, head_dim, n_rep):
    """Weighted sum of compressed values in rotated space, rotate back once.

    Args:
        weights: (B, n_heads, seq_q, seq_compressed) float32
        packed_v_indices: (B, n_kv, seq_compressed, packed_dim) uint8
        v_norms: (B, n_kv, seq_compressed, 1) float16
        centroids_v: (n_centroids,) float32
        rotation_v: (head_dim, head_dim) float32 — rotation matrix (NOT transposed)
        bits: int
        head_dim: int
        n_rep: int

    Returns:
        (B, n_heads, seq_q, head_dim) float32
    """
    # Unpack and gather centroids
    v_idx = unpack_indices(packed_v_indices, bits, head_dim)
    v_centroid_vals = centroids_v[v_idx]

    # Scale by norms in rotated space
    v_scaled = v_centroid_vals * v_norms.astype(mx.float32)

    # GQA
    if n_rep > 1:
        v_scaled = mx.repeat(v_scaled, n_rep, axis=1)

    # Weighted sum in rotated space
    # weights: (B, n_heads, seq_q, seq_compressed)
    # v_scaled: (B, n_heads, seq_compressed, head_dim)
    output_rot = weights @ v_scaled

    # Rotate back once: (B, n_heads, seq_q, head_dim) @ (head_dim, head_dim)
    output = output_rot @ rotation_v

    return output


def fused_turboquant_sdpa(queries, cache_layer, scale, mask=None,
                           new_keys=None, new_values=None):
    """Fused attention: compute scores directly on compressed KV cache indices.

    Never materializes full FP16 K/V for compressed (prefill) tokens.
    Handles mixed compressed + uncompressed (new) tokens.

    Args:
        queries: (B, n_heads, seq_q, head_dim) — from model projection
        cache_layer: cache object with _tq_k_indices, _tq_k_norms, etc.
        scale: 1/sqrt(head_dim)
        mask: attention mask or None
        new_keys: (B, n_kv, n_new, head_dim) float16 — recent uncompressed tokens
        new_values: (B, n_kv, n_new, head_dim) float16 — recent uncompressed tokens

    Returns:
        (B, n_heads, seq_q, head_dim) — attention output
    """
    B, n_heads, seq_q, d = queries.shape
    k_compressor = cache_layer._tq_k_compressor
    v_compressor = cache_layer._tq_v_compressor
    bits = cache_layer._tq_bits
    head_dim = cache_layer._tq_head_dim
    n_kv_heads = cache_layer._tq_k_indices.shape[1]
    n_rep = n_heads // n_kv_heads

    has_compressed = cache_layer._tq_k_indices is not None
    has_new = new_keys is not None and new_keys.shape[2] > 0

    all_scores = []

    # ── Compressed token scores ──
    if has_compressed:
        q_rot_k = prerotate_queries(queries, k_compressor.rotation_t)
        scores_c = compressed_scores(
            q_rot_k, cache_layer._tq_k_indices, cache_layer._tq_k_norms,
            k_compressor.centroids, bits, head_dim, n_rep, scale)
        all_scores.append(scores_c)

    # ── New (uncompressed) token scores ──
    if has_new:
        new_k = new_keys.astype(mx.float32)
        if n_rep > 1:
            new_k = mx.repeat(new_k, n_rep, axis=1)
        scores_n = (queries.astype(mx.float32) @ mx.transpose(new_k, (0, 1, 3, 2))) * scale
        all_scores.append(scores_n)

    # ── Concatenate and softmax ──
    scores = mx.concatenate(all_scores, axis=-1) if len(all_scores) > 1 else all_scores[0]

    if mask is not None:
        if isinstance(mask, mx.array):
            scores = scores + mask.astype(scores.dtype)

    weights = mx.softmax(scores, axis=-1)

    # ── Value aggregation ──
    output = mx.zeros((B, n_heads, seq_q, head_dim), dtype=mx.float32)

    if has_compressed:
        seq_c = cache_layer._tq_k_indices.shape[2]
        weights_c = weights[:, :, :, :seq_c]
        output_c = compressed_value_aggregate(
            weights_c, cache_layer._tq_v_indices, cache_layer._tq_v_norms,
            v_compressor.centroids, v_compressor.rotation, bits, head_dim, n_rep)
        output = output + output_c

    if has_new:
        seq_c = cache_layer._tq_k_indices.shape[2] if has_compressed else 0
        weights_n = weights[:, :, :, seq_c:]
        new_v = new_values.astype(mx.float32)
        if n_rep > 1:
            new_v = mx.repeat(new_v, n_rep, axis=1)
        output_n = weights_n @ new_v
        output = output + output_n

    return output.astype(queries.dtype)
