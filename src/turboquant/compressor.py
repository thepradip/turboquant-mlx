"""
Stage 1: PolarQuant MSE compression — pure MLX.
Rotate → quantize → return indices + residual for QJL.

Codebook is cached: built once per (bits, head_dim), reused across all layers.
Only rotation matrix differs per layer (different seed).

v0.5.0 — Cumulative-sum quantization: instead of building the full
(..., head_dim, n_centroids) distance tensor, we compare against each
boundary and sum the boolean results. For 4-bit this is 15 comparisons
vs a 16-wide broadcast, eliminating the 536 MB/layer intermediate.
"""

import mlx.core as mx
import numpy as np
from .codebook import build_codebook, build_rotation

# Global caches
_codebook_cache = {}   # (bits, head_dim) → mx.array centroids
_boundary_cache = {}   # (bits, head_dim) → mx.array decision boundaries
_rotation_cache = {}   # (head_dim, seed) → (rotation, rotation_t)


def _get_codebook(head_dim: int, bits: int) -> mx.array:
    """Get or build codebook. Cached — same codebook for all layers."""
    key = (bits, head_dim)
    if key not in _codebook_cache:
        centroids_np = build_codebook(bits, head_dim)
        _codebook_cache[key] = mx.array(centroids_np)
        # Precompute decision boundaries (midpoints between sorted centroids)
        boundaries_np = (centroids_np[:-1] + centroids_np[1:]) / 2.0
        _boundary_cache[key] = mx.array(boundaries_np)
        mx.eval(_codebook_cache[key], _boundary_cache[key])
    return _codebook_cache[key]


def _get_boundaries(head_dim: int, bits: int) -> mx.array:
    """Get precomputed decision boundaries for cumsum quantization."""
    _get_codebook(head_dim, bits)  # ensure built
    return _boundary_cache[(bits, head_dim)]


def _get_rotation(head_dim: int, seed: int):
    """Get or build rotation. Cached per (head_dim, seed)."""
    key = (head_dim, seed)
    if key not in _rotation_cache:
        r = mx.array(build_rotation(head_dim, seed))
        rt = mx.transpose(r)
        mx.eval(r, rt)
        _rotation_cache[key] = (r, rt)
    return _rotation_cache[key]


def _quantize_cumsum(y: mx.array, boundaries: mx.array) -> mx.array:
    """
    Quantize by counting how many boundaries each value exceeds.

    For sorted boundaries [b0, b1, ..., b_{n-1}], the bin index for value v is:
        sum(v >= b_i for i in 0..n-1)

    This avoids building the (..., head_dim, n_centroids) distance tensor.
    Each comparison produces a same-shape boolean, summed in-place.

    Memory: O(input_size) vs O(input_size × n_centroids) for brute-force.
    """
    # Start with zeros, add 1 for each boundary exceeded
    indices = mx.zeros(y.shape, dtype=mx.uint8)
    for i in range(boundaries.shape[0]):
        indices = indices + (y >= boundaries[i]).astype(mx.uint8)
    return indices


class PolarQuantMLX:
    """
    Stage 1 of TurboQuant: MSE-optimal quantization.
    Rotates vectors, quantizes per-coordinate with Lloyd-Max codebook.

    Codebook is shared across all instances with same (bits, head_dim).
    Only rotation matrix is unique per layer (via seed).

    v0.5.0: Uses cumulative boundary comparisons instead of brute-force
    argmin over all centroids. For 4-bit (15 boundaries), peak memory
    is O(input_size) instead of O(input_size × 16).
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42):
        self.head_dim = head_dim
        self.bits = bits

        # Shared codebook (cached)
        self.centroids = _get_codebook(head_dim, bits)

        # Decision boundaries for quantization (cached)
        self.boundaries = _get_boundaries(head_dim, bits)

        # Per-layer rotation (cached per seed)
        self.rotation, self.rotation_t = _get_rotation(head_dim, seed)

    def quantize(self, x: mx.array):
        """Quantize to indices via boundary comparisons. Input: (..., head_dim). Returns uint8."""
        x = x.astype(mx.float32)
        y = x @ self.rotation_t
        return _quantize_cumsum(y, self.boundaries)

    def dequantize(self, indices: mx.array) -> mx.array:
        """Dequantize indices back to vectors. Returns float32."""
        y_hat = self.centroids[indices.astype(mx.uint32)]
        return y_hat @ self.rotation

    def quantize_with_residual(self, x: mx.array):
        """Quantize and return (indices, residual)."""
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        residual = x.astype(mx.float32) - x_hat
        return indices, residual
