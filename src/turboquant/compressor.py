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


# ═══════════════════════════════════════════════════════
#  Bit-packing: uint8 indices → packed bits for real savings
# ═══════════════════════════════════════════════════════

def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack uint8 indices into dense bit representation.

    4-bit: 2 indices per byte (nibble packing)  → 2x smaller
    3-bit: 8 indices per 3 bytes (24 bits)      → 2.67x smaller
    2-bit: 4 indices per byte                   → 4x smaller

    Args:
        indices: uint8 array, last dim = head_dim
        bits: quantization bits (2, 3, or 4)

    Returns:
        Packed uint8 array with smaller last dim.
    """
    shape = indices.shape          # (..., head_dim)
    flat = indices.reshape(-1, shape[-1])  # (N, head_dim)
    n, d = flat.shape

    if bits == 4:
        # Pack 2 nibbles per byte: high|low
        assert d % 2 == 0, f"head_dim {d} must be even for 4-bit packing"
        lo = flat[:, 0::2]             # even indices
        hi = flat[:, 1::2]             # odd indices
        packed = (hi << 4) | (lo & 0x0F)
        packed = packed.astype(mx.uint8)
        return packed.reshape(*shape[:-1], d // 2)

    elif bits == 3:
        # Pack 8 values (24 bits) into 3 bytes
        # Pad head_dim to multiple of 8
        pad = (8 - d % 8) % 8
        if pad > 0:
            flat = mx.concatenate([flat, mx.zeros((n, pad), dtype=mx.uint8)], axis=-1)
            d = d + pad
        groups = d // 8
        flat = flat.reshape(n, groups, 8)  # (N, groups, 8)
        # Each value is 0..7 (3 bits). Pack 8 values into 3 bytes (24 bits).
        b0 = (flat[:, :, 0] | (flat[:, :, 1] << 3) | (flat[:, :, 2] << 6)).astype(mx.uint8)
        b1 = ((flat[:, :, 2] >> 2) | (flat[:, :, 3] << 1) | (flat[:, :, 4] << 4) | (flat[:, :, 5] << 7)).astype(mx.uint8)
        b2 = ((flat[:, :, 5] >> 1) | (flat[:, :, 6] << 2) | (flat[:, :, 7] << 5)).astype(mx.uint8)
        packed = mx.stack([b0, b1, b2], axis=-1).reshape(n, groups * 3)
        return packed.reshape(*shape[:-1], groups * 3)

    elif bits == 2:
        # Pack 4 values per byte
        assert d % 4 == 0, f"head_dim {d} must be multiple of 4 for 2-bit packing"
        flat = flat.reshape(n, d // 4, 4)
        packed = (flat[:, :, 0] | (flat[:, :, 1] << 2) | (flat[:, :, 2] << 4) | (flat[:, :, 3] << 6)).astype(mx.uint8)
        return packed.reshape(*shape[:-1], d // 4)

    else:
        return indices  # no packing for other bit widths


def unpack_indices(packed: mx.array, bits: int, head_dim: int) -> mx.array:
    """Unpack dense bit representation back to uint8 indices.

    Args:
        packed: packed uint8 array
        bits: quantization bits (2, 3, or 4)
        head_dim: original last dimension size

    Returns:
        uint8 array with last dim = head_dim
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])
    n = flat.shape[0]

    if bits == 4:
        lo = flat & 0x0F
        hi = (flat >> 4) & 0x0F
        # Interleave: lo[0], hi[0], lo[1], hi[1], ...
        unpacked = mx.zeros((n, head_dim), dtype=mx.uint8)
        unpacked = unpacked.at[:, 0::2].add(lo)
        unpacked = unpacked.at[:, 1::2].add(hi)
        return unpacked.reshape(*shape[:-1], head_dim)

    elif bits == 3:
        padded_dim = ((head_dim + 7) // 8) * 8
        groups = padded_dim // 8
        flat = flat.reshape(n, groups, 3)
        b0, b1, b2 = flat[:, :, 0], flat[:, :, 1], flat[:, :, 2]
        mask3 = mx.array(0x07, dtype=mx.uint8)
        v0 = b0 & mask3
        v1 = (b0 >> 3) & mask3
        v2 = ((b0 >> 6) | (b1 << 2)) & mask3
        v3 = (b1 >> 1) & mask3
        v4 = (b1 >> 4) & mask3
        v5 = ((b1 >> 7) | (b2 << 1)) & mask3
        v6 = (b2 >> 2) & mask3
        v7 = (b2 >> 5) & mask3
        unpacked = mx.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=-1).reshape(n, padded_dim)
        return unpacked[:, :head_dim].reshape(*shape[:-1], head_dim)

    elif bits == 2:
        flat_out = mx.stack([
            flat & 0x03,
            (flat >> 2) & 0x03,
            (flat >> 4) & 0x03,
            (flat >> 6) & 0x03,
        ], axis=-1).reshape(n, -1)
        return flat_out[:, :head_dim].reshape(*shape[:-1], head_dim)

    else:
        return packed
