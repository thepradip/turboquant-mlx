"""
TurboQuant KV Cache — stores compressed keys/values with QJL correction data.

v0.5.0: Stores uint8 indices + float16 norms (real memory savings).
Dequantizes on-the-fly when returning keys/values to the model.

Keys: MSE indices + norms + optional QJL signs + residual norms
Values: MSE indices + norms
"""

import mlx.core as mx
from .compressor import PolarQuantMLX
from .qjl import QJLMLX


class TurboQuantCache:
    """
    Drop-in KVCache replacement that compresses on insert.

    v0.5.0: Stores uint8 indices + float16 norms internally.
    Returns dequantized FP16 tensors to the model (same API as KVCache).

    Shapes match MLX's KVCache exactly:
      keys/values returned: (batch, n_kv_heads, seq_len, head_dim)
    """

    step = 256  # MLX checks this

    def __init__(self, head_dim: int, key_bits: int = 4, value_bits: int = 4,
                 layer_idx: int = 0, use_qjl: bool = False):
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.use_qjl = use_qjl

        # Full bits for MSE — quality over compression
        self.key_mse = PolarQuantMLX(head_dim, key_bits, seed=42 + layer_idx)
        self.val_mse = PolarQuantMLX(head_dim, value_bits, seed=1000 + layer_idx)

        # QJL only if enabled (disabled by default — adds variance, see README)
        if use_qjl:
            self.key_qjl = QJLMLX(head_dim, seed=43 + layer_idx)

        # Compressed storage (real memory savings)
        self.k_indices = None    # uint8: (B, n_kv, seq, head_dim)
        self.k_norms = None      # float16: (B, n_kv, seq, 1)
        self.v_indices = None    # uint8: (B, n_kv, seq, head_dim)
        self.v_norms = None      # float16: (B, n_kv, seq, 1)
        self.offset = 0

        # QJL correction data (only if use_qjl=True)
        self.qjl_signs = None       # (B, n_kv, seq, m) float32 ±1
        self.qjl_res_norms = None   # (B, n_kv, seq) float16
        self.qjl_key_norms = None   # (B, n_kv, seq) float16

    def _dequantize_keys(self) -> mx.array:
        """Reconstruct FP16 keys from indices + norms."""
        k_hat = self.key_mse.dequantize(self.k_indices)
        return (k_hat * self.k_norms.astype(mx.float32)).astype(mx.float16)

    def _dequantize_values(self) -> mx.array:
        """Reconstruct FP16 values from indices + norms."""
        v_hat = self.val_mse.dequantize(self.v_indices)
        return (v_hat * self.v_norms.astype(mx.float32)).astype(mx.float16)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """
        Compress new K,V and return dequantized versions.
        Also stores QJL correction data for attention.

        Args:
            keys: (B, n_kv_heads, new_seq, head_dim) — after RoPE
            values: (B, n_kv_heads, new_seq, head_dim)

        Returns:
            (dequantized_keys, dequantized_values) — full sequence
        """
        k_f = keys.astype(mx.float32)
        v_f = values.astype(mx.float32)

        # Key norms + normalize
        k_norms = mx.sqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True))
        k_norms = mx.maximum(k_norms, 1e-8)
        k_unit = k_f / k_norms

        # Key MSE quantize on unit sphere
        if self.use_qjl:
            k_indices, k_residual = self.key_mse.quantize_with_residual(k_unit)
            k_signs, k_res_norms = self.key_qjl.compute_signs(k_residual)
        else:
            k_indices = self.key_mse.quantize(k_unit)

        # Value: normalize + quantize
        v_norms = mx.sqrt(mx.sum(v_f * v_f, axis=-1, keepdims=True))
        v_norms = mx.maximum(v_norms, 1e-8)
        v_unit = v_f / v_norms
        v_indices = self.val_mse.quantize(v_unit)

        # Append to compressed storage
        k_norms_stored = k_norms.astype(mx.float16)
        v_norms_stored = v_norms.astype(mx.float16)

        if self.k_indices is None:
            self.k_indices = k_indices
            self.k_norms = k_norms_stored
            self.v_indices = v_indices
            self.v_norms = v_norms_stored
            if self.use_qjl:
                k_norms_sq = k_norms.squeeze(-1).astype(mx.float16)
                self.qjl_signs = k_signs
                self.qjl_res_norms = k_res_norms
                self.qjl_key_norms = k_norms_sq
        else:
            self.k_indices = mx.concatenate([self.k_indices, k_indices], axis=2)
            self.k_norms = mx.concatenate([self.k_norms, k_norms_stored], axis=2)
            self.v_indices = mx.concatenate([self.v_indices, v_indices], axis=2)
            self.v_norms = mx.concatenate([self.v_norms, v_norms_stored], axis=2)
            if self.use_qjl:
                k_norms_sq = k_norms.squeeze(-1).astype(mx.float16)
                self.qjl_signs = mx.concatenate([self.qjl_signs, k_signs], axis=2)
                self.qjl_res_norms = mx.concatenate([self.qjl_res_norms, k_res_norms], axis=2)
                self.qjl_key_norms = mx.concatenate([self.qjl_key_norms, k_norms_sq], axis=2)

        self.offset += keys.shape[2]

        # Return dequantized for standard attention compatibility
        return self._dequantize_keys(), self._dequantize_values()

    @property
    def keys(self):
        """Dequantize keys on access (compatibility with standard attention)."""
        if self.k_indices is None:
            return None
        return self._dequantize_keys()

    @keys.setter
    def keys(self, value):
        # Allow None assignment (for compact_cache)
        if value is None:
            self.k_indices = None
            self.k_norms = None

    @property
    def values(self):
        """Dequantize values on access (compatibility with standard attention)."""
        if self.v_indices is None:
            return None
        return self._dequantize_values()

    @values.setter
    def values(self, value):
        # Allow None assignment (for compact_cache)
        if value is None:
            self.v_indices = None
            self.v_norms = None

    @property
    def nbytes(self):
        """Real memory footprint (compressed)."""
        total = 0
        if self.k_indices is not None:
            total += self.k_indices.nbytes + self.k_norms.nbytes
        if self.v_indices is not None:
            total += self.v_indices.nbytes + self.v_norms.nbytes
        if self.qjl_signs is not None:
            total += self.qjl_signs.nbytes
        if self.qjl_res_norms is not None:
            total += self.qjl_res_norms.nbytes
        if self.qjl_key_norms is not None:
            total += self.qjl_key_norms.nbytes
        return total

    def size(self):
        return self.offset

    def make_mask(self, N, return_array, window_size=None):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array=return_array, window_size=window_size)

    @property
    def state(self):
        if self.k_indices is None:
            return None, None
        return self._dequantize_keys(), self._dequantize_values()

    def is_trimmable(self):
        return False

    def empty(self):
        return self.k_indices is None
