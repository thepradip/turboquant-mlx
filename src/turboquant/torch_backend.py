"""
TurboQuant PyTorch backend — works on CPU, CUDA (T4/A100/H100), MPS.

Same algorithm as MLX version, using PyTorch ops.
Accesses KV cache through HuggingFace transformers DynamicCache.
"""

import time
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from .codebook import build_codebook, build_rotation


class PolarQuantTorch:
    """PolarQuant compressor using PyTorch tensors."""

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = torch.device(device)

        centroids_np = build_codebook(bits, head_dim)
        rotation_np = build_rotation(head_dim, seed)

        self.centroids = torch.tensor(centroids_np, dtype=torch.float32, device=self.device)
        self.rotation = torch.tensor(rotation_np, dtype=torch.float32, device=self.device)
        self.rotation_t = self.rotation.T.contiguous()

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = x @ self.rotation_t
        dists = (y.unsqueeze(-1) - self.centroids).abs()
        return dists.argmin(dim=-1).to(torch.uint8)

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        y_hat = self.centroids[indices.long()]
        return y_hat @ self.rotation


def compress_cache_torch(
    kv_cache: Any,
    model: Any = None,
    bits: int = 4,
    window_size: int = 0,
    min_context: int = 0,
    device: str = "cpu",
) -> Dict:
    """
    Compress HuggingFace DynamicCache in-place using PyTorch.

    Works on CPU, CUDA, MPS. Same approach as MLX version:
    prefill first (FP16), then compress.

    Args:
        kv_cache: HuggingFace past_key_values (DynamicCache or tuple).
        model: HuggingFace model (for auto-detecting head_dim).
        bits: Quantization bits (2, 3, or 4).
        window_size: Keep last N tokens uncompressed. 0 = compress all.
        min_context: Skip if context shorter than this.
        device: torch device string.

    Returns:
        Dict with cosine, compress_ms, layers_compressed, memory stats.
    """
    # Auto-detect head_dim
    head_dim = None
    if model is not None:
        config = model.config
        head_dim = config.hidden_size // config.num_attention_heads

    t0 = time.time()
    cos_scores = []
    total_orig = 0
    total_comp = 0

    # Handle DynamicCache (transformers >= 4.36)
    if hasattr(kv_cache, 'layers'):
        num_layers = len(kv_cache.layers)
        for li in range(num_layers):
            layer = kv_cache.layers[li]
            keys = layer.keys if hasattr(layer, 'keys') else None
            values = layer.values if hasattr(layer, 'values') else None
            if keys is None or values is None:
                continue

            seq = keys.shape[2]
            if seq <= min_context:
                continue

            compress_end = seq - window_size if window_size > 0 else seq
            if compress_end <= 0:
                continue

            kd = keys.shape[-1]
            if head_dim is None:
                head_dim = kd
            if kd < 2:
                continue

            k = keys[:, :, :compress_end, :]
            v = values[:, :, :compress_end, :]

            # Compress keys
            k_mse = PolarQuantTorch(kd, bits=bits, seed=42 + li, device=str(k.device))
            k_f = k.float()
            k_norms = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k_unit = k_f / k_norms
            k_hat_unit = k_mse.dequantize(k_mse.quantize(k_unit))
            k_hat = (k_hat_unit * k_norms).to(k.dtype)

            # Compress values
            v_mse = PolarQuantTorch(kd, bits=bits, seed=1000 + li, device=str(v.device))
            v_f = v.float()
            v_norms = v_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_unit = v_f / v_norms
            v_hat_unit = v_mse.dequantize(v_mse.quantize(v_unit))
            v_hat = (v_hat_unit * v_norms).to(v.dtype)

            # Cosine
            flat_k = k_f.reshape(-1, kd)
            flat_r = k_hat.float().reshape(-1, kd)
            cos = torch.nn.functional.cosine_similarity(flat_k, flat_r, dim=-1).mean().item()
            cos_scores.append(cos)

            # Memory
            n = k.numel()
            total_orig += n * 2 * 2
            total_comp += (n * bits / 8 + k.shape[2] * k.shape[1] * 2) * 2

            # Write back
            keys[:, :, :compress_end, :] = k_hat
            values[:, :, :compress_end, :] = v_hat

    # Handle tuple format (older transformers)
    elif isinstance(kv_cache, (list, tuple)):
        for li, layer_kv in enumerate(kv_cache):
            if isinstance(layer_kv, (list, tuple)) and len(layer_kv) == 2:
                keys, values = layer_kv
            else:
                continue

            seq = keys.shape[2]
            if seq <= min_context:
                continue

            compress_end = seq - window_size if window_size > 0 else seq
            if compress_end <= 0:
                continue

            kd = keys.shape[-1]
            if head_dim is None:
                head_dim = kd
            if kd < 2:
                continue

            k = keys[:, :, :compress_end, :]
            v = values[:, :, :compress_end, :]

            k_mse = PolarQuantTorch(kd, bits=bits, seed=42 + li, device=str(k.device))
            k_f = k.float()
            k_norms = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k_unit = k_f / k_norms
            k_hat = (k_mse.dequantize(k_mse.quantize(k_unit)) * k_norms).to(k.dtype)

            v_mse = PolarQuantTorch(kd, bits=bits, seed=1000 + li, device=str(v.device))
            v_f = v.float()
            v_norms = v_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_unit = v_f / v_norms
            v_hat = (v_mse.dequantize(v_mse.quantize(v_unit)) * v_norms).to(v.dtype)

            flat_k = k_f.reshape(-1, kd)
            flat_r = k_hat.float().reshape(-1, kd)
            cos = torch.nn.functional.cosine_similarity(flat_k, flat_r, dim=-1).mean().item()
            cos_scores.append(cos)

            n = k.numel()
            total_orig += n * 2 * 2
            total_comp += (n * bits / 8 + k.shape[2] * k.shape[1] * 2) * 2

            keys[:, :, :compress_end, :] = k_hat
            values[:, :, :compress_end, :] = v_hat

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
