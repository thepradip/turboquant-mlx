"""
Custom 1-bit model loader for Bonsai-8B on MLX.

MLX's mx.quantize() doesn't support bits=1. This module provides:
- Bonsai1BitLinear: custom layer that dequantizes 1-bit packed uint32 weights on the fly
- load_bonsai_1bit(): drop-in replacement for mlx_lm.load() that handles 1-bit models

Weight format (from safetensors):
- weight: (out_features, in_features/32) uint32 — 32 binary values packed per uint32
- scales: (out_features, num_groups) float16
- biases: (out_features, num_groups) float16
- group_size=128, affine: value = bit * scale + bias

Usage:
    from turboquant.bonsai_loader import load_bonsai_1bit
    model, tokenizer = load_bonsai_1bit("path/to/Bonsai-8B-mlx")

Author: Pradip Tivhale, April 2026
"""

import math
import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


# ============================================================================
# Metal kernel for 1-bit affine matmul (avoids full dequantization)
#
# Computes y = x @ dequant(W).T where W is 1-bit packed.
# Decomposition per group: y += scale * bit_dot + bias * x_sum
#   bit_dot = sum of x[j] where bit[j]=1
#   x_sum   = sum of all x[j] in group
# This avoids materializing the full float weight matrix.
# ============================================================================

def _make_bonsai_1bit_kernel():
    source = """
    // Each thread computes one (batch, output) element.
    // grid: (out_features, batch_size, 1)
    uint out_idx = thread_position_in_grid.x;
    uint batch_idx = thread_position_in_grid.y;

    if (out_idx >= out_features || batch_idx >= batch_size)
        return;

    float acc = 0.0f;

    uint packed_per_group = group_size / 32;  // uint32s per group

    for (uint g = 0; g < num_groups; g++) {
        float s = (float)scales[out_idx * num_groups + g];
        float b = (float)biases_q[out_idx * num_groups + g];

        float bit_dot = 0.0f;
        float x_sum = 0.0f;

        for (uint p = 0; p < packed_per_group; p++) {
            uint packed = packed_w[out_idx * (in_features / 32) + g * packed_per_group + p];

            for (uint bit = 0; bit < 32; bit++) {
                uint feat_idx = g * group_size + p * 32 + bit;
                float xv = (float)x[batch_idx * in_features + feat_idx];
                x_sum += xv;
                if ((packed >> bit) & 1u) {
                    bit_dot += xv;
                }
            }
        }

        acc += s * bit_dot + b * x_sum;
    }

    out[batch_idx * out_features + out_idx] = static_cast<T>(acc);
    """

    return mx.fast.metal_kernel(
        name="bonsai_1bit_matmul",
        input_names=["x", "packed_w", "scales", "biases_q"],
        output_names=["out"],
        source=source,
    )


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _make_bonsai_1bit_kernel()
    return _kernel


# ============================================================================
# Bonsai1BitLinear layer
# ============================================================================

class Bonsai1BitLinear(nn.Module):
    """Linear layer with 1-bit quantized weights (affine: val = bit * scale + bias).

    Stores weights as packed uint32 (32 bits per element) with per-group
    float16 scales and biases. Forward pass uses a custom Metal kernel
    that computes matmul directly on packed weights without full dequantization.
    """

    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.packed_dim = in_features // 32  # 32 bits per uint32

        # Quantized weight storage
        self.weight = mx.zeros((out_features, self.packed_dim), dtype=mx.uint32)
        self.scales = mx.zeros((out_features, self.num_groups), dtype=mx.float16)
        self.biases = mx.zeros((out_features, self.num_groups), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((out_features,))

        self.freeze()

    def _dequantize_mlx(self):
        """Pure MLX ops dequantization (fallback)."""
        shifts = mx.arange(32, dtype=mx.uint32)
        # (out, packed_dim, 32)
        bits = ((self.weight[:, :, None] >> shifts) & mx.array(1, dtype=mx.uint32))
        bits = bits.reshape(self.out_features, self.num_groups, self.group_size)
        bits = bits.astype(self.scales.dtype)
        # Affine: val = bit * scale + bias
        w = bits * self.scales[:, :, None] + self.biases[:, :, None]
        return w.reshape(self.out_features, self.in_features)

    def __call__(self, x):
        original_shape = x.shape
        dtype = x.dtype

        # Flatten batch dims
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        batch_size = x.shape[0]

        try:
            kernel = _get_kernel()
            y = kernel(
                inputs=[x, self.weight, self.scales, self.biases],
                template=[
                    ("T", dtype),
                    ("in_features", self.in_features),
                    ("out_features", self.out_features),
                    ("group_size", self.group_size),
                    ("num_groups", self.num_groups),
                    ("batch_size", batch_size),
                ],
                grid=(self.out_features, batch_size, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[(batch_size, self.out_features)],
                output_dtypes=[dtype],
            )[0]
        except Exception:
            # Fallback: pure MLX ops
            w = self._dequantize_mlx().astype(dtype)
            y = x @ w.T

        if len(original_shape) > 2:
            y = y.reshape(*original_shape[:-1], self.out_features)

        if "bias" in self:
            y = y + self["bias"]
        return y


# ============================================================================
# Bonsai1BitEmbedding layer
# ============================================================================

class Bonsai1BitEmbedding(nn.Module):
    """Embedding layer with 1-bit quantized weights.

    Same packed format as Bonsai1BitLinear but dequantizes selected rows
    (indexed lookup) instead of full matmul.
    """

    def __init__(self, num_embeddings, dims, group_size=128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.num_groups = dims // group_size
        self.packed_dim = dims // 32

        self.weight = mx.zeros((num_embeddings, self.packed_dim), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, self.num_groups), dtype=mx.float16)
        self.biases = mx.zeros((num_embeddings, self.num_groups), dtype=mx.float16)

        self.freeze()

    def __call__(self, x):
        # Index into packed weights, scales, biases
        w = self.weight[x]       # (..., packed_dim) uint32
        s = self.scales[x]       # (..., num_groups) float16
        b = self.biases[x]       # (..., num_groups) float16

        # Unpack bits: (..., packed_dim, 32)
        shifts = mx.arange(32, dtype=mx.uint32)
        bits = ((w[..., :, None] >> shifts) & mx.array(1, dtype=mx.uint32))

        # Flatten to (..., dims) then group to (..., num_groups, group_size)
        orig_shape = x.shape  # batch dims
        bits = bits.reshape(*orig_shape, self.dims)
        bits = bits.reshape(*orig_shape, self.num_groups, self.group_size)
        bits = bits.astype(s.dtype)

        # Affine dequantize: val = bit * scale + bias
        # s, b: (..., num_groups) -> (..., num_groups, 1)
        dequant = bits * s[..., :, None] + b[..., :, None]

        # Reshape to (..., dims)
        return dequant.reshape(*orig_shape, self.dims)

    def as_linear(self, x):
        """Use embedding as output projection (for tied weights)."""
        w = self._dequantize_all().astype(x.dtype)
        return x @ w.T

    def _dequantize_all(self):
        """Full dequantization of all embeddings."""
        shifts = mx.arange(32, dtype=mx.uint32)
        bits = ((self.weight[:, :, None] >> shifts) & mx.array(1, dtype=mx.uint32))
        bits = bits.reshape(self.num_embeddings, self.num_groups, self.group_size)
        bits = bits.astype(self.scales.dtype)
        w = bits * self.scales[:, :, None] + self.biases[:, :, None]
        return w.reshape(self.num_embeddings, self.dims)


# ============================================================================
# bonsai_1bit_quantize: replace nn.Linear -> Bonsai1BitLinear
# ============================================================================

def bonsai_1bit_quantize(model, group_size=128, weights=None):
    """Replace nn.Linear and nn.Embedding with 1-bit versions where quantized weights exist."""
    quantize_layers = []

    for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(module, nn.Linear):
            if weights is not None and f"{name}.scales" not in weights:
                continue
            out_features, in_features = module.weight.shape
            has_bias = "bias" in module
            new_layer = Bonsai1BitLinear(
                in_features, out_features, bias=has_bias, group_size=group_size,
            )
            quantize_layers.append((name, new_layer))

        elif isinstance(module, nn.Embedding):
            if weights is not None and f"{name}.scales" not in weights:
                continue
            num_embeddings, dims = module.weight.shape
            # dims is the real embedding dim (hidden_size, e.g. 4096)
            # packed weights in safetensors will be (num_embeddings, dims/32)
            new_layer = Bonsai1BitEmbedding(
                num_embeddings, dims, group_size=group_size,
            )
            quantize_layers.append((name, new_layer))

    if quantize_layers:
        model.update_modules(tree_unflatten(quantize_layers))
    return model


# ============================================================================
# load_bonsai_1bit: drop-in replacement for mlx_lm.load()
# ============================================================================

def load_bonsai_1bit(path_or_hf_repo, lazy=False, tokenizer_config=None):
    """Load a Bonsai 1-bit model on MLX.

    Drop-in replacement for mlx_lm.load() that handles bits=1 which
    MLX doesn't natively support.

    Args:
        path_or_hf_repo: Path to model directory (must have config.json + safetensors)
        lazy: If True, don't eval parameters immediately
        tokenizer_config: Optional tokenizer config overrides

    Returns:
        (model, tokenizer) — same as mlx_lm.load()
    """
    from mlx_lm.utils import load_config, load_tokenizer, _get_classes, _download

    model_path = Path(_download(path_or_hf_repo))
    config = load_config(model_path)

    # Extract quantization info before removing it
    quant_config = config.get("quantization", {})
    group_size = quant_config.get("group_size", 128)
    bits = quant_config.get("bits", 1)

    if bits != 1:
        # Not a 1-bit model, use standard loader
        from mlx_lm import load
        return load(str(model_path), tokenizer_config=tokenizer_config, lazy=lazy)

    # Load weights from safetensors
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Remove quantization from config so model creates nn.Linear layers
    config_no_quant = {k: v for k, v in config.items() if k != "quantization"}

    # Create model architecture
    model_class, model_args_class = _get_classes(config=config_no_quant)
    model_args = model_args_class.from_dict(config_no_quant)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Replace nn.Linear with Bonsai1BitLinear
    model = bonsai_1bit_quantize(model, group_size=group_size, weights=weights)
    model.eval()

    # Load weights with key audit
    model_keys = set(k for k, _ in model.parameters())
    weight_keys = set(weights.keys())
    missing = model_keys - weight_keys
    unexpected = weight_keys - model_keys
    if missing:
        print(f"  WARNING: {len(missing)} missing keys in checkpoint: {list(missing)[:5]}...")
    if unexpected:
        print(f"  INFO: {len(unexpected)} extra keys in checkpoint (ignored)")
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    # Load tokenizer
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )

    return model, tokenizer
