"""TurboQuant — KV-cache compression for LLM inference on Apple Silicon."""
__version__ = "0.4.0"

from .patch import (
    compress_cache,
    chunked_prefill,
    patch_model,
    make_turboquant_cache,
    get_head_dim,
    get_num_layers,
    get_model_config,
)
from .compressor import PolarQuantMLX
from .qjl import QJLMLX
from .cache import TurboQuantCache
from .attention import turboquant_sdpa
