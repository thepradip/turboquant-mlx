"""
TurboQuant Benchmark — Baseline vs MLX Quantized vs TurboQuant.

Tests one model at a time across context lengths: 1K, 2K, 4K, 8K, 16K, 32K, 64K.
Three modes per context:
  - baseline: FP16 KV cache (no compression)
  - mlx_quantized: MLX built-in QuantizedKVCache (4-bit group-wise)
  - turboquant: TurboQuant PolarQuant compression (4-bit Lloyd-Max)

Results saved via save_experiment() to results/ directory.

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --model mlx-community/Qwen3.5-4B-MLX-4bit
"""

import argparse
import gc
import os
import sys
import time
import traceback

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

# Add src to path so we can import turboquant
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from turboquant import (
    compress_cache,
    chunked_prefill,
    get_model_config,
    save_experiment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/Qwen3.5-9B-4bit",
]

CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
GEN_TOKENS = 1000
CHUNK_SIZE = 2048
BITS = 4

# Source text for context
SOURCE_TEXT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Serving_LLMs_in_Production_V2.md"
)


def measure_kv_mb(cache) -> float:
    """Compute actual KV cache memory in MB from tensor sizes.
    Handles both standard KVCache and QuantizedKVCache."""
    total_bytes = 0
    for c in cache:
        if hasattr(c, "nbytes"):
            # QuantizedKVCache has .nbytes property
            total_bytes += c.nbytes
        else:
            if hasattr(c, "keys") and c.keys is not None:
                total_bytes += c.keys.size * c.keys.dtype.size
            if hasattr(c, "values") and c.values is not None:
                total_bytes += c.values.size * c.values.dtype.size
    return total_bytes / (1024 * 1024)


def generate_tokens(model, tokenizer, logits, cache, max_tokens):
    """Generate tokens and return (tokens_list, gen_tps, ttft_ms, gen_ms)."""
    # First token — measure TTFT
    t_first = time.time()
    y = mx.argmax(logits[:, -1, :], axis=-1)
    first_logits = model(y.reshape(1, -1), cache=cache)
    mx.eval(first_logits)
    ttft_ms = (time.time() - t_first) * 1000

    tokens = [y.item()]
    logits = first_logits

    # Remaining tokens
    t_gen_start = time.time()
    for _ in range(max_tokens - 1):
        y = mx.argmax(logits[:, -1, :], axis=-1)
        tok_id = y.item()
        if tok_id == tokenizer.eos_token_id:
            break
        tokens.append(tok_id)
        logits = model(y.reshape(1, -1), cache=cache)
        mx.eval(logits)
    gen_ms = (time.time() - t_gen_start) * 1000

    total_gen_tokens = len(tokens)
    gen_tps = (total_gen_tokens - 1) / (gen_ms / 1000) if gen_ms > 0 and total_gen_tokens > 1 else 0

    return tokens, gen_tps, ttft_ms, gen_ms


def run_single_test(model, tokenizer, ids, context_len, mode, model_name):
    """Run one test (baseline, mlx_quantized, or turboquant). Returns result dict."""
    actual_tokens = len(ids)
    result = {
        "model_name": model_name,
        "context_tokens": actual_tokens,
        "target_context": context_len,
        "mode": mode,
        "bits": BITS,
        "gen_target": GEN_TOKENS,
    }

    try:
        # Always prefill with standard FP16 KVCache
        cache = make_prompt_cache(model)

        t_prefill = time.time()
        logits = chunked_prefill(model, ids, cache, chunk_size=CHUNK_SIZE)
        mx.eval(logits)
        prefill_ms = (time.time() - t_prefill) * 1000

        result["prefill_ms"] = round(prefill_ms, 1)
        result["kv_memory_before_mb"] = round(measure_kv_mb(cache), 1)

        # Post-prefill compression
        compress_result = None
        compress_ms = 0

        if mode == "turboquant":
            compress_result = compress_cache(cache, model=model, bits=BITS)
            compress_ms = compress_result.get("compress_ms", 0)
            result["kv_memory_after_mb"] = round(measure_kv_mb(cache), 1)
            # v0.5.0: measure real compressed size from indices+norms
            comp_bytes = 0
            for c in cache:
                if hasattr(c, '_tq_k_indices') and c._tq_k_indices is not None:
                    comp_bytes += c._tq_k_indices.nbytes + c._tq_k_norms.nbytes
                    comp_bytes += c._tq_v_indices.nbytes + c._tq_v_norms.nbytes
            if comp_bytes > 0:
                result["kv_compressed_mb"] = round(comp_bytes / 1024 / 1024, 1)
            result.update(compress_result)

        elif mode == "mlx_quantized":
            t_quant = time.time()
            for i, c in enumerate(cache):
                if hasattr(c, "to_quantized") and c.offset > 0:
                    cache[i] = c.to_quantized(group_size=64, bits=BITS)
            mx.eval(*[c.keys[0] for c in cache if hasattr(c, "bits")] or [mx.array(0)])
            compress_ms = (time.time() - t_quant) * 1000
            result["compress_ms"] = round(compress_ms, 0)
            result["kv_memory_after_mb"] = round(measure_kv_mb(cache), 1)

        else:
            # baseline — no compression
            result["kv_memory_after_mb"] = result["kv_memory_before_mb"]

        # Generate
        t_total_start = time.time()
        tokens, gen_tps, ttft_ms, gen_ms = generate_tokens(
            model, tokenizer, logits, cache, GEN_TOKENS
        )
        total_gen_s = (time.time() - t_total_start)

        result["gen_tokens"] = len(tokens)
        result["gen_tps"] = round(gen_tps, 1)
        result["ttft_ms"] = round(ttft_ms, 1)
        result["gen_ms"] = round(gen_ms, 1)
        result["response"] = tokenizer.decode(tokens)[:300]
        result["passed"] = True

        total_s = prefill_ms / 1000 + compress_ms / 1000 + total_gen_s
        result["total_time_s"] = round(total_s, 1)

        del cache, logits
        gc.collect()

        return result

    except Exception as e:
        result["passed"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"    ERROR: {result['error']}")
        gc.collect()
        return result


def run_model_benchmark(model_name):
    """Benchmark one model across all context lengths."""
    print(f"\n{'='*70}")
    print(f"Loading: {model_name}")
    print(f"{'='*70}")

    model, tokenizer = mlx_lm.load(model_name)
    config = get_model_config(model)
    print(f"Config: {config}")

    # Load source text and tokenize
    with open(SOURCE_TEXT_PATH) as f:
        source_text = f.read()

    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Analyze and summarize the following document in detail:\n\n{source_text}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    all_ids = mx.array(tokenizer.encode(chat_text))
    max_available = len(all_ids)
    print(f"Source tokens available: {max_available}")

    for ctx_len in CONTEXT_LENGTHS:
        if ctx_len > max_available:
            print(f"\n--- {ctx_len} tokens: SKIP (only {max_available} available) ---")
            continue

        ids = all_ids[:ctx_len]
        actual = len(ids)

        print(f"\n--- {ctx_len} tokens (actual: {actual}) ---")

        for mode in ["baseline", "mlx_quantized", "turboquant"]:
            print(f"  [{mode}] running...", end=" ", flush=True)
            t0 = time.time()

            result = run_single_test(model, tokenizer, ids, ctx_len, mode, model_name)

            elapsed = time.time() - t0

            if result["passed"]:
                kv_before = result.get("kv_memory_before_mb", 0)
                kv_after = result.get("kv_memory_after_mb", 0)
                saved = kv_before - kv_after
                extra = ""
                if mode == "turboquant":
                    extra = f" cos={result.get('cosine', '?')}"
                if mode != "baseline":
                    extra += f" saved={saved:.1f}MB"
                    extra += f" comp={result.get('compress_ms', 0):.0f}ms"

                print(f"OK ({elapsed:.1f}s) "
                      f"prefill={result['prefill_ms']:.0f}ms "
                      f"ttft={result['ttft_ms']:.0f}ms "
                      f"gen={result['gen_tps']:.1f}tps "
                      f"kv={kv_after:.1f}MB"
                      f"{extra}")
            else:
                print(f"FAIL ({elapsed:.1f}s) {result.get('error', '')[:80]}")

            # Save result
            filepath = save_experiment(
                model_name=model_name,
                compress_result={
                    k: result[k] for k in [
                        "cosine", "compress_ms", "layers_compressed",
                        "original_mb", "compressed_mb", "saved_mb", "ratio",
                    ] if k in result
                } or None,
                model=model,
                context_tokens=actual,
                gen_tokens=result.get("gen_tokens", 0),
                gen_tps=result.get("gen_tps", 0),
                ttft_ms=result.get("ttft_ms", 0),
                passed=result.get("passed", False),
                response=result.get("response", ""),
                notes=f"benchmark {mode}",
                bits=BITS,
                mode=mode,
                prefill_ms=result.get("prefill_ms", 0),
                gen_ms=result.get("gen_ms", 0),
                kv_memory_before_mb=result.get("kv_memory_before_mb", 0),
                kv_memory_after_mb=result.get("kv_memory_after_mb", 0),
                total_time_s=result.get("total_time_s", 0),
                target_context=ctx_len,
                error=result.get("error", ""),
            )
            print(f"    saved: {os.path.basename(filepath)}")

    # Unload model
    del model, tokenizer
    gc.collect()
    print(f"\nUnloaded {model_name}")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Run single model (e.g. mlx-community/Qwen3.5-4B-MLX-4bit)")
    args = parser.parse_args()

    models = [args.model] if args.model else ALL_MODELS

    print("TurboQuant Benchmark")
    print(f"Models: {models}")
    print(f"Contexts: {CONTEXT_LENGTHS}")
    print(f"Gen tokens: {GEN_TOKENS}")
    print(f"Modes: baseline, mlx_quantized, turboquant")
    print(f"Bits: {BITS}")

    if not os.path.exists(SOURCE_TEXT_PATH):
        print(f"ERROR: Source text not found at {SOURCE_TEXT_PATH}")
        sys.exit(1)

    for model_name in models:
        try:
            run_model_benchmark(model_name)
        except Exception as e:
            print(f"\nFATAL ERROR on {model_name}: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nResults saved in results/ directory.")
    print("Use list_experiments() to review.")


if __name__ == "__main__":
    main()
