#!/usr/bin/env python3
"""
TurboQuant Unified Evaluation Suite v1.1

Single script for standardized quality + needle-in-haystack testing
across fp16, tq_4bit, and tq_3bit KV cache configurations.

Metrics: accuracy, throughput, latency, TTFT, memory, compression quality.
Optional: LLM-as-judge scoring via Claude API (--judge flag).

Run:
  python3 benchmarks/tq_eval.py --model mlx-community/Qwen3.5-4B-MLX-4bit
  python3 benchmarks/tq_eval.py --model mlx-community/gemma-4-E4B-it-4bit --skip-needle
  python3 benchmarks/tq_eval.py --model ... --configs fp16 tq_4bit tq_3bit --judge --report eval.html
"""

import sys, os, json, time, argparse, gc, re
from datetime import datetime
from collections import defaultdict
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from turboquant import compress_cache, chunked_prefill, get_model_config

ROOT = os.path.dirname(__file__)
QUESTIONS_FILE = os.path.join(ROOT, "tq_eval_65_questions.json")
KV_MAP = {"fp16": 0, "tq_4bit": 4, "tq_3bit": 3}

NEEDLES = [
    {"id": "needle_alpha",
     "needle": "SECRET-ALPHA: The classified research budget was exactly $847.3 million, allocated across 14 laboratories under project codename NIGHTINGALE, approved by Director Sarah Chen on March 17, 2025.",
     "question": "What was the classified research budget, the project codename, and who approved it?",
     "check_numbers": ["847.3"], "check_words": ["nightingale", "sarah"]},
    {"id": "needle_beta",
     "needle": "SECRET-BETA: Sensor grid KR-9047 detected anomalous readings of 2,341 microsieverts at coordinates 52.3676N 4.9041E, triggering evacuation protocol OMEGA-7 for sectors 12 through 19 at exactly 03:47:22 UTC.",
     "question": "What reading did sensor KR-9047 detect, what protocol was triggered, and for which sectors?",
     "check_numbers": ["2341", "12", "19"], "check_words": ["omega"]},
    {"id": "needle_gamma",
     "needle": "SECRET-GAMMA: The archaeological team carbon-dated Artifact ZX-8891 to 4,720 years BP at depth 7.3 meters, with copper-tin ratio 91:9 matching Minoan production, contradicting the previous Anatolian attribution by Professor James Whitfield.",
     "question": "How old was Artifact ZX-8891, at what depth was it found, and whose attribution did it contradict?",
     "check_numbers": ["4720", "7.3"], "check_words": ["whitfield"]},
]


# ════════════════════════════════════════════════════════════
#  Reused functions
# ════════════════════════════════════════════════════════════

def format_prompt(tokenizer, raw_prompt, enable_thinking=False):
    """Apply chat template for correct model-specific formatting."""
    messages = [{"role": "user", "content": raw_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return raw_prompt
    except Exception:
        return raw_prompt


def _get_stop_tokens(tokenizer):
    """Get all stop token IDs for this model."""
    stop = {tokenizer.eos_token_id}
    for text in ["<turn|>", "<|endoftext|>", "<|im_end|>"]:
        try:
            ids = tokenizer.encode(text)
            if len(ids) == 1:
                stop.add(ids[0])
        except Exception:
            pass
    return stop


def check_answer(answer_text, check_numbers, check_words):
    """Check if expected values are present in the answer."""
    text = answer_text.lower()
    details = {"numbers": {}, "words": {}}
    passed = True
    for n in check_numbers:
        found = str(n).lower() in text
        details["numbers"][n] = found
        if not found:
            passed = False
    for w in check_words:
        found = w.lower() in text
        details["words"][w] = found
        if not found:
            passed = False
    return passed, details


def load_env(env_path=None):
    """Load .env file into os.environ. Returns dict of loaded keys."""
    if env_path is None:
        env_path = os.path.join(ROOT, "..", ".env")
    if not os.path.isfile(env_path):
        return {}
    loaded = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if val and val != "<your-api-key>":
                    os.environ.setdefault(key, val)
                    loaded[key] = val
    return loaded


def init_judge_client():
    """Initialize OpenAI client from .env config. Returns (client, model_name) or (None, None)."""
    env = load_env()
    endpoint = os.environ.get("JUDGE_ENDPOINT", "")
    api_key = os.environ.get("JUDGE_API_KEY", "")
    model = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")

    if not api_key or api_key == "<your-api-key>":
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(base_url=endpoint, api_key=api_key) if endpoint else OpenAI(api_key=api_key)
        return client, model
    except ImportError:
        print("  WARNING: openai package not installed. Run: pip install openai")
        return None, None
    except Exception as e:
        print(f"  WARNING: Failed to init judge client: {e}")
        return None, None


def llm_judge(question, reference_answer, model_answer, category, client, judge_model):
    """Use LLM to score model_answer against reference_answer. Returns dict with score 0-10 and reasoning."""
    if not model_answer.strip():
        return {"score": 0, "reasoning": "Empty answer", "verdict": "FAIL"}

    system = (
        "You are a strict evaluator. Your job is to determine if the model's answer is CORRECT "
        "by comparing it against the reference answer.\n\n"
        "DO NOT just check if keywords or numbers appear in the text. You must verify:\n"
        "1. Does the model reach the SAME final answer/conclusion as the reference?\n"
        "2. Is the reasoning/logic correct, not just the surface text?\n"
        "3. Are ALL parts of the question answered, not just some?\n"
        "4. For multi-part questions: check EACH part independently.\n\n"
        "Scoring:\n"
        "- 10: Correct final answer with valid reasoning for all parts\n"
        "- 7-9: Correct final answer but minor gaps (missing units, one sub-part weak)\n"
        "- 4-6: Partially correct — some parts right, some wrong or missing\n"
        "- 1-3: Wrong final answer even if some intermediate steps are right\n"
        "- 0: Completely wrong, irrelevant, or empty\n\n"
        "Category-specific rules:\n"
        "- MATH/FINANCE: The final numerical answer MUST match. Right steps but wrong number = max 3.\n"
        "- CODING: Code must be functionally correct (right algorithm, handles edge cases). "
        "Style differences are OK. Wrong algorithm or logic bugs = max 4.\n"
        "- REASONING: The conclusion must match. Getting there via different logic is fine if correct.\n"
        "- LONG_CONTEXT: All requested facts (names, numbers, dates) must be present and correct.\n"
        "- INSTRUCTION: Must follow ALL formatting constraints (order, count, structure).\n\n"
        "IMPORTANT: A number appearing in the answer does NOT mean it's the answer to the question. "
        "The model might mention '160' in a calculation step but give a DIFFERENT final answer. "
        "Check what the model actually concludes, not what numbers appear.\n\n"
        "Respond in EXACTLY this JSON format, nothing else:\n"
        '{"score": <0-10>, "reasoning": "<2-3 sentences explaining what matched and what didn\'t>", '
        '"verdict": "<PASS|PARTIAL|FAIL>"}'
    )

    prompt_text = (
        f"Category: {category}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer (ground truth):\n{reference_answer}\n\n"
        f"Model's Answer (to evaluate):\n{model_answer[:3000]}\n\n"
        "Compare the model's answer against the reference. "
        "Check if the model arrives at the same correct answer, not just whether it contains similar words. "
        "JSON response only."
    )

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                max_completion_tokens=250,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_text},
                ],
            )
            text = response.choices[0].message.content.strip()

            # Parse JSON — try full match first, then relaxed
            try:
                # Try direct parse (handles nested quotes, escapes)
                result = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract JSON object with regex, allow nested content
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    raw = json_match.group()
                    # Clean common LLM issues: smart quotes, trailing commas
                    raw = raw.replace('\u201c', '"').replace('\u201d', '"').replace("'", "'")
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        # Last resort: extract fields manually
                        score_m = re.search(r'"score"\s*:\s*(\d+)', raw)
                        verdict_m = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
                        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
                        if score_m:
                            result = {
                                "score": int(score_m.group(1)),
                                "verdict": verdict_m.group(1) if verdict_m else "UNKNOWN",
                                "reasoning": reason_m.group(1) if reason_m else text[:100],
                            }
                        else:
                            if attempt < max_retries - 1:
                                last_error = f"JSON parse failed: {text[:80]}"
                                continue
                            return {"score": 0, "reasoning": f"Could not parse after {max_retries} attempts: {text[:80]}", "verdict": "FAIL"}
                else:
                    if attempt < max_retries - 1:
                        last_error = f"No JSON in response: {text[:80]}"
                        continue
                    return {"score": 0, "reasoning": f"No JSON found after {max_retries} attempts: {text[:80]}", "verdict": "FAIL"}

            result["score"] = max(0, min(10, int(result.get("score", 0))))
            if "verdict" not in result:
                result["verdict"] = "PASS" if result["score"] >= 7 else "PARTIAL" if result["score"] >= 4 else "FAIL"
            return result

        except Exception as e:
            last_error = str(e)[:100]
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
            return {"score": -1, "reasoning": f"Judge failed after {max_retries} attempts: {last_error}", "verdict": "ERROR"}


def generate_answer(model, tokenizer, prompt, kv_mode, bits, max_tokens=500):
    """Generate using production path: compact=True + generate_step() for TQ modes.
    FP16 uses standard generation (no compression)."""
    from turboquant import generate_step

    stop_tokens = _get_stop_tokens(tokenizer)
    ids = mx.array(tokenizer.encode(prompt))
    n_prompt = len(ids)

    mx.eval(mx.array([0]))
    metal_before = mx.get_active_memory()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model)

    t_wall = time.time()
    t0 = time.time()
    if n_prompt > 2048:
        logits = chunked_prefill(model, ids, cache, chunk_size=2048)
    else:
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
    prefill_ms = (time.time() - t0) * 1000
    prefill_tps = n_prompt / (prefill_ms / 1000) if prefill_ms > 0 else 0

    mx.eval(mx.array([0]))
    kv_fp16_bytes = 0
    for c in cache:
        if hasattr(c, 'keys') and c.keys is not None:
            used = getattr(c, 'offset', c.keys.shape[2])
            if used > 0:
                kv_fp16_bytes += c.keys[:, :, :used, :].nbytes + c.values[:, :, :used, :].nbytes

    comp_info = None
    compress_ms = 0
    use_generate_step = False
    if kv_mode != "fp16":
        t1 = time.time()
        comp_info = compress_cache(cache, model=model, bits=bits, compact=True)
        compress_ms = (time.time() - t1) * 1000
        use_generate_step = True

    ttft_ms = prefill_ms + compress_ms
    mx.eval(mx.array([0]))
    metal_after = mx.get_active_memory()

    t_gen = time.time()
    tokens, token_times = [], []
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    for _ in range(max_tokens):
        tok_id = y.item()
        if tok_id in stop_tokens:
            break
        t_tok = time.time()
        tokens.append(tok_id)
        if use_generate_step:
            logits = generate_step(model, y[:, None], cache)
        else:
            logits = model(y[:, None], cache=cache)
            mx.eval(logits)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        token_times.append((time.time() - t_tok) * 1000)

    gen_ms = (time.time() - t_gen) * 1000
    wall_ms = (time.time() - t_wall) * 1000
    n_gen = len(tokens)
    gen_tps = n_gen / (gen_ms / 1000) if gen_ms > 0 else 0
    mx.eval(mx.array([0]))
    metal_peak = mx.get_peak_memory()

    avg_token_ms = mean(token_times) if token_times else 0
    p50_token_ms = sorted(token_times)[len(token_times) // 2] if token_times else 0
    p99_token_ms = sorted(token_times)[int(len(token_times) * 0.99)] if token_times else 0

    comp_bytes = 0
    for c in cache:
        for attr in ('_tq_k_indices', '_tq_k_norms', '_tq_v_indices', '_tq_v_norms'):
            t = getattr(c, attr, None)
            if t is not None:
                comp_bytes += t.nbytes

    metrics = {
        "prompt_tokens": n_prompt, "gen_tokens": n_gen,
        "ttft_ms": round(ttft_ms, 1), "prefill_ms": round(prefill_ms, 1),
        "prefill_tps": round(prefill_tps, 1), "compress_ms": round(compress_ms, 1),
        "gen_ms": round(gen_ms, 1), "gen_tps": round(gen_tps, 1),
        "wall_ms": round(wall_ms, 1),
        "avg_token_ms": round(avg_token_ms, 2),
        "p50_token_ms": round(p50_token_ms, 2),
        "p99_token_ms": round(p99_token_ms, 2),
        "kv_fp16_mb": round(kv_fp16_bytes / 1024**2, 4),
        "kv_compressed_mb": round(comp_bytes / 1024**2, 4),
        "metal_before_mb": round(metal_before / 1024**2, 1),
        "metal_after_mb": round(metal_after / 1024**2, 1),
        "metal_peak_mb": round(metal_peak / 1024**2, 1),
        "cosine": comp_info.get("cosine") if comp_info else None,
        "ratio": comp_info.get("ratio") if comp_info else None,
    }

    del cache, logits, y, ids
    gc.collect()
    mx.metal.clear_cache()
    return tokenizer.decode(tokens), metrics


def fetch_filler(target_chars, offset=0):
    """Fetch real text for needle-in-haystack — arxiv for medium, pg19 for long."""
    from datasets import load_dataset
    if target_chars < 60000:
        ds = load_dataset("ccdv/arxiv-summarization", split="test", streaming=True)
        parts, total, skipped = [], 0, 0
        for s in ds:
            if skipped < offset:
                skipped += 1
                continue
            parts.append(s["article"])
            total += len(s["article"])
            if total >= target_chars:
                break
        return "\n\n".join(parts)[:target_chars]
    else:
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        skipped = 0
        for s in ds:
            if skipped < offset:
                skipped += 1
                continue
            if len(s["text"]) >= target_chars:
                start = len(s["text"]) // 4
                return s["text"][start:start + target_chars]
        parts, total = [], 0
        ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        for s in ds:
            parts.append(s["text"])
            total += len(s["text"])
            if total >= target_chars:
                break
        return "\n\n".join(parts)[:target_chars]


def build_needle_prompt(filler, needle_spec):
    """Insert needle into filler text at ~60%, return formatted prompt."""
    pos = int(len(filler) * 0.6)
    nl = filler.rfind("\n", max(0, pos - 500), pos + 500)
    if nl > 0:
        pos = nl
    full_doc = filler[:pos] + "\n\n" + needle_spec["needle"] + "\n\n" + filler[pos:]
    return (
        "Read the document carefully. Answer the question using ONLY facts from "
        "the document. Include exact numbers and names.\n\n"
        f"--- DOCUMENT ---\n{full_doc}\n--- END ---\n\n"
        f"Question: {needle_spec['question']}\nAnswer:"
    )


# ════════════════════════════════════════════════════════════
#  Test Suites
# ════════════════════════════════════════════════════════════

def run_quality_suite(model, tokenizer, questions, configs, max_tokens, output_path, results, judge_client=None, judge_model=None):
    """Run 65 quality questions across all configs with resume support and optional LLM judge."""
    done = {cfg: {e["id"] for e in results["answers"].get(cfg, [])} for cfg in configs}

    for kv_name in configs:
        bits = KV_MAP.get(kv_name, 4)
        if kv_name not in results["answers"]:
            results["answers"][kv_name] = []
        remaining = [q for q in questions if q["id"] not in done.get(kv_name, set())]
        if not remaining:
            print(f"  {kv_name}: all {len(questions)} questions already done, skipping.")
            continue

        total_done = len(questions) - len(remaining)
        print(f"\n{'='*60}\n  QUALITY: {kv_name} (bits={bits}) — {len(remaining)} remaining\n{'='*60}")

        for i, q in enumerate(remaining):
            idx = total_done + i + 1
            print(f"  [{idx}/{len(questions)}] {q['category']:>12} | {q['id']}", end=" ... ", flush=True)

            formatted = format_prompt(tokenizer, q["prompt"])
            answer, metrics = generate_answer(model, tokenizer, formatted, kv_name, bits, max_tokens)
            keyword_passed, keyword_details = check_answer(answer, q.get("check_numbers", []), q.get("check_words", []))

            # LLM-as-judge scoring — this is the primary accuracy measure
            judge_result = None
            if judge_client and q.get("reference_answer"):
                judge_result = llm_judge(q["question"], q["reference_answer"], answer, q["category"], judge_client, judge_model)

            # Determine final pass/fail: judge verdict is primary when available
            if judge_result and judge_result.get("score", -1) >= 0:
                passed = judge_result["verdict"] == "PASS"
            else:
                passed = keyword_passed

            status = "PASS" if passed else "FAIL"
            judge_str = ""
            if judge_result and judge_result.get("score", -1) >= 0:
                judge_str = f" | judge={judge_result['score']}/10 {judge_result['verdict']}"
            print(f"{status} | {metrics['gen_tokens']} tok | "
                  f"TTFT={metrics['ttft_ms']:.0f}ms | gen={metrics['gen_tps']:.1f}t/s | "
                  f"wall={metrics['wall_ms']/1000:.1f}s | metal={metrics['metal_peak_mb']:.0f}MB{judge_str}", end="")
            if metrics["cosine"] is not None:
                print(f" | cos={metrics['cosine']} {metrics['ratio']}x", end="")
            print()

            entry = {
                "id": q["id"], "category": q["category"],
                "question": q["question"], "answer": answer,
                "reference_answer": q.get("reference_answer", ""),
                "metrics": metrics,
                "check_numbers": q.get("check_numbers", []),
                "check_words": q.get("check_words", []),
                "passed": passed,
                "keyword_passed": keyword_passed, "keyword_details": keyword_details,
            }
            if judge_result:
                entry["judge"] = judge_result

            results["answers"][kv_name].append(entry)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            gc.collect()
            mx.metal.clear_cache()

        p = sum(1 for e in results["answers"][kv_name] if e.get("passed"))
        j_scores = [e["judge"]["score"] for e in results["answers"][kv_name] if e.get("judge") and e["judge"].get("score", -1) >= 0]
        judge_str = f" | judge avg={mean(j_scores):.1f}/10" if j_scores else ""
        print(f"  {kv_name}: {p}/{len(results['answers'][kv_name])} passed{judge_str}")


def run_needle_suite(model, tokenizer, configs, contexts, output_path, results):
    """Run needle-in-haystack across contexts and configs with resume + OOM handling."""
    if "needles" not in results:
        results["needles"] = []
    done = {(n["context"], n["kv_config"], n["id"]) for n in results["needles"]}

    print(f"\n{'='*60}\n  NEEDLE-IN-HAYSTACK\n{'='*60}")

    # Pre-fetch filler text
    fillers = {}
    for ctx in contexts:
        target_chars = ctx * 4
        ctx_label = f"{ctx // 1024}K"
        print(f"  Fetching {ctx_label} filler ({target_chars // 1000}K chars)...", end=" ", flush=True)
        fillers[ctx] = fetch_filler(target_chars, offset=contexts.index(ctx) * 3)
        print(f"done ({len(fillers[ctx]) // 4} est. tokens)")

    for ctx in contexts:
        ctx_label = f"{ctx // 1024}K"
        filler = fillers[ctx]

        for kv_name in configs:
            bits = KV_MAP.get(kv_name, 4)

            for ni, needle in enumerate(NEEDLES):
                if (ctx_label, kv_name, needle["id"]) in done:
                    continue
                label = f"{kv_name} | {ctx_label} | {needle['id']}"
                print(f"\n  [{label}]", end=" ... ", flush=True)

                try:
                    raw_prompt = build_needle_prompt(filler, needle)
                    formatted = format_prompt(tokenizer, raw_prompt)
                    answer, metrics = generate_answer(model, tokenizer, formatted, kv_name, bits, max_tokens=200)
                    passed, details = check_answer(answer, needle["check_numbers"], needle["check_words"])

                    status_str = "PASS" if passed else "MISS"
                    print(f"{status_str} | prompt={metrics['prompt_tokens']} | "
                          f"gen={metrics['gen_tps']:.1f}t/s | KV={metrics['kv_fp16_mb']:.1f}MB", end="")
                    if metrics["cosine"] is not None:
                        print(f" | cos={metrics['cosine']} {metrics['ratio']}x", end="")
                    print()

                    entry = {
                        "id": needle["id"], "context": ctx_label,
                        "context_tokens": ctx, "kv_config": kv_name,
                        "question": needle["question"], "answer": answer,
                        "metrics": metrics,
                        "check_numbers": needle["check_numbers"],
                        "check_words": needle["check_words"],
                        "passed": passed, "check_details": details, "status": "OK",
                    }
                except Exception as e:
                    err = str(e)[:200]
                    is_oom = any(k in err.lower() for k in ["memory", "oom", "alloc", "metal"])
                    status = "OOM" if is_oom else "ERROR"
                    print(f"{status}: {err}")
                    entry = {
                        "id": needle["id"], "context": ctx_label,
                        "context_tokens": ctx, "kv_config": kv_name,
                        "question": needle["question"], "answer": "",
                        "metrics": {}, "check_numbers": needle["check_numbers"],
                        "check_words": needle["check_words"],
                        "passed": False, "check_details": {}, "status": status,
                    }
                    if is_oom:
                        gc.collect()
                        mx.metal.clear_cache()

                results["needles"].append(entry)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                gc.collect()
                mx.metal.clear_cache()


# ════════════════════════════════════════════════════════════
#  Aggregation
# ════════════════════════════════════════════════════════════

def aggregate(results, questions=None):
    """Compute summary statistics from raw results.
    If questions provided, also computes filtered scores using only reliable questions."""
    summary = {"quality": {}, "needles": {}, "degradation": {}}

    # Build reliable question set
    reliable_ids = None
    if questions:
        reliable_ids = {q["id"] for q in questions if q.get("reliable", True)}

    for cfg, answers in results.get("answers", {}).items():
        if not answers:
            continue
        m = [a["metrics"] for a in answers if a.get("metrics")]
        passed = sum(1 for a in answers if a.get("passed"))
        total = len(answers)

        by_cat = defaultdict(lambda: {"passed": 0, "total": 0})
        for a in answers:
            by_cat[a["category"]]["total"] += 1
            if a.get("passed"):
                by_cat[a["category"]]["passed"] += 1
        for cat in by_cat:
            t = by_cat[cat]["total"]
            by_cat[cat]["rate"] = round(by_cat[cat]["passed"] / t * 100, 1) if t else 0

        cosines = [x["cosine"] for x in m if x.get("cosine") is not None]
        ratios = [x["ratio"] for x in m if x.get("ratio") is not None]

        # Judge scores
        j_scores = [a["judge"]["score"] for a in answers if a.get("judge") and a["judge"].get("score", -1) >= 0]
        j_pass = sum(1 for a in answers if a.get("judge") and a["judge"].get("verdict") == "PASS")
        j_partial = sum(1 for a in answers if a.get("judge") and a["judge"].get("verdict") == "PARTIAL")

        # Judge by category
        j_by_cat = defaultdict(lambda: {"scores": [], "pass": 0, "total": 0})
        for a in answers:
            if a.get("judge") and a["judge"].get("score", -1) >= 0:
                j_by_cat[a["category"]]["scores"].append(a["judge"]["score"])
                j_by_cat[a["category"]]["total"] += 1
                if a["judge"].get("verdict") == "PASS":
                    j_by_cat[a["category"]]["pass"] += 1

        summary["quality"][cfg] = {
            "passed": passed, "total": total,
            "pass_rate": round(passed / total * 100, 1) if total else 0,
            "avg_gen_tps": round(mean(x["gen_tps"] for x in m), 1) if m else 0,
            "med_gen_tps": round(median(x["gen_tps"] for x in m), 1) if m else 0,
            "avg_ttft_ms": round(mean(x["ttft_ms"] for x in m), 0) if m else 0,
            "total_wall_s": round(sum(x["wall_ms"] for x in m) / 1000, 1) if m else 0,
            "avg_metal_peak_mb": round(mean(x["metal_peak_mb"] for x in m), 0) if m else 0,
            "avg_kv_fp16_mb": round(mean(x["kv_fp16_mb"] for x in m), 2) if m else 0,
            "avg_kv_compressed_mb": round(mean(x["kv_compressed_mb"] for x in m), 2) if m else 0,
            "avg_cosine": round(mean(cosines), 4) if cosines else None,
            "avg_ratio": round(mean(ratios), 2) if ratios else None,
            "avg_p50_token_ms": round(mean(x["p50_token_ms"] for x in m), 2) if m else 0,
            "avg_p99_token_ms": round(mean(x["p99_token_ms"] for x in m), 2) if m else 0,
            "by_category": dict(by_cat),
            "judge_avg_score": round(mean(j_scores), 2) if j_scores else None,
            "judge_pass": j_pass if j_scores else None,
            "judge_partial": j_partial if j_scores else None,
            "judge_by_category": {cat: {"avg_score": round(mean(v["scores"]), 2), "pass": v["pass"], "total": v["total"]} for cat, v in j_by_cat.items()} if j_scores else None,
        }

        # Filtered scores (reliable questions only)
        if reliable_ids:
            r_answers = [a for a in answers if a["id"] in reliable_ids]
            r_passed = sum(1 for a in r_answers if a.get("passed"))
            r_total = len(r_answers)
            r_j_scores = [a["judge"]["score"] for a in r_answers if a.get("judge") and a["judge"].get("score", -1) >= 0]
            r_j_pass = sum(1 for a in r_answers if a.get("judge") and a["judge"].get("verdict") == "PASS")
            summary["quality"][cfg]["reliable"] = {
                "passed": r_passed, "total": r_total,
                "pass_rate": round(r_passed / r_total * 100, 1) if r_total else 0,
                "judge_avg_score": round(mean(r_j_scores), 2) if r_j_scores else None,
                "judge_pass": r_j_pass,
            }

    # Needle summary
    for n in results.get("needles", []):
        cfg = n.get("kv_config", "?")
        ctx = n.get("context", "?")
        if cfg not in summary["needles"]:
            summary["needles"][cfg] = {}
        if ctx not in summary["needles"][cfg]:
            summary["needles"][cfg][ctx] = {"passed": 0, "total": 0}
        summary["needles"][cfg][ctx]["total"] += 1
        if n.get("passed"):
            summary["needles"][cfg][ctx]["passed"] += 1
    for cfg in summary["needles"]:
        for ctx in summary["needles"][cfg]:
            s = summary["needles"][cfg][ctx]
            s["rate"] = round(s["passed"] / s["total"] * 100, 1) if s["total"] else 0

    # Degradation vs fp16
    fp16_q = summary["quality"].get("fp16", {})
    fp16_n = summary["needles"].get("fp16", {})
    for cfg in summary["quality"]:
        if cfg == "fp16":
            continue
        cq = summary["quality"][cfg]
        q_delta = round(cq["pass_rate"] - fp16_q.get("pass_rate", 0), 1)
        tps_delta = round((cq["avg_gen_tps"] - fp16_q.get("avg_gen_tps", 0)) / fp16_q["avg_gen_tps"] * 100, 1) if fp16_q.get("avg_gen_tps") else 0
        mem_saving = round((1 - cq["avg_kv_compressed_mb"] / fp16_q["avg_kv_fp16_mb"]) * 100, 1) if fp16_q.get("avg_kv_fp16_mb") else 0

        # Needle delta
        n_delta = 0
        cn = summary["needles"].get(cfg, {})
        fp16_needle_total = sum(s["passed"] for s in fp16_n.values()) if fp16_n else 0
        fp16_needle_count = sum(s["total"] for s in fp16_n.values()) if fp16_n else 0
        cfg_needle_total = sum(s["passed"] for s in cn.values()) if cn else 0
        cfg_needle_count = sum(s["total"] for s in cn.values()) if cn else 0
        if fp16_needle_count and cfg_needle_count:
            n_delta = round((cfg_needle_total / cfg_needle_count - fp16_needle_total / fp16_needle_count) * 100, 1)

        summary["degradation"][f"{cfg}_vs_fp16"] = {
            "quality_delta": q_delta, "needle_delta": n_delta,
            "tps_delta_pct": tps_delta, "memory_savings_pct": mem_saving,
        }

    return summary


# ════════════════════════════════════════════════════════════
#  HTML Report
# ════════════════════════════════════════════════════════════

def generate_report(results, report_path):
    """Generate self-contained HTML report."""
    s = results.get("summary", {})
    model_short = results["model"].split("/")[-1]
    configs = results.get("configs_run", [])

    # Build HTML
    h = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>TQ Eval — {model_short}</title>
<style>
:root {{ --bg:#0f1117;--card:#21242f;--border:#2d3040;--text:#e0e0e8;--muted:#8b8fa0;--accent:#6366f1;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--teal:#14b8a6; }}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:24px;max-width:1200px;margin:0 auto;}}
h1{{font-size:24px;font-weight:700;margin-bottom:4px;}} h2{{font-size:18px;font-weight:600;margin:28px 0 12px;color:var(--accent);}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:20px;}}
.grid{{display:grid;gap:12px;}} .g3{{grid-template-columns:repeat(3,1fr);}} .g4{{grid-template-columns:repeat(4,1fr);}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;}}
.sv{{font-size:28px;font-weight:700;}} .sl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;}}
.ss{{font-size:12px;color:var(--muted);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} th{{text-align:left;padding:8px 10px;background:#1a1d27;color:var(--muted);font-size:11px;text-transform:uppercase;}}
td{{padding:8px 10px;border-bottom:1px solid var(--border);}}
.green{{color:var(--green);}} .red{{color:var(--red);}} .yellow{{color:var(--yellow);}} .teal{{color:var(--teal);}} .accent{{color:var(--accent);}}
.badge{{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:600;}}
.bp{{background:rgba(34,197,94,.15);color:var(--green);}} .bf{{background:rgba(239,68,68,.15);color:var(--red);}}
.divider{{height:1px;background:var(--border);margin:28px 0;}}
</style></head><body>
<h1>TurboQuant Eval — {model_short}</h1>
<div class="sub">{results.get('timestamp','')[:10]} | Configs: {', '.join(configs)} | {results.get('total_questions',0)} quality questions</div>
"""

    # Summary cards
    h += '<div class="grid g' + str(min(len(configs), 4)) + '">\n'
    for cfg in configs:
        q = s.get("quality", {}).get(cfg, {})
        pr = q.get("pass_rate", 0)
        cls = "green" if pr >= 70 else "yellow" if pr >= 50 else "red"
        cos = f"{q['avg_cosine']:.4f}" if q.get("avg_cosine") else "—"
        judge_str = f" | Judge: {q['judge_avg_score']:.1f}/10" if q.get("judge_avg_score") else ""
        h += f"""<div class="card" style="border-top:3px solid var(--accent);">
<div class="sv {cls}">{q.get('passed',0)}/{q.get('total',0)}</div>
<div class="sl">{cfg} pass rate ({pr}%)</div>
<div class="ss">TPS: {q.get('avg_gen_tps',0)} | TTFT: {q.get('avg_ttft_ms',0):.0f}ms | Cosine: {cos}{judge_str}</div>
</div>\n"""
    h += '</div>\n'

    # Degradation
    deg = s.get("degradation", {})
    if deg:
        h += '<h2>Degradation vs FP16</h2><div class="card"><table><tr><th>Config</th><th>Quality Delta</th><th>Needle Delta</th><th>TPS Change</th><th>Memory Saved</th></tr>\n'
        for key, d in deg.items():
            qd = d.get("quality_delta", 0)
            nd = d.get("needle_delta", 0)
            td = d.get("tps_delta_pct", 0)
            ms = d.get("memory_savings_pct", 0)
            qc = "green" if qd >= 0 else "red"
            nc = "green" if nd >= 0 else "red"
            h += f'<tr><td><strong>{key}</strong></td><td class="{qc}">{qd:+.1f}%</td><td class="{nc}">{nd:+.1f}%</td><td class="{"green" if td>=0 else "yellow"}">{td:+.1f}%</td><td class="teal">{ms:.1f}%</td></tr>\n'
        h += '</table></div>\n'

    # Performance table
    h += '<h2>Performance Comparison</h2><div class="card"><table><tr><th>Metric</th>'
    for cfg in configs:
        h += f'<th>{cfg}</th>'
    h += '</tr>\n'
    perf_rows = [
        ("Avg Gen TPS", "avg_gen_tps"), ("Median Gen TPS", "med_gen_tps"),
        ("Avg TTFT (ms)", "avg_ttft_ms"), ("Total Wall (s)", "total_wall_s"),
        ("Avg Peak Metal (MB)", "avg_metal_peak_mb"),
        ("Avg KV FP16 (MB)", "avg_kv_fp16_mb"), ("Avg KV Compressed (MB)", "avg_kv_compressed_mb"),
        ("Avg Cosine", "avg_cosine"), ("Avg Ratio", "avg_ratio"),
        ("Avg p50 Latency (ms)", "avg_p50_token_ms"), ("Avg p99 Latency (ms)", "avg_p99_token_ms"),
    ]
    for label, key in perf_rows:
        h += f'<tr><td>{label}</td>'
        for cfg in configs:
            v = s.get("quality", {}).get(cfg, {}).get(key, "—")
            h += f'<td>{v if v is not None else "—"}</td>'
        h += '</tr>\n'
    h += '</table></div>\n'

    # Category breakdown
    cats = set()
    for cfg in configs:
        cats.update(s.get("quality", {}).get(cfg, {}).get("by_category", {}).keys())
    if cats:
        h += '<h2>Category Breakdown</h2><div class="card"><table><tr><th>Category</th>'
        for cfg in configs:
            h += f'<th>{cfg}</th>'
        h += '</tr>\n'
        for cat in sorted(cats):
            h += f'<tr><td><strong>{cat}</strong></td>'
            for cfg in configs:
                cd = s.get("quality", {}).get(cfg, {}).get("by_category", {}).get(cat, {})
                p, t = cd.get("passed", 0), cd.get("total", 0)
                r = cd.get("rate", 0)
                h += f'<td>{p}/{t} ({r}%)</td>'
            h += '</tr>\n'
        h += '</table></div>\n'

    # Needle results
    needle_s = s.get("needles", {})
    if needle_s:
        h += '<h2>Needle-in-Haystack</h2><div class="card"><table><tr><th>Context</th>'
        for cfg in configs:
            h += f'<th>{cfg}</th>'
        h += '</tr>\n'
        all_ctx = sorted(set(ctx for cfg_data in needle_s.values() for ctx in cfg_data))
        for ctx in all_ctx:
            h += f'<tr><td><strong>{ctx}</strong></td>'
            for cfg in configs:
                nd = needle_s.get(cfg, {}).get(ctx, {})
                p, t = nd.get("passed", 0), nd.get("total", 0)
                cls = "green" if p == t and t > 0 else "red" if p == 0 and t > 0 else "muted"
                h += f'<td class="{cls}">{p}/{t}</td>'
            h += '</tr>\n'
        h += '</table></div>\n'

    h += f'<div class="divider"></div><p class="ss" style="text-align:center;">TurboQuant Eval v1.0 | {model_short} | {results.get("timestamp","")[:10]}</p></body></html>'

    with open(report_path, "w") as f:
        f.write(h)
    print(f"\n  Report: {report_path}")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Unified Evaluation Suite")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--configs", nargs="+", default=["fp16", "tq_4bit", "tq_3bit"],
                        choices=["fp16", "tq_4bit", "tq_3bit"])
    parser.add_argument("--max-questions", type=int, default=0, help="0=all 65")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768],
                        help="Needle context lengths (e.g. 32768 65536)")
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--report", default=None, help="HTML report path")
    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge scoring (config in .env: JUDGE_ENDPOINT, JUDGE_API_KEY, JUDGE_MODEL)")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model name (default: from .env JUDGE_MODEL)")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].replace("-", "_").lower()
    if args.output is None:
        args.output = os.path.join(ROOT, f"tq_eval_{model_short}.json")

    # Resume: load existing results if output file exists
    if os.path.isfile(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"  Resuming from {args.output}")
        for cfg in results.get("answers", {}):
            print(f"    {cfg}: {len(results['answers'][cfg])} quality answers done")
        print(f"    needles: {len(results.get('needles', []))} done")
    else:
        results = {
            "benchmark": "TurboQuant Eval v1.0",
            "model": args.model,
            "model_config": {},
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "configs_run": args.configs,
            "chat_template": True,
            "answers": {},
            "needles": [],
        }

    # Load questions
    with open(QUESTIONS_FILE) as f:
        qdata = json.load(f)
    questions = qdata["questions"]
    if args.max_questions > 0:
        questions = questions[:args.max_questions]
    results["total_questions"] = len(questions)
    results["questions_version"] = qdata.get("version", "?")

    # Load model
    print(f"\n{'='*70}")
    print(f"  TurboQuant Eval v1.0")
    print(f"  Model:     {args.model}")
    print(f"  Configs:   {', '.join(args.configs)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Needle:    {'skip' if args.skip_needle else ', '.join(f'{c//1024}K' for c in args.contexts)}")
    print(f"  Output:    {args.output}")
    print(f"{'='*70}")

    model, tokenizer = mlx_load(args.model)
    config = get_model_config(model)
    results["model_config"] = config
    print(f"  Config: {config}")

    test_fmt = format_prompt(tokenizer, "test")
    print(f"  Chat template: {repr(test_fmt[:60])}...")

    # Init LLM judge if requested
    judge_client = None
    judge_model_name = None
    if args.judge:
        judge_client, judge_model_name = init_judge_client()
        if args.judge_model:
            judge_model_name = args.judge_model
        if judge_client:
            print(f"  LLM Judge: {judge_model_name} @ {os.environ.get('JUDGE_ENDPOINT', 'default')}")
        else:
            print(f"  LLM Judge: DISABLED (check .env file — JUDGE_API_KEY missing or invalid)")

    # Run quality suite
    run_quality_suite(model, tokenizer, questions, args.configs, args.max_tokens, args.output, results, judge_client=judge_client, judge_model=judge_model_name)

    # Run needle suite
    if not args.skip_needle:
        run_needle_suite(model, tokenizer, args.configs, args.contexts, args.output, results)

    # Aggregate
    results["summary"] = aggregate(results, questions=questions)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Report
    if args.report:
        generate_report(results, args.report)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS (54 reliable questions)")
    print(f"{'='*70}")
    for cfg, q in results["summary"].get("quality", {}).items():
        cos = f"cos={q['avg_cosine']:.4f}" if q.get("avg_cosine") else ""
        r = q.get("reliable", {})
        if r:
            score = f" | score={r['judge_avg_score']:.1f}/10" if r.get("judge_avg_score") else ""
            print(f"  {cfg:>10}: {r['passed']}/{r['total']} ({r['pass_rate']}%){score} | "
                  f"TPS={q['avg_gen_tps']} | TTFT={q['avg_ttft_ms']:.0f}ms {cos}")
        else:
            score = f" | score={q['judge_avg_score']:.1f}/10" if q.get("judge_avg_score") else ""
            print(f"  {cfg:>10}: {q['passed']}/{q['total']} ({q['pass_rate']}%){score} | "
                  f"TPS={q['avg_gen_tps']} | TTFT={q['avg_ttft_ms']:.0f}ms {cos}")
    for cfg, ctx_data in results["summary"].get("needles", {}).items():
        for ctx, nd in ctx_data.items():
            print(f"  {cfg:>10} @ {ctx}: {nd['passed']}/{nd['total']} ({nd['rate']}%)")
    for key, d in results["summary"].get("degradation", {}).items():
        print(f"  {key}: quality={d['quality_delta']:+.1f}% needle={d['needle_delta']:+.1f}% "
              f"tps={d['tps_delta_pct']:+.1f}% mem_saved={d['memory_savings_pct']:.1f}%")

    print(f"\n  Output: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
