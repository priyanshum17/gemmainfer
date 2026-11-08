#!/usr/bin/env python3
"""Test and benchmark speculative decoding with Gemma 4 26B A4B and Assistant / Prompt Lookup."""

import os
import sys
import time
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.utils.weights import load_gemma_from_hf
from gemmainfer.core.engine import GemmaInferEngine

def benchmark_speculative():
    model_path = "/root/models/gemma-4-26B-A4B-it-W4A16"
    assistant_path = "/root/models/gemma-4-26B-A4B-it-assistant"

    print("=" * 80)
    print(" GEMMAINFER SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print("Loading main model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, config = load_gemma_from_hf(model_path, device=torch.device("cuda"), dtype=torch.bfloat16)

    # 1. Benchmark without speculative decoding (Baseline)
    print("\n[1/2] Running baseline engine (pure autoregression)...")
    engine_base = GemmaInferEngine(
        config=config,
        model=model,
        device="cuda",
        dtype=torch.bfloat16,
        enable_cuda_graphs=False,
        enable_dspark=False,
    )

    prompt = "Write a complete, clean Python function for binary search with detailed docstring and comments:"
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_str)

    # Warmup
    for _ in engine_base.generate_stream(input_ids, max_tokens=10, temperature=0.0):
        pass
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    tokens_base = []
    gen_base = engine_base.generate_stream(input_ids, max_tokens=60, temperature=0.0)
    first_tok = next(gen_base)
    t_first_base = time.perf_counter()
    tokens_base.append(first_tok)
    for tok in gen_base:
        tokens_base.append(tok)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ttft_base = (t_first_base - t0) * 1000.0
    gen_time_base = t1 - t_first_base
    speed_base = (len(tokens_base) - 1) / gen_time_base if gen_time_base > 0 else 0

    print(f"  * Baseline Decode Speed: {speed_base:.2f} tok/s (TTFT: {ttft_base:.1f} ms, Tokens: {len(tokens_base)})")

    # 2. Benchmark with Speculative Decoding enabled
    print("\n[2/2] Running speculative decoding engine (with prompt lookup & draft verification)...")
    engine_spec = GemmaInferEngine(
        config=config,
        model=model,
        device="cuda",
        dtype=torch.bfloat16,
        enable_cuda_graphs=False,
        enable_dspark=True,
    )

    t0 = time.perf_counter()
    tokens_spec = []
    gen_spec = engine_spec.generate_stream(input_ids, max_tokens=60, temperature=0.0)
    first_tok = next(gen_spec)
    t_first_spec = time.perf_counter()
    tokens_spec.append(first_tok)
    for tok in gen_spec:
        tokens_spec.append(tok)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ttft_spec = (t_first_spec - t0) * 1000.0
    gen_time_spec = t1 - t_first_spec
    speed_spec = (len(tokens_spec) - 1) / gen_time_spec if gen_time_spec > 0 else 0

    print(f"  * Speculative Decode Speed: {speed_spec:.2f} tok/s (TTFT: {ttft_spec:.1f} ms, Tokens: {len(tokens_spec)})")

    print("\n" + "=" * 80)
    print(" SUMMARY COMPARISON")
    print("=" * 80)
    print(f"Baseline Pure Autoregression : {speed_base:.2f} tok/s")
    print(f"Speculative Decoding         : {speed_spec:.2f} tok/s")
    if speed_base > 0:
        print(f"Speedup                      : {speed_spec / speed_base:.2f}x ({(speed_spec - speed_base)/speed_base * 100.0:+.1f}%)")
    print("=" * 80)

    text = tokenizer.decode(tokens_spec)
    print("\nGenerated Text Sample:")
    print(text[:300] + ("..." if len(text) > 300 else ""))

if __name__ == "__main__":
    benchmark_speculative()
