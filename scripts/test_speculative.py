#!/usr/bin/env python3
"""Test and benchmark speculative decoding with Gemma 4 26B A4B and Assistant / Prompt Lookup.

Usage:
  # Synthetic evaluation (zero download, immediate GPU verification):
  python scripts/test_speculative.py --synthetic --device cuda

  # Real weights evaluation:
  python scripts/test_speculative.py --model-path /path/to/base --assistant-model-path /path/to/assistant
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.utils.weights import load_gemma_from_hf
from gemmainfer.core.engine import GemmaInferEngine


def setup_tokenizer(model_path: str = None):
    if model_path and os.path.exists(model_path):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[WARN] Could not load AutoTokenizer from {model_path}: {e}")

    class SimpleTokenizer:
        def encode(self, text: str):
            return [ord(c) % 256000 for c in text]

        def decode(self, token_ids):
            return "".join([chr(t % 128) if 32 <= (t % 128) <= 126 else " " for t in token_ids])

    return SimpleTokenizer()


def benchmark_speculative(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    is_cuda = device.type == "cuda" and torch.cuda.is_available()

    print("=" * 80)
    print(" GEMMAINFER SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print(f"Device:             {args.device}")
    print(f"Precision:          {args.dtype}")
    print(f"Max Tokens:         {args.max_tokens}")

    assistant_model = None
    if args.synthetic or not (args.model_path and os.path.exists(args.model_path)):
        if not args.synthetic and args.model_path:
            print(f"[INFO] Model path '{args.model_path}' not found. Initializing synthetic Gemma 4 26B A4B.")
        else:
            print("[INFO] Initializing synthetic Gemma 4 26B A4B weights...")
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        model = GemmaForCausalLM(config).to(device=device, dtype=dtype)
        tokenizer = setup_tokenizer(None)
    else:
        print(f"Loading weights from {args.model_path}...")
        model, config = load_gemma_from_hf(args.model_path, device=device, dtype=dtype)
        tokenizer = setup_tokenizer(args.model_path)

        if args.assistant_model_path and os.path.exists(args.assistant_model_path):
            try:
                from transformers import AutoModelForCausalLM
                print(f"Loading assistant model from {args.assistant_model_path}...")
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    args.assistant_model_path,
                    dtype=dtype,
                    device_map=args.device,
                )
                print(f"Loaded assistant model successfully: {type(assistant_model)}")
            except Exception as e:
                print(f"[WARN] Failed to load assistant model: {e}")

    # 1. Benchmark without speculative decoding (Baseline)
    print("\n[1/2] Running baseline engine (pure autoregression)...")
    engine_base = GemmaInferEngine(
        config=config,
        model=model,
        device=args.device,
        dtype=dtype,
        enable_cuda_graphs=False,
        enable_dspark=False,
    )

    prompt = "Write a complete, clean Python function for binary search with detailed docstring and comments:"
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt_str)
    else:
        input_ids = tokenizer.encode(prompt)

    # Warmup
    for _ in engine_base.generate_stream(input_ids, max_tokens=10, temperature=0.0):
        pass
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    tokens_base = []
    gen_base = engine_base.generate_stream(input_ids, max_tokens=args.max_tokens, temperature=0.0)
    try:
        first_tok = next(gen_base)
    except StopIteration:
        first_tok = 1
    t_first_base = time.perf_counter()
    tokens_base.append(first_tok)
    for tok in gen_base:
        tokens_base.append(tok)
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ttft_base = (t_first_base - t0) * 1000.0
    gen_time_base = t1 - t_first_base
    speed_base = (len(tokens_base) - 1) / gen_time_base if gen_time_base > 0 else 0.0

    print(f"  * Baseline Decode Speed: {speed_base:.2f} tok/s (TTFT: {ttft_base:.1f} ms, Tokens: {len(tokens_base)})")

    # 2. Benchmark with Speculative Decoding enabled
    print("\n[2/2] Running speculative decoding engine (with prompt lookup & draft verification)...")
    engine_spec = GemmaInferEngine(
        config=config,
        model=model,
        assistant_model=assistant_model,
        device=args.device,
        dtype=dtype,
        enable_cuda_graphs=False,
        enable_dspark=True,
    )

    t0 = time.perf_counter()
    tokens_spec = []
    gen_spec = engine_spec.generate_stream(input_ids, max_tokens=args.max_tokens, temperature=0.0)
    try:
        first_tok = next(gen_spec)
    except StopIteration:
        first_tok = 1
    t_first_spec = time.perf_counter()
    tokens_spec.append(first_tok)
    for tok in gen_spec:
        tokens_spec.append(tok)
    if is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ttft_spec = (t_first_spec - t0) * 1000.0
    gen_time_spec = t1 - t_first_spec
    speed_spec = (len(tokens_spec) - 1) / gen_time_spec if gen_time_spec > 0 else 0.0

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


def main():
    parser = argparse.ArgumentParser(
        description="GemmaInfer Speculative Decoding Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("GEMMA_MODEL_PATH", None),
        help="Path to HuggingFace model weights directory",
    )
    parser.add_argument(
        "--assistant-model-path",
        type=str,
        default=os.environ.get("GEMMA_ASSISTANT_PATH", None),
        help="Path to companion draft assistant model directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run benchmark with synthetic Gemma 4 26B A4B weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for benchmark (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Weights precision data type",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Number of tokens to generate per pass",
    )
    args = parser.parse_args()
    benchmark_speculative(args)


if __name__ == "__main__":
    main()
