#!/usr/bin/env python3
"""Benchmark latency, TTFT, and generation throughput for GemmaInfer.

Evaluates Gemma 4 26B A4B across representative prompt scenarios (Short chat,
reasoning, and code generation) measuring Time-To-First-Token (TTFT) and decode tok/s.

Usage:
  # Benchmark with synthetic weights (zero download, tests memory footprint & latency):
  python scripts/benchmark.py --synthetic --device cuda

  # Benchmark with real HuggingFace checkpoint:
  python scripts/benchmark.py --model-path /path/to/gemma-4-26B-A4B-it-W4A16 --device cuda
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.core.engine import GemmaInferEngine
from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.utils.weights import load_gemma_from_hf


def setup_tokenizer(model_path: str = None):
    if model_path and os.path.exists(model_path):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[WARN] Could not load AutoTokenizer from {model_path}: {e}")

    class SimpleTokenizer:
        def encode(self, text: str) -> List[int]:
            return [ord(c) % 256000 for c in text]

        def decode(self, token_ids: List[int]) -> str:
            return "".join([chr(t % 128) if 32 <= (t % 128) <= 126 else " " for t in token_ids])

    return SimpleTokenizer()


def format_prompt(tokenizer, user_text: str) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(prompt_str)
    prompt_str = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    return tokenizer.encode(prompt_str)


def run_benchmark(args):
    dtype = getattr(torch, args.dtype)
    is_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

    print("=" * 80)
    print(" GEMMAINFER INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Device:            {args.device}")
    if is_cuda:
        print(f"GPU Hardware:      {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:        {total_vram:.2f} GB")
    print(f"Precision:         {args.dtype}")
    print(f"Speculative DSpark:{not args.no_dspark}")
    print(f"CUDA Graphs:       {not args.no_cuda_graphs if is_cuda else False}")
    print("=" * 80)

    assistant_model = None
    if args.synthetic or not (args.model_path and os.path.exists(args.model_path)):
        if not args.synthetic and args.model_path:
            print(f"[INFO] Model path '{args.model_path}' not found on host. Running with synthetic Gemma 4 26B A4B.")
        else:
            print("[INFO] Initializing synthetic Gemma 4 26B A4B weights...")
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        model = GemmaForCausalLM(config).to(device=args.device, dtype=dtype)
        tokenizer = setup_tokenizer(None)
    else:
        print(f"Loading weights from {args.model_path}...")
        model, config = load_gemma_from_hf(args.model_path, device=torch.device(args.device), dtype=dtype)
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
                print(f"Loaded assistant drafter: {type(assistant_model).__name__}")
            except Exception as e:
                print(f"[WARN] Failed to load assistant model: {e}")

    engine = GemmaInferEngine(
        config=config,
        model=model,
        assistant_model=assistant_model,
        device=args.device,
        dtype=dtype,
        enable_cuda_graphs=not args.no_cuda_graphs if is_cuda else False,
        enable_dspark=not args.no_dspark,
    )

    prompts: List[Tuple[str, str]] = [
        ("Short Prompt (Chat)", "What is an operating system? Answer in one short sentence:"),
        ("Reasoning / Math", "Solve step by step: What is the sum of integers from 1 to 50?"),
        ("Code Generation", "Write a clean Python function for binary search with comments:"),
    ]

    # Warmup pass
    warmup_ids = format_prompt(tokenizer, "Hello")
    for _ in engine.generate_stream(warmup_ids, max_tokens=5, temperature=0.0):
        pass
    if is_cuda:
        torch.cuda.synchronize()

    print("\n" + "=" * 80)
    header = f"{'Scenario':<26} | {'TTFT (ms)':<10} | {'Tokens':<8} | {'Decode tok/s':<14} | {'Total Time':<10}"
    print(header)
    print("-" * 80)

    for name, p_text in prompts:
        input_ids = format_prompt(tokenizer, p_text)

        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        gen = engine.generate_stream(
            input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        try:
            first_tok = next(gen)
        except StopIteration:
            continue

        if is_cuda:
            torch.cuda.synchronize()
        t_first = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000.0

        tokens = [first_tok]
        for tok in gen:
            tokens.append(tok)

        if is_cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_time = t_end - t_first
        gen_tokens = len(tokens) - 1
        tps = gen_tokens / gen_time if gen_time > 0 else 0.0
        total_time = t_end - t_start

        row = f"{name:<26} | {ttft_ms:<10.1f} | {len(tokens):<8} | {tps:<14.2f} | {total_time:<9.2f}s"
        print(row)

    print("=" * 80)
    if is_cuda:
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Allocated VRAM: {vram_alloc:.2f} GB | Peak VRAM: {vram_peak:.2f} GB / {total_vram:.2f} GB")
        print(f"Available VRAM Headroom: {total_vram - vram_peak:.2f} GB")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GemmaInfer Performance and Throughput Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/models/gemma-4-26B-A4B-it-W4A16",
        help="Path to HuggingFace model weights directory",
    )
    parser.add_argument(
        "--assistant-model-path",
        type=str,
        default=None,
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
        help="Model weights precision data type",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Sampling top-p probability threshold",
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Disable CUDA Graph capture for decode",
    )
    parser.add_argument(
        "--no-dspark",
        action="store_true",
        help="Disable DSpark speculative decoding acceleration",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
