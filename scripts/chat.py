#!/usr/bin/env python3
"""Interactive CLI generation runner for GemmaInfer on RTX 3090 / 4090.

Supports real HuggingFace checkpoints and zero-download synthetic evaluation
to verify 24GB memory footprint, TTFT, and generation throughput immediately.

Usage:
  # Synthetic Gemma 4 26B A4B evaluation (tests GPU latency without downloading weights):
  python scripts/chat.py --synthetic --device cuda

  # Real weights inference:
  python scripts/chat.py --model-path /path/to/gemma-4-26b --device cuda
"""

import argparse
import sys
import time
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.engine import GemmaInferEngine
from gemmainfer.utils.weights import load_gemma_from_hf


def setup_tokenizer(model_path: str = None):
    if model_path:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[WARN] Could not load AutoTokenizer from {model_path}: {e}")
    
    # Fallback character/ascii tokenizer for synthetic testing
    class SimpleTokenizer:
        def encode(self, text: str):
            return [ord(c) % 256000 for c in text]

        def decode(self, token_ids):
            return "".join([chr(t % 128) if 32 <= (t % 128) <= 126 else " " for t in token_ids])

    return SimpleTokenizer()


def main():
    parser = argparse.ArgumentParser(description="GemmaInfer Interactive CLI")
    parser.add_argument("--model-path", type=str, default=None, help="Path to HuggingFace model weights")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic Gemma 4 26B A4B weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--no-dspark", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print("=" * 65)
    print(" GemmaInfer Interactive Inference Engine")
    print("=" * 65)
    print(f"Device:            {args.device}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU Model:         {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity:     {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Precision:         {args.dtype}")
    print(f"CUDA Graphs:       {not args.no_cuda_graphs}")
    print(f"DSpark Drafter:    {not args.no_dspark}")
    print("=" * 65)

    tokenizer = setup_tokenizer(args.model_path)

    if args.model_path and not args.synthetic:
        print(f"\nLoading model weights from {args.model_path}...")
        t0 = time.time()
        model, config = load_gemma_from_hf(args.model_path, device=torch.device(args.device), dtype=dtype)
        print(f"Loaded weights in {time.time() - t0:.2f}s")
    else:
        print("\nInitializing Gemma 4 26B A4B synthetic model on GPU...")
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        model = GemmaForCausalLM(config).to(device=args.device, dtype=dtype)

    engine = GemmaInferEngine(
        config=config,
        model=model,
        device=args.device,
        dtype=dtype,
        enable_cuda_graphs=not args.no_cuda_graphs,
        enable_dspark=not args.no_dspark,
    )

    if args.device == "cuda":
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"Allocated GPU VRAM: {allocated:.2f} GB / 24.00 GB")

    print("\nReady! Type a prompt and press Enter. (Type 'quit' or 'exit' to stop)\n")

    while True:
        try:
            prompt = input("User > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit"]:
                print("Exiting GemmaInfer.")
                break

            prompt_ids = tokenizer.encode(prompt)
            if not prompt_ids:
                prompt_ids = [1]

            print("GemmaInfer > ", end="", flush=True)

            t_start = time.perf_counter()
            first_token_time = None
            generated_tokens = []

            for token_id in engine.generate_stream(
                prompt_token_ids=prompt_ids,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_tokens.append(token_id)
                text_chunk = tokenizer.decode([token_id])
                print(text_chunk, end="", flush=True)

            t_end = time.perf_counter()
            print()

            total_time = t_end - t_start
            num_tokens = len(generated_tokens)
            ttft_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else 0.0
            decode_time = t_end - (first_token_time or t_start)
            tpot_ms = ((decode_time / max(num_tokens - 1, 1)) * 1000.0) if num_tokens > 1 else 0.0
            tokens_per_sec = num_tokens / max(total_time, 1e-4)

            print(f"[{num_tokens} tokens | TTFT: {ttft_ms:.1f}ms | TPOT: {tpot_ms:.1f}ms | {tokens_per_sec:.1f} tok/s]\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}\n")


if __name__ == "__main__":
    main()
