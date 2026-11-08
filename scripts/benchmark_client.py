#!/usr/bin/env python3
"""Live HTTP endpoint benchmarking client for GemmaInfer OpenAI-compatible server.

Measures Time-To-First-Token (TTFT), streaming decode throughput (tok/s),
and end-to-end request latency across standard chat, reasoning, and coding prompts.

Usage:
  # Benchmark local server on default port 8000:
  python scripts/benchmark_client.py

  # Benchmark with custom host and port:
  python scripts/benchmark_client.py --host 127.0.0.1 --port 8000

  # Target custom URL endpoint:
  python scripts/benchmark_client.py --url http://127.0.0.1:8000/v1/chat/completions
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import List, Tuple


def benchmark_endpoint(args):
    url = args.url if args.url else f"http://{args.host}:{args.port}/v1/chat/completions"

    prompts: List[Tuple[str, str]] = [
        ("Short Prompt (Chat)", "What is an operating system? Answer in one short sentence:"),
        ("Reasoning / Math", "Solve step by step: What is the sum of integers from 1 to 50?"),
        ("Code Generation", "Write a clean Python function for binary search with comments:"),
    ]

    print("\n" + "=" * 80)
    print(" GEMMAINFER LIVE ENDPOINT BENCHMARK CLIENT")
    print("=" * 80)
    print(f"Target URL:        {url}")
    print(f"Model ID:          {args.model}")
    print(f"Max Tokens:        {args.max_tokens}")
    print(f"Temperature:       {args.temperature}")
    print("=" * 80)
    header = f"{'Scenario':<25} | {'TTFT (ms)':<10} | {'Tokens':<8} | {'Decode (tok/s)':<14} | {'Total Time (s)':<14}"
    print(header)
    print("-" * 80)

    for name, prompt_text in prompts:
        payload = json.dumps({
            "model": args.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t_start = time.perf_counter()
        t_first = None
        chunk_times = []
        token_count = 0

        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        t_now = time.perf_counter()
                        if t_first is None:
                            t_first = t_now
                        chunk_times.append(t_now)
                        token_count += 1
        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to connect to endpoint {url}: {e.reason}")
            print(f"Ensure GemmaInfer server is running: python scripts/serve.py --host {args.host} --port {args.port}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            sys.exit(1)

        t_end = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000.0 if t_first else 0.0
        decode_duration = t_end - t_first if t_first else 0.0
        decode_tokens = token_count - 1 if token_count > 1 else token_count
        tok_s = decode_tokens / decode_duration if decode_duration > 0 else 0.0
        total_time = t_end - t_start

        print(f"{name:<25} | {ttft_ms:<10.1f} | {token_count:<8} | {tok_s:<14.2f} | {total_time:<14.2f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="GemmaInfer Live HTTP Serving Endpoint Benchmark Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Serving server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Serving server port",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Full URL to chat completions endpoint (overrides --host and --port)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4-26b-a4b",
        help="Model identifier name for payload",
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
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds",
    )
    args = parser.parse_args()
    benchmark_endpoint(args)


if __name__ == "__main__":
    main()
