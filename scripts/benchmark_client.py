#!/usr/bin/env python3
import time
import json
import urllib.request

def benchmark_endpoint(url="http://127.0.0.1:8080/v1/chat/completions"):
    prompts = [
        ("Short Prompt (Chat)", "What is an operating system? Answer in one short sentence:"),
        ("Reasoning / Math", "Solve step by step: What is the sum of integers from 1 to 50?"),
        ("Code Generation", "Write a clean Python function for binary search with comments:")
    ]

    print("\n" + "="*80)
    print(" GEMMAINFER LIVE ENDPOINT BENCHMARK (RTX 3090 - Gemma 4 26B A4B W4A16)")
    print("="*80)
    header = f"{'Scenario':<25} | {'TTFT (ms)':<10} | {'Tokens':<8} | {'Decode (tok/s)':<14} | {'Total Time (s)':<14}"
    print(header)
    print("-" * 80)

    for name, prompt_text in prompts:
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 60,
            "temperature": 0.2,
            "stream": True
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        t_start = time.perf_counter()
        t_first = None
        chunk_times = []
        token_count = 0

        with urllib.request.urlopen(req) as resp:
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

        t_end = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000.0 if t_first else 0.0
        decode_duration = t_end - t_first if t_first else 0.0
        decode_tokens = token_count - 1 if token_count > 1 else token_count
        tok_s = decode_tokens / decode_duration if decode_duration > 0 else 0.0
        total_time = t_end - t_start

        print(f"{name:<25} | {ttft_ms:<10.1f} | {token_count:<8} | {tok_s:<14.2f} | {total_time:<14.2f}")

    print("="*80 + "\n")

if __name__ == "__main__":
    benchmark_endpoint()
