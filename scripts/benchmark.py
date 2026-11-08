#!/usr/bin/env python3
import time
import torch
from transformers import AutoTokenizer
from gemmainfer.utils.weights import load_gemma_from_hf
from gemmainfer.core.engine import GemmaInferEngine

def run_benchmark():
    model_path = "/root/models/gemma-4-26B-A4B-it-W4A16"
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, config = load_gemma_from_hf(model_path, device=torch.device("cuda"), dtype=torch.bfloat16)
    engine = GemmaInferEngine(config=config, model=model, device="cuda", dtype=torch.bfloat16, enable_cuda_graphs=False)

    prompts = [
        ("Short Prompt (12 tok)", "What is an operating system? Answer in one short sentence:"),
        ("Medium Prompt (48 tok)", "Write a concise explanation of how neural network backpropagation works using gradient descent:"),
        ("Code Prompt (35 tok)", "Write a clean Python function for binary search with comments:")
    ]

    print("\n" + "="*75)
    header = f"{'Scenario':<26} | {'TTFT (ms)':<10} | {'Tokens':<8} | {'Decode tok/s':<14} | {'Total Time':<10}"
    print(header)
    print("="*75)

    for name, p_text in prompts:
        messages = [{"role": "user", "content": p_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt)

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        gen = engine.generate_stream(input_ids, max_tokens=60, temperature=0.7)

        first_tok = next(gen)
        torch.cuda.synchronize()
        t_first = time.perf_counter()
        ttft_ms = (t_first - t_start) * 1000.0

        tokens = [first_tok]
        for tok in gen:
            tokens.append(tok)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_time = t_end - t_first
        gen_tokens = len(tokens) - 1
        tps = gen_tokens / gen_time if gen_time > 0 else 0
        total_time = t_end - t_start

        row = f"{name:<26} | {ttft_ms:<10.1f} | {len(tokens):<8} | {tps:<14.2f} | {total_time:<10.2f}s"
        print(row)

    print("="*75)
    vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak VRAM: {vram:.2f} GB / 24.00 GB (Free: {24.0 - vram:.2f} GB)")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_benchmark()
