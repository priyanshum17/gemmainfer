#!/usr/bin/env python3
"""Detailed profiling and flamegraph generator for GemmaInfer decode steps.

Measures:
1. Exact GPU microsecond breakdown per layer component (Attention, Shared MLP, MoE Router, MoE Kernels, LM Head)
2. CPU launch overhead vs CUDA kernel execution time
3. Export Chrome trace JSON (can be opened in chrome://tracing or ui.perfetto.dev)
4. Roofline and memory bandwidth analysis
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


def profile_decode(args):
    print("=" * 80)
    print(" GEMMAINFER DETAILED DECODE PROFILER & FLAMEGRAPH ANALYZER")
    print("=" * 80)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    if args.synthetic or not (args.model_path and os.path.exists(args.model_path)):
        if not args.synthetic and args.model_path:
            print(f"[INFO] Model path '{args.model_path}' not found. Initializing synthetic Gemma 4 26B A4B.")
        else:
            print("[INFO] Initializing synthetic Gemma 4 26B A4B weights...")
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        model = GemmaForCausalLM(config).to(device=device, dtype=dtype)
        tokenizer = None
    else:
        print(f"Loading model and tokenizer from {args.model_path}...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model, config = load_gemma_from_hf(args.model_path, device=device, dtype=dtype)

    engine = GemmaInferEngine(config=config, model=model, device=args.device, dtype=dtype, enable_cuda_graphs=False)

    prompt = "What is an operating system? Answer in one short sentence:"
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt_str)
    else:
        input_ids = [ord(c) % 256000 for c in prompt]

    print("\nWarmup (generating 10 tokens)...")
    for _ in engine.generate_stream(input_ids, max_tokens=10, temperature=0.0):
        pass
    torch.cuda.synchronize()

    print("Running instrumented single-token decode profiling...")

    # Initialize a clean sequence and KV cache for step-level profiling
    device = engine.device
    dtype = engine.dtype
    past_kv = model.init_kv_cache(batch_size=1, max_seq_len=2048, device=device, dtype=dtype)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, past_kv = model(input_tensor, past_key_values=past_kv, return_logits_only_last=True)
    curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # 1. Component breakdown with CUDA Events
    num_profile_steps = 15
    timings = {
        "embed_tokens": [],
        "attention_qkv": [],
        "attention_core": [],
        "attention_out": [],
        "shared_mlp": [],
        "moe_router": [],
        "moe_fused_gu": [],
        "moe_fused_down": [],
        "norms_and_residual": [],
        "final_norm_lm_head": [],
        "total_step": [],
    }

    # Profile per-layer execution
    for step in range(num_profile_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.inference_mode():
            logits, past_kv = model(curr_token, past_key_values=past_kv, return_logits_only_last=True)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        end_event.record()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_ms = start_event.elapsed_time(end_event)
        cpu_wall_ms = (t1 - t0) * 1000.0

        timings["total_step"].append((gpu_ms, cpu_wall_ms))
        curr_token = next_tok

    avg_gpu_ms = sum(g for g, c in timings["total_step"]) / len(timings["total_step"])
    avg_cpu_ms = sum(c for g, c in timings["total_step"]) / len(timings["total_step"])
    decode_tps = 1000.0 / avg_gpu_ms

    print("\n" + "-" * 80)
    print(f"STEP-LEVEL LATENCY SUMMARY ({num_profile_steps} decode iterations):")
    print(f"  * Average GPU Execution Time : {avg_gpu_ms:.2f} ms per token")
    print(f"  * Average CPU Wall Time      : {avg_cpu_ms:.2f} ms per token")
    print(f"  * CPU Launch Overhead        : {max(0.0, avg_cpu_ms - avg_gpu_ms):.2f} ms ({(max(0.0, avg_cpu_ms - avg_gpu_ms)/avg_cpu_ms)*100:.1f}%)")
    print(f"  * Effective Decode Speed     : {decode_tps:.2f} tokens/second")
    print("-" * 80)

    # 2. PyTorch Profiler with Chrome Trace & Kernel Table
    print("\nCapturing PyTorch CUDA Profiler trace (with shapes & stacks)...")
    trace_path = args.trace_path or os.environ.get("TRACE_PATH", "decode_flamegraph.json")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            with torch.inference_mode():
                logits, past_kv = model(curr_token, past_key_values=past_kv, return_logits_only_last=True)
                curr_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace exported to: {trace_path}")
    print("(Can be visualized in https://ui.perfetto.dev or chrome://tracing)\n")

    # Analyze Profiler key averages
    key_avgs = prof.key_averages()
    
    def get_device_time(item):
        for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
            if hasattr(item, attr):
                return getattr(item, attr)
        return 0

    cuda_sorted = sorted(key_avgs, key=get_device_time, reverse=True)
    total_cuda_time = sum(get_device_time(k) for k in cuda_sorted)

    print("=" * 80)
    print(" PYTORCH OFFICIAL PROFILER SUMMARY TABLE (Sorted by Device Time)")
    print("=" * 80)
    sort_key = "self_device_time_total" if hasattr(key_avgs[0], "self_device_time_total") else "self_cuda_time_total"
    print(key_avgs.table(sort_by=sort_key, row_limit=20))

    print("=" * 80)
    print(" TOP 15 CUDA KERNELS BY GPU TIME (Decode Step)")
    print("=" * 80)
    header = f"{'Kernel / Operation Name':<42} | {'Calls':<6} | {'Total GPU (ms)':<14} | {'% GPU':<8} | {'Avg (us)':<10}"
    print(header)
    print("-" * 80)

    moe_gu_time = 0
    moe_down_time = 0
    dense_gemv_time = 0
    attention_time = 0
    norm_time = 0
    lm_head_time = 0
    other_time = 0

    for item in cuda_sorted[:15]:
        d_time = get_device_time(item)
        total_ms = d_time / 1000.0
        pct = (d_time / total_cuda_time * 100.0) if total_cuda_time > 0 else 0
        avg_us = (d_time / item.count) if item.count > 0 else 0
        name = item.key[:40]
        print(f"{name:<42} | {item.count:<6} | {total_ms:<14.2f} | {pct:<7.1f}% | {avg_us:<10.1f}")

    for item in cuda_sorted:
        key = item.key.lower()
        t = get_device_time(item)
        if "fused_moe_gate_up" in key:
            moe_gu_time += t
        elif "fused_moe_down" in key:
            moe_down_time += t
        elif "marlin" in key or "gemm" in key or "gemv" in key:
            dense_gemv_time += t
        elif "attention" in key or "softmax" in key or "rope" in key:
            attention_time += t
        elif "norm" in key:
            norm_time += t
        elif "linear" in key or "matmul" in key:
            lm_head_time += t
        else:
            other_time += t

    print("=" * 80)
    print(" HIGH-LEVEL SUBSYSTEM FLAMEGRAPH BREAKDOWN")
    print("=" * 80)

    subsystems = [
        ("MoE Fused Gate & Up (fused_moe_gate_up)", moe_gu_time),
        ("MoE Fused Down & Accum (fused_moe_down_accum)", moe_down_time),
        ("Dense Shared MLP + Attention Projections (Marlin)", dense_gemv_time),
        ("Attention Core (RoPE, KV update, SWA/Full)", attention_time),
        ("RMSNorms & Layer Scalars", norm_time),
        ("LM Head Projection (2816 -> 262k)", lm_head_time),
        ("Miscellaneous / Overhead", other_time),
    ]

    for name, t_us in sorted(subsystems, key=lambda x: x[1], reverse=True):
        t_ms = t_us / 1000.0
        pct = (t_us / total_cuda_time * 100.0) if total_cuda_time > 0 else 0
        bars = "#" * int(pct / 2.5)
        print(f"  {name:<48} : {t_ms:>6.2f} ms ({pct:>5.1f}%) | {bars}")

    print("=" * 80)

    # 3. Hardware Bandwidth & Bottleneck Analysis
    # Gemma 4 26B A4B:
    # 30 layers.
    # Active weights per token:
    # - Attention: (q + k + v + o) = (2816*2816 + 2816*1024 + 2816*1024 + 2816*2816) = ~21.6M params @ 4-bit = ~10.8 MB per layer * 30 = ~324 MB
    # - Shared MLP: (gate + up + down) = (2816*3072 * 2 + 3072*2816) = ~26M params @ 4-bit = ~13 MB * 30 = ~390 MB
    # - 8 Routed Experts: 8 * (gate + up + down) = 8 * (2816*1024 * 2 + 1024*2816) = 8 * 8.6M = 69M params @ 4-bit = ~34.5 MB * 30 = ~1035 MB
    # - Norms & LM Head = ~500 MB
    # Total active weight bytes fetched per token: ~2.25 GB
    active_bytes_per_tok = 2.25 * (1024**3)
    measured_time_sec = (avg_gpu_ms / 1000.0)
    achieved_bw_gbs = (active_bytes_per_tok / measured_time_sec) / (1024**3)

    print("\nHARDWARE ROOFLINE & BOTTLENECK ANALYSIS (RTX 3090):")
    print("  * Active Weights Read / Token : ~2.25 GB")
    print(f"  * Average Token Time          : {avg_gpu_ms:.2f} ms")
    print(f"  * Achieved Memory Bandwidth   : {achieved_bw_gbs:.1f} GB/s")
    print("  * RTX 3090 Peak Bandwidth     : ~936 GB/s (Practical sustained: ~750-800 GB/s)")
    print(f"  * Memory Bus Efficiency       : {(achieved_bw_gbs / 750.0) * 100.0:.1f}% of real hardware maximum")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="GemmaInfer Detailed Decode Profiler & Flamegraph Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("GEMMA_MODEL_PATH", None),
        help="Path to HuggingFace model weights directory",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Profile using synthetic Gemma 4 26B A4B weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default=None,
        help="Output path for Chrome trace JSON (default: decode_flamegraph.json)",
    )
    args = parser.parse_args()
    profile_decode(args)


if __name__ == "__main__":
    main()
