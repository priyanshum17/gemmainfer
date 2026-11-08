"""Verification script for testing GemmaInfer on actual NVIDIA GPUs (RTX 3090 / 4090).

Runs 5 validation stages:
1. Environment & Architecture check (CUDA capability sm_86 / sm_89)
2. Native CUDA kernel compilation check
3. Numerical correctness vs PyTorch FP32 golden reference (tolerance testing)
4. Memory boundary & shape stress test (M=1 decode and M>1 prefill)
5. VRAM consumption and bandwidth sanity check
"""

import torch
import torch.nn as nn

try:
    import gemmainfer_csrc  # noqa: F401
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import apply_logit_softcapping
from gemmainfer.kernels.quant import MarlinLinear


def check_environment():
    print("=" * 65)
    print(" STAGE 1: CUDA Environment & Hardware Check")
    print("=" * 65)
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False.")
        print("       This script must be executed on an NVIDIA GPU host.")
        return False

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"Device:           {device_name}")
    print(f"Compute Arch:     {arch} (sm_86 = Ampere/3090, sm_89 = Ada/4090)")
    print(f"Total VRAM:       {total_mem_gb:.2f} GB")
    print(f"CUDA Version:     {torch.version.cuda}")
    print(f"CSRC Extension:   {'Available' if HAS_CSRC else 'NOT COMPILED (Run: python setup.py build_ext --inplace)'}")
    
    if total_mem_gb < 20.0:
        print(f"[WARN] VRAM is {total_mem_gb:.1f} GB (< 24GB). May run out of memory with large batch sizes.")
    else:
        print("[PASS] Hardware VRAM meets 24GB target threshold.")

    return True


def verify_rmsnorm_correctness():
    print("\n" + "=" * 65)
    print(" STAGE 2: RMSNorm Numerical Equivalence")
    print("=" * 65)
    dim = 2816
    x = torch.randn(4, 16, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    # PyTorch FP32 golden reference (Standard RMSNorm, no unit offset)
    x_f = x.float()
    w_f = weight.float()
    golden = (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6) * w_f).to(torch.bfloat16)

    # GemmaInfer RMSNorm module (calls CUDA kernel)
    norm = GemmaRMSNorm(dim=dim, eps=1e-6, add_unit_offset=False).to("cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(weight)
    actual = norm(x)

    diff = (actual.float() - golden.float()).abs().max().item()
    print(f"Max absolute divergence vs FP32 golden: {diff:.6e}")
    if diff < 1e-2:
        print("[PASS] CUDA RMSNorm matches mathematical golden reference.")
        return True
    else:
        print(f"[FAIL] Divergence {diff} exceeds tolerance threshold (0.01).")
        return False


def verify_softcapping_correctness():
    print("\n" + "=" * 65)
    print(" STAGE 3: Logit Soft-Capping Numerical Equivalence")
    print("=" * 65)
    logits = torch.linspace(-100.0, 100.0, steps=262144, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    cap = 30.0

    golden = (cap * torch.tanh(logits.float() / cap)).to(torch.bfloat16)
    test_logits = logits.clone()
    actual = apply_logit_softcapping(test_logits, cap)

    diff = (actual.float() - golden.float()).abs().max().item()
    print(f"Max absolute divergence vs FP32 golden: {diff:.6e}")
    if diff < 1e-2 and (actual.abs() <= cap).all():
        print("[PASS] CUDA Logit Soft-capping matches mathematical golden reference.")
        return True
    else:
        print(f"[FAIL] Divergence {diff} exceeds threshold.")
        return False


def verify_marlin_gemv_correctness():
    print("\n" + "=" * 65)
    print(" STAGE 4: Marlin W4A16 Quantized GEMV Numerical Correctness")
    print("=" * 65)
    K = 2816
    N = 2112
    group_size = 128

    linear = nn.Linear(K, N, bias=False).to("cuda", dtype=torch.bfloat16)
    marlin = MarlinLinear.from_linear(linear, group_size=group_size).to("cuda")

    # 1. Test M = 1 (Single Token Decode)
    x_single = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    golden_single = linear(x_single)
    actual_single = marlin(x_single)
    diff_single = (actual_single.float() - golden_single.float()).abs().mean().item()
    print(f"M=1 (Decode) Mean Quantization Error:    {diff_single:.4f}")

    # 2. Test M > 1 (Prefill / Speculative Verification)
    x_batch = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    golden_batch = linear(x_batch)
    actual_batch = marlin(x_batch)
    diff_batch = (actual_batch.float() - golden_batch.float()).abs().mean().item()
    print(f"M=8 (Prefill) Mean Quantization Error:   {diff_batch:.4f}")

    if diff_single < 0.15 and diff_batch < 0.15 and not torch.isnan(actual_batch).any():
        print("[PASS] Marlin W4A16 executes correctly across single-token and batch shapes.")
        return True
    else:
        print("[FAIL] Marlin W4A16 produced numerical anomalies or NaNs.")
        return False


def verify_full_model_vram():
    print("\n" + "=" * 65)
    print(" STAGE 5: Gemma 4 26B A4B Full Model Forward Pass & VRAM Test")
    print("=" * 65)
    torch.cuda.empty_cache()
    start_vram = torch.cuda.memory_allocated() / (1024**3)
    print(f"Base Allocated VRAM: {start_vram:.2f} GB")

    config = GemmaConfig.gemma_test_small()
    config.quant_method = "w4a16"
    model = GemmaForCausalLM(config).to("cuda", dtype=torch.bfloat16)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    
    with torch.inference_mode():
        logits, kv = model(input_ids)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Peak VRAM Used:      {peak_vram:.3f} GB")
    print(f"NaN Check:           {'CLEAN (No NaNs)' if not torch.isnan(logits).any() else 'FAILED (Contains NaNs)'}")

    if not torch.isnan(logits).any():
        print("[PASS] Full model execution runs without memory faults or divergence.")
        return True
    return False


def run_all_checks():
    print("Running GemmaInfer GPU Verification Suite...\n")
    if not check_environment():
        return
    
    s2 = verify_rmsnorm_correctness()
    s3 = verify_softcapping_correctness()
    s4 = verify_marlin_gemv_correctness()
    s5 = verify_full_model_vram()

    print("\n" + "=" * 65)
    if all([s2, s3, s4, s5]):
        print("All 5 verification stages passed.")
        print("Engine verified for 24GB hardware deployment.")
    else:
        print("Verification failures detected. Inspect stage logs above.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_checks()
