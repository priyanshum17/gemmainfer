"""Micro-kernel test harness for NVIDIA compute-sanitizer.

Runs synthetic boundary checks directly on individual CUDA kernels without downloading
or deploying any model weights. Takes < 2 seconds to execute.

Usage on an NVIDIA GPU machine:
  compute-sanitizer --tool memcheck python tests/test_cuda_kernels_sanitizer.py
  compute-sanitizer --tool racecheck python tests/test_cuda_kernels_sanitizer.py
"""

import sys
import torch

try:
    import gemmainfer_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False


def test_rmsnorm_kernel_sanitizer():
    if not torch.cuda.is_available() or not HAS_CSRC:
        return
    dim = 2816
    for bsz in [1, 2, 8]:
        x = torch.randn(bsz, dim, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        gemmainfer_csrc.rmsnorm(x, w, out, 1e-6, False)
        gemmainfer_csrc.rmsnorm(x, w, out, 1e-6, True)
    torch.cuda.synchronize()


def test_softcapping_kernel_sanitizer():
    if not torch.cuda.is_available() or not HAS_CSRC:
        return
    for num_elements in [1, 32, 262144]:
        logits = torch.randn(1, num_elements, device="cuda", dtype=torch.bfloat16)
        gemmainfer_csrc.softcapping(logits, 30.0)
    torch.cuda.synchronize()


def test_marlin_kernel_sanitizer():
    if not torch.cuda.is_available() or not HAS_CSRC:
        return
    K = 2816
    N = 2112
    group_size = 64

    # Test M = 1 (decode), M = 5 (speculative verification), and M = 16 (prefill)
    for M in [1, 5, 16]:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B_packed = torch.randint(-100000, 100000, (K // 8, N), device="cuda", dtype=torch.int32)
        scales = torch.randn(K // group_size, N, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        gemmainfer_csrc.marlin_w4a16(A, B_packed, scales, C, group_size)
    torch.cuda.synchronize()


def test_token_union_moe_kernel_sanitizer():
    if not torch.cuda.is_available() or not HAS_CSRC:
        return
    K = 2816
    inter_dim = 704
    num_experts = 128
    top_k = 8
    group_size = 64

    gate_up_packed = torch.randint(-100000, 100000, (num_experts, K // 8, 2 * inter_dim), device="cuda", dtype=torch.int32)
    gate_up_scales = torch.randn(num_experts, K // group_size, 2 * inter_dim, device="cuda", dtype=torch.bfloat16)
    down_packed = torch.randint(-100000, 100000, (num_experts, inter_dim // 8, K), device="cuda", dtype=torch.int32)
    down_scales = torch.randn(num_experts, inter_dim // group_size, K, device="cuda", dtype=torch.bfloat16)

    for M in [1, 5]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # Create overlapping expert indices across tokens to stress Token-Union deduplication
        indices = torch.randint(0, 16, (M, top_k), device="cuda", dtype=torch.int32)
        weights = torch.softmax(torch.randn(M, top_k, device="cuda", dtype=torch.float32), dim=-1).to(torch.bfloat16)
        out = torch.zeros_like(x)
        gemmainfer_csrc.fused_batched_moe(
            x, gate_up_packed, gate_up_scales, down_packed, down_scales, indices, weights, out, group_size
        )
    torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        print("Note: CUDA not detected on local host.")
        print("Run this script on an NVIDIA GPU machine with compute-sanitizer:")
        print("  compute-sanitizer --tool memcheck python tests/test_cuda_kernels_sanitizer.py")
        sys.exit(0)

    print("Running micro-kernel sanitizer tests on CUDA...")
    test_rmsnorm_kernel_sanitizer()
    test_softcapping_kernel_sanitizer()
    test_marlin_kernel_sanitizer()
    test_token_union_moe_kernel_sanitizer()
    print("All kernel sanitizer passes executed cleanly.")


if __name__ == "__main__":
    main()
