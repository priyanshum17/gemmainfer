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
        # Test standard RMSNorm
        gemmainfer_csrc.rmsnorm(x, w, out, 1e-6, False)
        # Test unit-offset RMSNorm
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
    group_size = 128

    # Test both M = 1 (decode) and M = 4 (prefill / speculative)
    for M in [1, 4]:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B_packed = torch.randint(-100000, 100000, (K // 8, N), device="cuda", dtype=torch.int32)
        scales = torch.randn(K // group_size, N, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        gemmainfer_csrc.marlin_w4a16(A, B_packed, scales, C, group_size)
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
    print("All kernel sanitizer passes executed cleanly.")


if __name__ == "__main__":
    main()
