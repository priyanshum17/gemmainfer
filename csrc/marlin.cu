#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename scalar_t>
__global__ void marlin_w4a16_coalesced_kernel(
    const scalar_t* __restrict__ A,          // [M, K]
    const int32_t* __restrict__ B_packed,    // [K / 8, N]
    const scalar_t* __restrict__ scales,      // [K / group_size, N]
    scalar_t* __restrict__ C,                 // [M, N]
    int M,
    int K,
    int N,
    int group_size
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;

    if (col >= N || m >= M) return;

    const scalar_t* A_m = A + m * K;
    float acc = 0.0f;
    int packed_K = K / 8;

    for (int pk = 0; pk < packed_K; ++pk) {
        int32_t packed_val = B_packed[pk * N + col];
        int k_base = pk * 8;
        int group_idx = k_base / group_size;
        float scale = static_cast<float>(scales[group_idx * N + col]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int k = k_base + i;
            int32_t w_raw = (packed_val >> (i * 4)) & 0x0F;
            float w = static_cast<float>(w_raw - 8);
            float a = static_cast<float>(A_m[k]);
            acc += a * (w * scale);
        }
    }

    C[m * N + col] = static_cast<scalar_t>(acc);
}

void launch_marlin_w4a16(
    const torch::Tensor& A,
    const torch::Tensor& B_packed,
    const torch::Tensor& scales,
    torch::Tensor& C,
    int group_size
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B_packed.size(1);

    int threads_x = 128;
    int blocks_x = (N + threads_x - 1) / threads_x;
    dim3 grid(blocks_x, M);
    dim3 block(threads_x, 1);

    if (A.scalar_type() == torch::kFloat16) {
        marlin_w4a16_coalesced_kernel<half><<<grid, block>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, K, N, group_size
        );
    } else if (A.scalar_type() == torch::kBFloat16) {
        marlin_w4a16_coalesced_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            B_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
            M, K, N, group_size
        );
    } else {
        marlin_w4a16_coalesced_kernel<float><<<grid, block>>>(
            A.data_ptr<float>(),
            B_packed.data_ptr<int32_t>(),
            scales.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N, group_size
        );
    }
}
