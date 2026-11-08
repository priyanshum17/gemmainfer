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

__device__ __forceinline__ float gelu_tanh(float x) {
    float x_cube = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template <typename scalar_t>
__global__ void fused_moe_gate_up_kernel(
    const scalar_t* __restrict__ A,               // [1, K]
    const int32_t* __restrict__ B_all_packed,     // [num_experts, K / 8, 2 * I]
    const scalar_t* __restrict__ scales_all,      // [num_experts, K / group_size, 2 * I]
    const int32_t* __restrict__ expert_indices,   // [top_k]
    scalar_t* __restrict__ out_acts,              // [top_k, I]
    int K,
    int I,
    int group_size,
    int num_experts,
    int top_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y;

    if (col >= I || k_idx >= top_k) return;

    int expert_id = expert_indices[k_idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    int packed_K = K / 8;
    int stride_2I = 2 * I;
    int group_stride = K / group_size;

    const int32_t* B_expert = B_all_packed + static_cast<size_t>(expert_id) * packed_K * stride_2I;
    const scalar_t* S_expert = scales_all + static_cast<size_t>(expert_id) * group_stride * stride_2I;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;

    for (int pk = 0; pk < packed_K; ++pk) {
        int k_base = pk * 8;
        int group_idx = k_base / group_size;

        int32_t packed_gate = B_expert[pk * stride_2I + col];
        int32_t packed_up = B_expert[pk * stride_2I + (col + I)];

        float scale_gate = static_cast<float>(S_expert[group_idx * stride_2I + col]);
        float scale_up = static_cast<float>(S_expert[group_idx * stride_2I + (col + I)]);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float a = static_cast<float>(A[k_base + i]);

            int32_t w_raw_gate = (packed_gate >> (i * 4)) & 0x0F;
            acc_gate += a * (static_cast<float>(w_raw_gate - 8) * scale_gate);

            int32_t w_raw_up = (packed_up >> (i * 4)) & 0x0F;
            acc_up += a * (static_cast<float>(w_raw_up - 8) * scale_up);
        }
    }

    float act = gelu_tanh(acc_gate) * acc_up;
    out_acts[k_idx * I + col] = static_cast<scalar_t>(act);
}

template <typename scalar_t>
__global__ void fused_moe_down_accum_kernel(
    const scalar_t* __restrict__ out_acts,        // [top_k, I]
    const int32_t* __restrict__ down_all_packed,  // [num_experts, I / 8, K]
    const scalar_t* __restrict__ down_scales_all, // [num_experts, I / group_size, K]
    const int32_t* __restrict__ expert_indices,   // [top_k]
    const scalar_t* __restrict__ expert_weights,  // [top_k]
    scalar_t* __restrict__ out,                   // [1, K]
    int I,
    int K,
    int group_size,
    int num_experts,
    int top_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= K) return;

    int packed_I = I / 8;
    int group_stride = I / group_size;

    float total_acc = 0.0f;

    for (int k_idx = 0; k_idx < top_k; ++k_idx) {
        int expert_id = expert_indices[k_idx];
        if (expert_id < 0 || expert_id >= num_experts) continue;

        float weight = static_cast<float>(expert_weights[k_idx]);
        const scalar_t* act_k = out_acts + k_idx * I;

        const int32_t* B_expert = down_all_packed + static_cast<size_t>(expert_id) * packed_I * K;
        const scalar_t* S_expert = down_scales_all + static_cast<size_t>(expert_id) * group_stride * K;

        float expert_acc = 0.0f;

        for (int pk = 0; pk < packed_I; ++pk) {
            int i_base = pk * 8;
            int group_idx = i_base / group_size;

            int32_t packed_val = B_expert[pk * K + col];
            float scale = static_cast<float>(S_expert[group_idx * K + col]);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float a = static_cast<float>(act_k[i_base + i]);
                int32_t w_raw = (packed_val >> (i * 4)) & 0x0F;
                expert_acc += a * (static_cast<float>(w_raw - 8) * scale);
            }
        }

        total_acc += expert_acc * weight;
    }

    out[col] += static_cast<scalar_t>(total_acc);
}

void launch_fused_batched_moe(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_packed,
    const torch::Tensor& gate_up_scales,
    const torch::Tensor& down_packed,
    const torch::Tensor& down_scales,
    const torch::Tensor& expert_indices,
    const torch::Tensor& expert_weights,
    torch::Tensor& out,
    int group_size
) {
    int num_experts = gate_up_packed.size(0);
    int top_k = expert_indices.size(0);
    int K = x.size(1);
    int two_I = gate_up_packed.size(2);
    int I = two_I / 2;

    auto indices_i32 = expert_indices.to(torch::kInt32).contiguous();
    auto weights_cast = expert_weights.to(x.dtype()).contiguous();
    auto out_acts = torch::empty({top_k, I}, x.options());

    int threads = 128;
    dim3 grid_gu((I + threads - 1) / threads, top_k);
    dim3 grid_down((K + threads - 1) / threads, 1);

    if (x.scalar_type() == torch::kBFloat16) {
        fused_moe_gate_up_kernel<__nv_bfloat16><<<grid_gu, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            gate_up_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(gate_up_scales.data_ptr<at::BFloat16>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out_acts.data_ptr<at::BFloat16>()),
            K, I, group_size, num_experts, top_k
        );

        fused_moe_down_accum_kernel<__nv_bfloat16><<<grid_down, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(out_acts.data_ptr<at::BFloat16>()),
            down_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scales.data_ptr<at::BFloat16>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(weights_cast.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            I, K, group_size, num_experts, top_k
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        fused_moe_gate_up_kernel<half><<<grid_gu, threads>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            gate_up_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(gate_up_scales.data_ptr<at::Half>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<half*>(out_acts.data_ptr<at::Half>()),
            K, I, group_size, num_experts, top_k
        );

        fused_moe_down_accum_kernel<half><<<grid_down, threads>>>(
            reinterpret_cast<const half*>(out_acts.data_ptr<at::Half>()),
            down_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(down_scales.data_ptr<at::Half>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(weights_cast.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            I, K, group_size, num_experts, top_k
        );
    } else {
        fused_moe_gate_up_kernel<float><<<grid_gu, threads>>>(
            x.data_ptr<float>(),
            gate_up_packed.data_ptr<int32_t>(),
            gate_up_scales.data_ptr<float>(),
            indices_i32.data_ptr<int32_t>(),
            out_acts.data_ptr<float>(),
            K, I, group_size, num_experts, top_k
        );

        fused_moe_down_accum_kernel<float><<<grid_down, threads>>>(
            out_acts.data_ptr<float>(),
            down_packed.data_ptr<int32_t>(),
            down_scales.data_ptr<float>(),
            indices_i32.data_ptr<int32_t>(),
            weights_cast.data_ptr<float>(),
            out.data_ptr<float>(),
            I, K, group_size, num_experts, top_k
        );
    }
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
