#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    float x_cube = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Multi-row Split-K W4A16 kernel for M in [1, 8] (decode and speculative verification).
// Reads packed 4-bit weights and scales from HBM ONCE across all M tokens.
template <typename scalar_t, int MAX_M = 8>
__global__ void marlin_w4a16_splitk_batched_kernel(
    const scalar_t* __restrict__ A,          // [M, K]
    const int32_t* __restrict__ B_packed,    // [K / 8, N]
    const scalar_t* __restrict__ scales,      // [K / group_size, N]
    scalar_t* __restrict__ C,                 // [M, N]
    int M,
    int K,
    int N,
    int group_size,
    int split_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k_slice = blockIdx.y;

    if (k_slice >= split_k) return;

    int num_groups = K / group_size;
    int groups_per_slice = (num_groups + split_k - 1) / split_k;
    int g_start = k_slice * groups_per_slice;
    int g_end = min(g_start + groups_per_slice, num_groups);
    if (g_start >= g_end) return;

    int k_start = g_start * group_size;
    int k_slice_len = (g_end - g_start) * group_size;

    extern __shared__ char smem_splitk[];
    scalar_t* s_A = reinterpret_cast<scalar_t*>(smem_splitk); // [M, k_slice_len]

    int total_smem_elems = M * k_slice_len;
    for (int idx = threadIdx.x; idx < total_smem_elems; idx += blockDim.x) {
        int m_idx = idx / k_slice_len;
        int k_local = idx % k_slice_len;
        s_A[idx] = A[m_idx * K + k_start + k_local];
    }
    __syncthreads();

    if (col >= N) return;

    int words_per_group = group_size / 8;
    float slice_acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        slice_acc[m] = 0.0f;
    }

    for (int g = g_start; g < g_end; ++g) {
        float scale = static_cast<float>(scales[g * N + col]);
        int pk_base = g * words_per_group;
        int k_local_base = (g - g_start) * group_size;

        float group_acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            group_acc[m] = 0.0f;
        }

        #pragma unroll
        for (int w = 0; w < words_per_group; ++w) {
            int pk = pk_base + w;
            int32_t packed_val = B_packed[pk * N + col];
            int local_k = k_local_base + w * 8;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int k = local_k + i;
                int32_t w_raw = (packed_val >> (i * 4)) & 0x0F;
                float weight_val = static_cast<float>(w_raw - 8);

                for (int m = 0; m < M; ++m) {
                    float a = static_cast<float>(s_A[m * k_slice_len + k]);
                    group_acc[m] += a * weight_val;
                }
            }
        }

        for (int m = 0; m < M; ++m) {
            slice_acc[m] += group_acc[m] * scale;
        }
    }

    for (int m = 0; m < M; ++m) {
        atomicAdd(&C[m * N + col], static_cast<scalar_t>(slice_acc[m]));
    }
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

    if (m >= M) return;

    extern __shared__ char smem_coal[];
    scalar_t* s_A = reinterpret_cast<scalar_t*>(smem_coal);
    const scalar_t* A_m = A + m * K;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        s_A[idx] = A_m[idx];
    }
    __syncthreads();

    if (col >= N) return;

    float acc = 0.0f;
    int num_groups = K / group_size;
    int words_per_group = group_size / 8;

    for (int g = 0; g < num_groups; ++g) {
        float scale = static_cast<float>(scales[g * N + col]);
        int pk_base = g * words_per_group;
        float group_acc = 0.0f;

        #pragma unroll
        for (int w = 0; w < words_per_group; ++w) {
            int pk = pk_base + w;
            int32_t packed_val = B_packed[pk * N + col];
            int k_base = pk * 8;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int k = k_base + i;
                int32_t w_raw = (packed_val >> (i * 4)) & 0x0F;
                float w = static_cast<float>(w_raw - 8);
                float a = static_cast<float>(s_A[k]);
                group_acc += a * w;
            }
        }
        acc += group_acc * scale;
    }

    C[m * N + col] = static_cast<scalar_t>(acc);
}

// Token-Union Coalesced Multi-Token MoE Gate+Up Kernel.
// Deduplicates identical expert IDs across all M * top_k routed slots on-the-fly in shared memory.
// Fetches each unique expert's W4A16 weights from HBM strictly ONCE and applies them to all
// matching tokens simultaneously in registers.
template <typename scalar_t, int MAX_MATCH = 8, int MAX_SLOTS = 128>
__global__ void fused_moe_token_union_gate_up_kernel(
    const scalar_t* __restrict__ A,               // [M, K]
    const int32_t* __restrict__ B_all_packed,     // [num_experts, K / 8, 2 * I]
    const scalar_t* __restrict__ scales_all,      // [num_experts, K / group_size, 2 * I]
    const int32_t* __restrict__ expert_indices,   // [M * top_k]
    scalar_t* __restrict__ out_acts,              // [M * top_k, I]
    int M,
    int K,
    int I,
    int group_size,
    int num_experts,
    int top_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int u_idx = blockIdx.y;
    int total_slots = M * top_k;

    __shared__ int s_expert_id;
    __shared__ int s_match_count;
    __shared__ int s_match_tok[MAX_MATCH];
    __shared__ int s_match_slot[MAX_MATCH];

    if (threadIdx.x == 0) {
        int unique_seen = 0;
        int target_expert = -1;
        int limit = min(total_slots, MAX_SLOTS);

        for (int s = 0; s < limit; ++s) {
            int e = expert_indices[s];
            if (e < 0 || e >= num_experts) continue;
            bool already = false;
            for (int p = 0; p < s; ++p) {
                if (expert_indices[p] == e) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                if (unique_seen == u_idx) {
                    target_expert = e;
                    break;
                }
                unique_seen++;
            }
        }

        s_expert_id = target_expert;
        int m_cnt = 0;
        if (target_expert >= 0) {
            for (int s = 0; s < limit; ++s) {
                if (expert_indices[s] == target_expert && m_cnt < MAX_MATCH) {
                    s_match_tok[m_cnt] = s / top_k;
                    s_match_slot[m_cnt] = s;
                    m_cnt++;
                }
            }
        }
        s_match_count = m_cnt;
    }
    __syncthreads();

    int expert_id = s_expert_id;
    int match_count = s_match_count;
    if (expert_id < 0 || match_count == 0) return;

    extern __shared__ char smem_gu[];
    scalar_t* s_A = reinterpret_cast<scalar_t*>(smem_gu); // [match_count, K]

    int total_a_elems = match_count * K;
    for (int idx = threadIdx.x; idx < total_a_elems; idx += blockDim.x) {
        int m = idx / K;
        int k_off = idx % K;
        int tok_idx = s_match_tok[m];
        s_A[idx] = A[tok_idx * K + k_off];
    }
    __syncthreads();

    if (col >= I) return;

    int packed_K = K / 8;
    int stride_2I = 2 * I;
    int num_groups = K / group_size;
    int words_per_group = group_size / 8;

    const int32_t* B_expert = B_all_packed + static_cast<size_t>(expert_id) * packed_K * stride_2I;
    const scalar_t* S_expert = scales_all + static_cast<size_t>(expert_id) * num_groups * stride_2I;

    float acc_gate[MAX_MATCH];
    float acc_up[MAX_MATCH];
    #pragma unroll
    for (int m = 0; m < MAX_MATCH; ++m) {
        acc_gate[m] = 0.0f;
        acc_up[m] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        float scale_gate = static_cast<float>(S_expert[g * stride_2I + col]);
        float scale_up = static_cast<float>(S_expert[g * stride_2I + (col + I)]);
        int pk_base = g * words_per_group;
        int k_base = g * group_size;

        float group_gate[MAX_MATCH];
        float group_up[MAX_MATCH];
        #pragma unroll
        for (int m = 0; m < MAX_MATCH; ++m) {
            group_gate[m] = 0.0f;
            group_up[m] = 0.0f;
        }

        #pragma unroll
        for (int w = 0; w < words_per_group; ++w) {
            int pk = pk_base + w;
            int32_t packed_gate = B_expert[pk * stride_2I + col];
            int32_t packed_up = B_expert[pk * stride_2I + (col + I)];
            int local_k = k_base + w * 8;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int k_idx = local_k + i;
                float w_gate = static_cast<float>(((packed_gate >> (i * 4)) & 0x0F) - 8);
                float w_up = static_cast<float>(((packed_up >> (i * 4)) & 0x0F) - 8);

                for (int m = 0; m < match_count; ++m) {
                    float a = static_cast<float>(s_A[m * K + k_idx]);
                    group_gate[m] += a * w_gate;
                    group_up[m] += a * w_up;
                }
            }
        }

        for (int m = 0; m < match_count; ++m) {
            acc_gate[m] += group_gate[m] * scale_gate;
            acc_up[m] += group_up[m] * scale_up;
        }
    }

    for (int m = 0; m < match_count; ++m) {
        float act = gelu_tanh(acc_gate[m]) * acc_up[m];
        int slot_idx = s_match_slot[m];
        out_acts[slot_idx * I + col] = static_cast<scalar_t>(act);
    }
}

// Token-Union Coalesced Multi-Token MoE Down+Accumulate Kernel.
// Reads each unique expert's down_proj weights from HBM strictly ONCE across all matching draft tokens.
template <typename scalar_t, int MAX_MATCH = 8, int MAX_SLOTS = 128>
__global__ void fused_moe_token_union_down_accum_kernel(
    const scalar_t* __restrict__ out_acts,        // [M * top_k, I]
    const int32_t* __restrict__ down_all_packed,  // [num_experts, I / 8, K]
    const scalar_t* __restrict__ down_scales_all, // [num_experts, I / group_size, K]
    const int32_t* __restrict__ expert_indices,   // [M * top_k]
    const scalar_t* __restrict__ expert_weights,  // [M * top_k]
    scalar_t* __restrict__ out,                   // [M, K]
    int M,
    int I,
    int K,
    int group_size,
    int num_experts,
    int top_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int u_idx = blockIdx.y;
    int total_slots = M * top_k;

    __shared__ int s_expert_id;
    __shared__ int s_match_count;
    __shared__ int s_match_tok[MAX_MATCH];
    __shared__ int s_match_slot[MAX_MATCH];
    __shared__ float s_match_weight[MAX_MATCH];

    if (threadIdx.x == 0) {
        int unique_seen = 0;
        int target_expert = -1;
        int limit = min(total_slots, MAX_SLOTS);

        for (int s = 0; s < limit; ++s) {
            int e = expert_indices[s];
            if (e < 0 || e >= num_experts) continue;
            bool already = false;
            for (int p = 0; p < s; ++p) {
                if (expert_indices[p] == e) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                if (unique_seen == u_idx) {
                    target_expert = e;
                    break;
                }
                unique_seen++;
            }
        }

        s_expert_id = target_expert;
        int m_cnt = 0;
        if (target_expert >= 0) {
            for (int s = 0; s < limit; ++s) {
                if (expert_indices[s] == target_expert && m_cnt < MAX_MATCH) {
                    s_match_tok[m_cnt] = s / top_k;
                    s_match_slot[m_cnt] = s;
                    s_match_weight[m_cnt] = static_cast<float>(expert_weights[s]);
                    m_cnt++;
                }
            }
        }
        s_match_count = m_cnt;
    }
    __syncthreads();

    int expert_id = s_expert_id;
    int match_count = s_match_count;
    if (expert_id < 0 || match_count == 0) return;

    extern __shared__ char smem_down[];
    scalar_t* s_act = reinterpret_cast<scalar_t*>(smem_down); // [match_count, I]

    int total_act_elems = match_count * I;
    for (int idx = threadIdx.x; idx < total_act_elems; idx += blockDim.x) {
        int m = idx / I;
        int i_off = idx % I;
        int slot_idx = s_match_slot[m];
        s_act[idx] = out_acts[slot_idx * I + i_off];
    }
    __syncthreads();

    if (col >= K) return;

    int packed_I = I / 8;
    int num_groups = I / group_size;
    int words_per_group = group_size / 8;

    const int32_t* B_expert = down_all_packed + static_cast<size_t>(expert_id) * packed_I * K;
    const scalar_t* S_expert = down_scales_all + static_cast<size_t>(expert_id) * num_groups * K;

    float expert_acc[MAX_MATCH];
    #pragma unroll
    for (int m = 0; m < MAX_MATCH; ++m) {
        expert_acc[m] = 0.0f;
    }

    for (int g = 0; g < num_groups; ++g) {
        float scale = static_cast<float>(S_expert[g * K + col]);
        int pk_base = g * words_per_group;
        int i_local_base = g * group_size;

        float group_acc[MAX_MATCH];
        #pragma unroll
        for (int m = 0; m < MAX_MATCH; ++m) {
            group_acc[m] = 0.0f;
        }

        #pragma unroll
        for (int w = 0; w < words_per_group; ++w) {
            int pk = pk_base + w;
            int32_t packed_val = B_expert[pk * K + col];
            int local_i = i_local_base + w * 8;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int i_idx = local_i + i;
                float w_val = static_cast<float>(((packed_val >> (i * 4)) & 0x0F) - 8);

                for (int m = 0; m < match_count; ++m) {
                    float a = static_cast<float>(s_act[m * I + i_idx]);
                    group_acc[m] += a * w_val;
                }
            }
        }

        for (int m = 0; m < match_count; ++m) {
            expert_acc[m] += group_acc[m] * scale;
        }
    }

    for (int m = 0; m < match_count; ++m) {
        int tok_idx = s_match_tok[m];
        float weight = s_match_weight[m];
        atomicAdd(&out[tok_idx * K + col], static_cast<scalar_t>(expert_acc[m] * weight));
    }
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
    int M = x.size(0);
    int K = x.size(1);
    int num_experts = gate_up_packed.size(0);
    int two_I = gate_up_packed.size(2);
    int I = two_I / 2;
    int total_slots = expert_indices.numel();
    int top_k = total_slots / M;

    auto indices_i32 = (expert_indices.scalar_type() == torch::kInt32 && expert_indices.is_contiguous())
        ? expert_indices
        : expert_indices.to(torch::kInt32).contiguous();
    auto weights_cast = (expert_weights.scalar_type() == x.scalar_type() && expert_weights.is_contiguous())
        ? expert_weights
        : expert_weights.to(x.dtype()).contiguous();

    auto out_acts = torch::empty({total_slots, I}, x.options());

    int threads = 128;
    dim3 grid_gu((I + threads - 1) / threads, total_slots);
    dim3 grid_down((K + threads - 1) / threads, total_slots);

    int max_match = min(M, 8);
    size_t smem_gu = max_match * K * x.element_size();
    size_t smem_down = max_match * I * x.element_size();

    if (x.scalar_type() == torch::kBFloat16) {
        fused_moe_token_union_gate_up_kernel<__nv_bfloat16><<<grid_gu, threads, smem_gu>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            gate_up_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(gate_up_scales.data_ptr<at::BFloat16>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out_acts.data_ptr<at::BFloat16>()),
            M, K, I, group_size, num_experts, top_k
        );

        fused_moe_token_union_down_accum_kernel<__nv_bfloat16><<<grid_down, threads, smem_down>>>(
            reinterpret_cast<const __nv_bfloat16*>(out_acts.data_ptr<at::BFloat16>()),
            down_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scales.data_ptr<at::BFloat16>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(weights_cast.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M, I, K, group_size, num_experts, top_k
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        fused_moe_token_union_gate_up_kernel<half><<<grid_gu, threads, smem_gu>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            gate_up_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(gate_up_scales.data_ptr<at::Half>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<half*>(out_acts.data_ptr<at::Half>()),
            M, K, I, group_size, num_experts, top_k
        );

        fused_moe_token_union_down_accum_kernel<half><<<grid_down, threads, smem_down>>>(
            reinterpret_cast<const half*>(out_acts.data_ptr<at::Half>()),
            down_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(down_scales.data_ptr<at::Half>()),
            indices_i32.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(weights_cast.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            M, I, K, group_size, num_experts, top_k
        );
    } else {
        fused_moe_token_union_gate_up_kernel<float><<<grid_gu, threads, smem_gu>>>(
            x.data_ptr<float>(),
            gate_up_packed.data_ptr<int32_t>(),
            gate_up_scales.data_ptr<float>(),
            indices_i32.data_ptr<int32_t>(),
            out_acts.data_ptr<float>(),
            M, K, I, group_size, num_experts, top_k
        );

        fused_moe_token_union_down_accum_kernel<float><<<grid_down, threads, smem_down>>>(
            out_acts.data_ptr<float>(),
            down_packed.data_ptr<int32_t>(),
            down_scales.data_ptr<float>(),
            indices_i32.data_ptr<int32_t>(),
            weights_cast.data_ptr<float>(),
            out.data_ptr<float>(),
            M, I, K, group_size, num_experts, top_k
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

    if (M <= 8) {
        int split_k = 8;
        dim3 grid(blocks_x, split_k);
        dim3 block(threads_x, 1);
        C.zero_();
        int num_groups = K / group_size;
        int groups_per_slice = (num_groups + split_k - 1) / split_k;
        int k_slice_len = groups_per_slice * group_size;
        size_t smem_splitk = M * k_slice_len * A.element_size();

        if (A.scalar_type() == torch::kBFloat16) {
            marlin_w4a16_splitk_batched_kernel<__nv_bfloat16><<<grid, block, smem_splitk>>>(
                reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
                B_packed.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
                M, K, N, group_size, split_k
            );
        } else if (A.scalar_type() == torch::kFloat16) {
            marlin_w4a16_splitk_batched_kernel<half><<<grid, block, smem_splitk>>>(
                reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                B_packed.data_ptr<int32_t>(),
                reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
                reinterpret_cast<half*>(C.data_ptr<at::Half>()),
                M, K, N, group_size, split_k
            );
        } else {
            marlin_w4a16_splitk_batched_kernel<float><<<grid, block, smem_splitk>>>(
                A.data_ptr<float>(),
                B_packed.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                C.data_ptr<float>(),
                M, K, N, group_size, split_k
            );
        }
        return;
    }

    dim3 grid(blocks_x, M);
    dim3 block(threads_x, 1);
    size_t smem_coal = K * A.element_size();

    if (A.scalar_type() == torch::kFloat16) {
        marlin_w4a16_coalesced_kernel<half><<<grid, block, smem_coal>>>(
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
            B_packed.data_ptr<int32_t>(),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, K, N, group_size
        );
    } else if (A.scalar_type() == torch::kBFloat16) {
        marlin_w4a16_coalesced_kernel<__nv_bfloat16><<<grid, block, smem_coal>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            B_packed.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
            M, K, N, group_size
        );
    } else {
        marlin_w4a16_coalesced_kernel<float><<<grid, block, smem_coal>>>(
            A.data_ptr<float>(),
            B_packed.data_ptr<int32_t>(),
            scales.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N, group_size
        );
    }
}
