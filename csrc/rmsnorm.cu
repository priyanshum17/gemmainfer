#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void gemma_rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps,
    int hidden_size,
    bool add_unit_offset,
    bool has_weight
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    extern __shared__ float s_warp_sums[];

    const T* row_in = input + row * hidden_size;
    T* row_out = output + row * hidden_size;

    float thread_sq_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_in[i]);
        thread_sq_sum += val * val;
    }

    float warp_sum = warp_reduce_sum(thread_sq_sum);
    if (lane == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sq_sum = 0.0f;
    if (warp_id == 0) {
        float val = (tid < num_warps) ? s_warp_sums[tid] : 0.0f;
        block_sq_sum = warp_reduce_sum(val);
        if (lane == 0) {
            s_warp_sums[0] = block_sq_sum;
        }
    }
    __syncthreads();

    float rsqrt_val = rsqrtf((s_warp_sums[0] / static_cast<float>(hidden_size)) + eps);

    if (has_weight) {
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            float in_val = static_cast<float>(row_in[i]);
            float w_val = static_cast<float>(weight[i]);
            float scale = add_unit_offset ? (1.0f + w_val) : w_val;
            row_out[i] = static_cast<T>(in_val * rsqrt_val * scale);
        }
    } else {
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            float in_val = static_cast<float>(row_in[i]);
            row_out[i] = static_cast<T>(in_val * rsqrt_val);
        }
    }
}

void launch_gemma_rmsnorm(
    torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float eps,
    bool add_unit_offset
) {
    int num_rows = input.numel() / input.size(-1);
    int hidden_size = input.size(-1);
    bool has_weight = weight.defined() && weight.numel() > 0;

    int threads = (hidden_size > 4096) ? 1024 : ((hidden_size > 2048) ? 512 : 256);
    size_t shared_mem = (threads / 32) * sizeof(float);

    if (input.scalar_type() == torch::kFloat16) {
        const half* w_ptr = has_weight ? reinterpret_cast<const half*>(weight.data_ptr<at::Half>()) : nullptr;
        gemma_rmsnorm_kernel<half><<<num_rows, threads, shared_mem>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            w_ptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            eps, hidden_size, add_unit_offset, has_weight
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* w_ptr = has_weight ? reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()) : nullptr;
        gemma_rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, shared_mem>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            w_ptr,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            eps, hidden_size, add_unit_offset, has_weight
        );
    } else {
        const float* w_ptr = has_weight ? weight.data_ptr<float>() : nullptr;
        gemma_rmsnorm_kernel<float><<<num_rows, threads, shared_mem>>>(
            input.data_ptr<float>(),
            w_ptr,
            output.data_ptr<float>(),
            eps, hidden_size, add_unit_offset, has_weight
        );
    }
}

template <typename T>
__global__ void fused_rope_inplace_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    const int64_t* __restrict__ position_ids,
    int S,
    int H_q,
    int H_k,
    int D,
    int half_D,
    int64_t q_stride_b,
    int64_t q_stride_h,
    int64_t q_stride_s,
    int64_t k_stride_b,
    int64_t k_stride_h,
    int64_t k_stride_s
) {
    int bs_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;
    if (d >= half_D) return;

    int b = bs_idx / S;
    int s = bs_idx % S;
    int64_t pos = position_ids[bs_idx];

    float c1 = static_cast<float>(cos[pos * D + d]);
    float s1 = static_cast<float>(sin[pos * D + d]);
    float c2 = static_cast<float>(cos[pos * D + d + half_D]);
    float s2 = static_cast<float>(sin[pos * D + d + half_D]);

    T* ptr = nullptr;
    if (head_idx < H_q) {
        ptr = q + b * q_stride_b + head_idx * q_stride_h + s * q_stride_s;
    } else if (k != nullptr) {
        int kh = head_idx - H_q;
        ptr = k + b * k_stride_b + kh * k_stride_h + s * k_stride_s;
    } else {
        return;
    }

    float x1 = static_cast<float>(ptr[d]);
    float x2 = static_cast<float>(ptr[d + half_D]);

    ptr[d] = static_cast<T>(x1 * c1 - x2 * s1);
    ptr[d + half_D] = static_cast<T>(x2 * c2 + x1 * s2);
}

void launch_fused_rope_inplace(
    torch::Tensor& q,
    torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& position_ids
) {
    // q: [B, H_q, S, D], k: [B, H_k, S, D] (or empty)
    int B = q.size(0);
    int H_q = q.size(1);
    int S = q.size(2);
    int D = q.size(3);
    int half_D = D / 2;

    bool has_k = k.defined() && k.numel() > 0;
    int H_k = has_k ? k.size(1) : 0;

    int64_t q_sb = q.stride(0);
    int64_t q_sh = q.stride(1);
    int64_t q_ss = q.stride(2);

    int64_t k_sb = has_k ? k.stride(0) : 0;
    int64_t k_sh = has_k ? k.stride(1) : 0;
    int64_t k_ss = has_k ? k.stride(2) : 0;

    auto pos_contig = position_ids.is_contiguous() ? position_ids : position_ids.contiguous();
    auto cos_contig = cos.is_contiguous() ? cos : cos.contiguous();
    auto sin_contig = sin.is_contiguous() ? sin : sin.contiguous();

    dim3 grid(B * S, H_q + H_k);
    dim3 block(half_D);

    if (q.scalar_type() == torch::kBFloat16) {
        __nv_bfloat16* k_ptr = has_k ? reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()) : nullptr;
        fused_rope_inplace_kernel<__nv_bfloat16><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            k_ptr,
            reinterpret_cast<const __nv_bfloat16*>(cos_contig.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(sin_contig.data_ptr<at::BFloat16>()),
            pos_contig.data_ptr<int64_t>(),
            S, H_q, H_k, D, half_D,
            q_sb, q_sh, q_ss, k_sb, k_sh, k_ss
        );
    } else if (q.scalar_type() == torch::kFloat16) {
        half* k_ptr = has_k ? reinterpret_cast<half*>(k.data_ptr<at::Half>()) : nullptr;
        fused_rope_inplace_kernel<half><<<grid, block>>>(
            reinterpret_cast<half*>(q.data_ptr<at::Half>()),
            k_ptr,
            reinterpret_cast<const half*>(cos_contig.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(sin_contig.data_ptr<at::Half>()),
            pos_contig.data_ptr<int64_t>(),
            S, H_q, H_k, D, half_D,
            q_sb, q_sh, q_ss, k_sb, k_sh, k_ss
        );
    } else {
        float* k_ptr = has_k ? k.data_ptr<float>() : nullptr;
        fused_rope_inplace_kernel<float><<<grid, block>>>(
            q.data_ptr<float>(),
            k_ptr,
            cos_contig.data_ptr<float>(),
            sin_contig.data_ptr<float>(),
            pos_contig.data_ptr<int64_t>(),
            S, H_q, H_k, D, half_D,
            q_sb, q_sh, q_ss, k_sb, k_sh, k_ss
        );
    }
}
