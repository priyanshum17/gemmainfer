#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void paged_swa_decode_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    T* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    int sliding_window,
    float scale
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    int seq_len = seq_lens[b_idx];
    if (seq_len <= 0) return;

    int kv_head_idx = h_idx / (num_heads / num_kv_heads);
    const T* q_vec = q + (b_idx * num_heads + h_idx) * head_dim;

    int start_pos = 0;
    if (sliding_window > 0 && seq_len > sliding_window) {
        start_pos = seq_len - sliding_window;
    }
    int active_tokens = seq_len - start_pos;

    extern __shared__ float s_mem[];
    float* s_scores = s_mem;
    __shared__ float s_warp_max[32];
    __shared__ float s_warp_sum[32];
    __shared__ float s_block_max;
    __shared__ float s_block_inv_sum;

    float thread_max = -1e20f;

    for (int i = tid; i < active_tokens; i += blockDim.x) {
        int pos = start_pos + i;
        int logical_block = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = block_tables[b_idx * max_blocks + logical_block];

        const T* k_vec = k_cache + ((phys_block * num_kv_heads + kv_head_idx) * block_size + block_offset) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
        }
        dot *= scale;
        s_scores[i] = dot;
        if (dot > thread_max) thread_max = dot;
    }

    float warp_max = warp_reduce_max(thread_max);
    if (lane == 0) {
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? s_warp_max[tid] : -1e20f;
        val = warp_reduce_max(val);
        if (lane == 0) {
            s_block_max = val;
        }
    }
    __syncthreads();

    float block_max = s_block_max;
    float thread_sum = 0.0f;

    for (int i = tid; i < active_tokens; i += blockDim.x) {
        float p = expf(s_scores[i] - block_max);
        s_scores[i] = p;
        thread_sum += p;
    }

    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) {
        s_warp_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? s_warp_sum[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            s_block_inv_sum = (val > 0.0f) ? (1.0f / val) : 0.0f;
        }
    }
    __syncthreads();

    float inv_sum = s_block_inv_sum;

    T* out_vec = out + (b_idx * num_heads + h_idx) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < active_tokens; ++i) {
            int pos = start_pos + i;
            int logical_block = pos / block_size;
            int block_offset = pos % block_size;
            int phys_block = block_tables[b_idx * max_blocks + logical_block];

            const T* v_vec = v_cache + ((phys_block * num_kv_heads + kv_head_idx) * block_size + block_offset) * head_dim;
            acc += (s_scores[i] * inv_sum) * static_cast<float>(v_vec[d]);
        }
        out_vec[d] = static_cast<T>(acc);
    }
}

void launch_paged_swa_decode(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    torch::Tensor& out,
    int sliding_window,
    float scale
) {
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int head_dim = q.size(2);
    int num_kv_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int max_blocks = block_tables.size(1);

    dim3 grid(batch_size, num_heads);
    int threads = 128;
    int max_active = (sliding_window > 0) ? min(sliding_window, 2048) : 2048;
    size_t shared_mem = max_active * sizeof(float);

    if (q.scalar_type() == torch::kBFloat16) {
        paged_swa_decode_kernel<__nv_bfloat16><<<grid, threads, shared_mem>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()),
            block_tables.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            num_heads, num_kv_heads, head_dim, block_size, max_blocks,
            sliding_window, scale
        );
    } else {
        paged_swa_decode_kernel<float><<<grid, threads, shared_mem>>>(
            q.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            block_tables.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            num_heads, num_kv_heads, head_dim, block_size, max_blocks,
            sliding_window, scale
        );
    }
}
