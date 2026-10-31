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

template <typename T>
__global__ void gemma_rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps,
    int hidden_size,
    bool add_unit_offset
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

    float warp_sum = warp_reduce_sum<float>(thread_sq_sum);
    if (lane == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sq_sum = 0.0f;
    if (warp_id == 0) {
        float val = (tid < num_warps) ? s_warp_sums[tid] : 0.0f;
        block_sq_sum = warp_reduce_sum<float>(val);
        if (lane == 0) {
            s_warp_sums[0] = block_sq_sum;
        }
    }
    __syncthreads();

    float rsqrt_val = rsqrtf((s_warp_sums[0] / static_cast<float>(hidden_size)) + eps);

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float in_val = static_cast<float>(row_in[i]);
        float w_val = static_cast<float>(weight[i]);
        float scale = add_unit_offset ? (1.0f + w_val) : w_val;
        row_out[i] = static_cast<T>(in_val * rsqrt_val * scale);
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

    int threads = (hidden_size > 4096) ? 1024 : ((hidden_size > 2048) ? 512 : 256);
    size_t shared_mem = (threads / 32) * sizeof(float);

    if (input.scalar_type() == torch::kFloat16) {
        gemma_rmsnorm_kernel<half><<<num_rows, threads, shared_mem>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            eps, hidden_size, add_unit_offset
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        gemma_rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, shared_mem>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            eps, hidden_size, add_unit_offset
        );
    } else {
        gemma_rmsnorm_kernel<float><<<num_rows, threads, shared_mem>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            eps, hidden_size, add_unit_offset
        );
    }
}
