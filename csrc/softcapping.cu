#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void gemma_softcapping_kernel(
    T* __restrict__ logits,
    float cap,
    float inv_cap,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = static_cast<float>(logits[idx]);
        logits[idx] = static_cast<T>(tanhf(val * inv_cap) * cap);
    }
}

void launch_gemma_softcapping(
    torch::Tensor& logits,
    float cap
) {
    int64_t total_elements = logits.numel();
    float inv_cap = 1.0f / cap;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (logits.scalar_type() == torch::kFloat16) {
        gemma_softcapping_kernel<half><<<blocks, threads>>>(
            reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
            cap, inv_cap, total_elements
        );
    } else if (logits.scalar_type() == torch::kBFloat16) {
        gemma_softcapping_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
            cap, inv_cap, total_elements
        );
    } else {
        gemma_softcapping_kernel<float><<<blocks, threads>>>(
            logits.data_ptr<float>(),
            cap, inv_cap, total_elements
        );
    }
}
