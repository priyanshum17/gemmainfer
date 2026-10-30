#include <torch/extension.h>

#ifdef USE_CUDA
void launch_gemma_rmsnorm(
    torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float eps,
    bool add_unit_offset
);

void launch_gemma_softcapping(
    torch::Tensor& logits,
    float cap
);

void launch_paged_swa_decode(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    torch::Tensor& out,
    int sliding_window,
    float scale
);

void launch_marlin_w4a16(
    const torch::Tensor& A,
    const torch::Tensor& B_packed,
    const torch::Tensor& scales,
    torch::Tensor& C,
    int group_size
);
#endif

void cpu_marlin_w4a16(
    const torch::Tensor& A,
    const torch::Tensor& B_packed,
    const torch::Tensor& scales,
    torch::Tensor& C,
    int group_size
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B_packed.size(1);
    int packed_K = K / 8;

    auto A_f = A.to(torch::kFloat32);
    auto scales_f = scales.to(torch::kFloat32);
    auto B_ptr = B_packed.data_ptr<int32_t>();
    auto A_ptr = A_f.data_ptr<float>();
    auto S_ptr = scales_f.data_ptr<float>();
    auto C_f = torch::zeros({M, N}, torch::kFloat32);
    auto C_ptr = C_f.data_ptr<float>();

    for (int m = 0; m < M; ++m) {
        for (int pk = 0; pk < packed_K; ++pk) {
            int k_base = pk * 8;
            for (int col = 0; col < N; ++col) {
                int32_t packed_val = B_ptr[pk * N + col];
                int group_idx = k_base / group_size;
                float scale = S_ptr[group_idx * N + col];

                for (int i = 0; i < 8; ++i) {
                    int k = k_base + i;
                    int32_t w_raw = (packed_val >> (i * 4)) & 0x0F;
                    float w = static_cast<float>(w_raw - 8);
                    float a = A_ptr[m * K + k];
                    C_ptr[m * N + col] += a * (w * scale);
                }
            }
        }
    }
    C.copy_(C_f.to(C.dtype()));
}

void cpu_gemma_rmsnorm(
    torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float eps,
    bool add_unit_offset
) {
    auto in_flat = input.view({-1, input.size(-1)}).to(torch::kFloat32);
    auto w = weight.to(torch::kFloat32);
    auto mean_sq = torch::mean(in_flat * in_flat, -1, true);
    auto rsqrt_val = torch::rsqrt(mean_sq + eps);
    auto scale = add_unit_offset ? (1.0f + w) : w;
    output.copy_((in_flat * rsqrt_val * scale).view_as(output));
}

void cpu_gemma_softcapping(
    torch::Tensor& logits,
    float cap
) {
    logits.copy_(cap * torch::tanh(logits / cap));
}

void rmsnorm_dispatch(
    torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    float eps,
    bool add_unit_offset
) {
#ifdef USE_CUDA
    if (input.is_cuda()) {
        launch_gemma_rmsnorm(input, weight, output, eps, add_unit_offset);
        return;
    }
#endif
    cpu_gemma_rmsnorm(input, weight, output, eps, add_unit_offset);
}

void softcapping_dispatch(
    torch::Tensor& logits,
    float cap
) {
#ifdef USE_CUDA
    if (logits.is_cuda()) {
        launch_gemma_softcapping(logits, cap);
        return;
    }
#endif
    cpu_gemma_softcapping(logits, cap);
}

void marlin_w4a16_dispatch(
    const torch::Tensor& A,
    const torch::Tensor& B_packed,
    const torch::Tensor& scales,
    torch::Tensor& C,
    int group_size
) {
#ifdef USE_CUDA
    if (A.is_cuda()) {
        launch_marlin_w4a16(A, B_packed, scales, C, group_size);
        return;
    }
#endif
    cpu_marlin_w4a16(A, B_packed, scales, C, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm_dispatch);
    m.def("softcapping", &softcapping_dispatch);
    m.def("marlin_w4a16", &marlin_w4a16_dispatch);
}
