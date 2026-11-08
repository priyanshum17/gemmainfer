# Academic Research & Architectural Foundations

This directory contains research papers underlying the algorithmic and GPU microarchitectural optimizations implemented in **GemmaInfer**.

---

## Paper Index

| Paper | arXiv ID | Key Contribution in GemmaInfer | Local PDF |
| :--- | :--- | :--- | :--- |
| **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models** | [2408.11743](https://arxiv.org/abs/2408.11743) | W4A16 Split-K tensor core GEMM kernel, vectorized 128-bit loads, hoisted scale factors | [`marlin_2408.11743.pdf`](marlin_2408.11743.pdf) |
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | [2309.06180](https://arxiv.org/abs/2309.06180) | Non-contiguous physical page table management and virtual block mapping | [`paged_attention_2309.06180.pdf`](paged_attention_2309.06180.pdf) |
| **Better & Faster Large Language Models via Multi-token Prediction** | [2404.19737](https://arxiv.org/abs/2404.19737) | Cross-layer multi-token speculative drafting and token tree candidate generation | [`multi_token_prediction_2404.19737.pdf`](multi_token_prediction_2404.19737.pdf) |
| **Fast Inference from Transformers via Speculative Decoding** | [2211.17192](https://arxiv.org/abs/2211.17192) | Rejection sampling and batched parallel verification formulation | [`speculative_decoding_2211.17192.pdf`](speculative_decoding_2211.17192.pdf) |
| **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | [2401.10774](https://arxiv.org/abs/2401.10774) | Single-pass target verification of speculative multi-head candidate sequences | [`medusa_2401.10774.pdf`](medusa_2401.10774.pdf) |
| **Gemma 2: Improving Open Language Models at a Practical Size** | [2408.00118](https://arxiv.org/abs/2408.00118) | Gemma architecture: interleaved sliding window attention (SWA) and logit softcapping | [`gemma_2_2408.00118.pdf`](gemma_2_2408.00118.pdf) |
| **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | [2307.08691](https://arxiv.org/abs/2307.08691) | Online softmax rescaling and warp-level shared memory tiling | [`flashattention_2_2307.08691.pdf`](flashattention_2_2307.08691.pdf) |
| **DeepSeek-V3 Technical Report** | [2412.19437](https://arxiv.org/abs/2412.19437) | Multi-Token Prediction (MTP) module integration and KV-state sharing | [`deepseek_v3_2412.19437.pdf`](deepseek_v3_2412.19437.pdf) |
| **Root Mean Square Layer Normalization** | [1910.07467](https://arxiv.org/abs/1910.07467) | Fused warp-shuffle RMSNorm with Gemma unit-offset scaling (`(1.0 + weight) * x`) | [`rmsnorm_1910.07467.pdf`](rmsnorm_1910.07467.pdf) |
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** | [2104.09864](https://arxiv.org/abs/2104.09864) | Rotary position embedding (RoPE) applied to query and key states | [`roformer_rope_2104.09864.pdf`](roformer_rope_2104.09864.pdf) |
