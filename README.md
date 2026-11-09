# GemmaInfer

A high-performance C++/CUDA inference engine for **Gemma 4 26B A4B** (25.2B total / 3.8B active parameters) optimized for **24GB consumer GPUs** (NVIDIA GeForce RTX 3090 / 4090).

GemmaInfer delivers production-grade single-stream interactive decode speeds by combining **Split-K MoE kernel fusion**, **collaborative on-chip shared memory caching**, and **cross-model speculative verification** with Google's official companion assistant model (`google/gemma-4-26B-A4B-it-assistant`).

---

## Architecture

```mermaid
flowchart LR
    A["Prompt Tokens"] --> B["Base Model Prefill"]
    B --> C["Shared KV States (L28 / L29)"]
    C --> D["Multi-Token MTP Drafter"]
    D --> E["Batched Split-K Verification"]
    B --> E
    E --> F["Streaming Output"]
```

---

## Performance Benchmarks (RTX 3090 24GB)

Live benchmark on NVIDIA GeForce RTX 3090 (24GB VRAM, PCIe 4.0, CUDA 13.2) generating 64 completion tokens:

| Engine | Quantization / Method | Single-Stream Decode Speed | Speedup vs Baseline |
| :--- | :--- | :--- | :--- |
| **GemmaInfer (Speculative MTP, $K=6$)** | **W4A16 Token-Union + Direct CUDA Drafter** | **79.22 tok/s** | **5.77x faster** |
| **GemmaInfer (Adaptive Speculative MTP)** | **W4A16 Token-Union + Adaptive Drafter ($K=4\leftrightarrow 6$)** | **76.14 tok/s** | **5.55x faster** |
| **GemmaInfer (Autoregressive)** | W4A16 Split-K + Fused RoPE/RMSNorm | **30.99 tok/s** | **2.26x faster** |
| **`llama.cpp`** | GGUF Q4_K_M (`--flash-attn`) | 22.50 tok/s | 1.64x faster |
| **vLLM** | W4A16 / AWQ Triton MoE | 20.60 tok/s | 1.50x faster |
| *Original Baseline* | Naive Autoregressive W4A16 | 13.72 tok/s | 1.00x (Reference) |

*GemmaInfer breaks through the physical GDDR6X single-token memory bandwidth roofline by drafting 4–6 tokens in ~2.9 ms/step with zero CPU-GPU sync and verifying all candidates in a single register-unrolled Token-Union target pass.*

---

## Core Acceleration Techniques

- **Split-K Grid Reduction:** Partitions the GEMM reduction dimension across 8 concurrent threadblocks (`split_k = 8`), launching 88 blocks to achieve 100% compute occupancy across all 82 SMs on Ampere GPUs.
- **Multi-Token MoE Kernel Fusion:** Fuses gate projection, up projection, GeLU tanh activation, down projection, and native BF16 hardware atomic accumulation into vectorized batch dispatches, eliminating thousands of runtime Python kernel launches.
- **Cross-Model KV-Sharing:** Reuses sliding window (layer 28) and full attention (layer 29) key-value states directly across model boundaries, allowing the companion drafter to predict multi-token sequences in under 5 ms without recomputing projections.
- **Hoisted Scale Dequantization:** Caches quantization scale factors once per weight group, removing redundant memory fetches across unrolled inner loops.
- **Collaborative Shared Memory Caching:** Stashes active token activations in fast on-chip `__shared__` memory buffers to eliminate repeated L2/DRAM roundtrips.
- **SWA Bounded Ring-Buffer Management:** Recycles physical memory pages across 1,024-token sliding window attention layers, slashing KV cache memory pressure by over 70%.

---

## Quick Start

### Build & Install
```bash
git clone https://github.com/priyanshum17/gemmainfer.git
cd gemmainfer
pip install -e .
python setup.py build_ext --inplace
```

### Launch OpenAI-Compatible API Server
```bash
python scripts/serve.py \
  --model-path /path/to/gemma-4-26B-A4B-it-W4A16 \
  --assistant-model-path /path/to/gemma-4-26B-A4B-it-assistant \
  --host 0.0.0.0 --port 8000
```

### Python API
```python
import torch
from gemmainfer.models.config import GemmaConfig
from gemmainfer.core.engine import GemmaInferEngine

config = GemmaConfig.gemma_4_26b_a4b()
config.quant_method = "w4a16"

engine = GemmaInferEngine(
    config=config,
    device="cuda",
    dtype=torch.bfloat16,
    enable_dspark=True,
)

tokens = engine.generate(prompt_token_ids=[2, 106, 1645, 108], max_tokens=64)
print("Generated tokens:", tokens)
```

---

## Verification

```bash
# Run unit test suite
pytest tests/

# Benchmark live server endpoint
python scripts/benchmark_client.py
```
