import time
import torch
from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.engine import GemmaInferEngine


def benchmark_latency():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if device == "cuda":
        config = GemmaConfig.gemma_26b_moe()
        config.quant_method = "w4a16"
        dtype = torch.bfloat16
    else:
        config = GemmaConfig.gemma_test_small()
        dtype = torch.float32

    model = GemmaForCausalLM(config).to(device=device, dtype=dtype)
    engine = GemmaInferEngine(config=config, model=model, device=device, dtype=dtype, enable_dspark=True)

    prompt = [1, 15, 230, 48, 92, 105, 300, 45, 99, 120]
    max_new_tokens = 50

    _ = engine.generate(prompt[:4], max_tokens=5)

    start_time = time.perf_counter()
    tokens = engine.generate(prompt, max_tokens=max_new_tokens)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    tps = len(tokens) / total_time
    ms_per_tok = (total_time / len(tokens)) * 1000

    print("------------------------------------------------------------")
    print(f"Device: {device} | Output Tokens: {len(tokens)}")
    print("------------------------------------------------------------")
    print(f"Total Time:        {total_time:.4f} s")
    print(f"Throughput:        {tps:.2f} tok/s")
    print(f"Latency per token: {ms_per_tok:.2f} ms")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    benchmark_latency()
