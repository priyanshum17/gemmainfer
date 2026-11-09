import time
import torch
from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.engine import GemmaInferEngine


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Gemma 4 26B A4B exact specifications: 25.23B total parameters, 3.82B active per token
    total_params = 25.23e9

    fp16_gb = (total_params * 2) / (1024**3)
    w4a16_gb = (total_params * (0.5 + 2.0 / 128)) / (1024**3)

    print("-------------------------------------------------------------------------")
    print(f"Gemma 4 26B A4B (Total: 25.23B | Active: 3.82B) on {device}")
    print(f"VRAM Static Weights: FP16 = {fp16_gb:.2f} GB | W4A16 = {w4a16_gb:.2f} GB (Leaves ~11 GB for KV cache)")
    print("-------------------------------------------------------------------------")

    if device == "cuda":
        config = GemmaConfig.gemma_4_26b_a4b()
        config.quant_method = "w4a16"
        dtype = torch.bfloat16
    else:
        config = GemmaConfig.gemma_test_small()
        config.quant_method = "w4a16"
        dtype = torch.float32

    model = GemmaForCausalLM(config).to(device=device, dtype=dtype)
    prompt = [1, 10, 20, 30, 10, 20, 30, 40, 50, 60]
    num_tokens = 30

    engine_base = GemmaInferEngine(config=config, model=model, device=device, dtype=dtype, enable_dspark=False)
    _ = engine_base.generate(prompt[:4], max_tokens=2)
    t0 = time.perf_counter()
    out1 = engine_base.generate(prompt, max_tokens=num_tokens)
    t1 = time.perf_counter()
    time_base = t1 - t0
    tps_base = len(out1) / time_base

    engine_dspark = GemmaInferEngine(config=config, model=model, device=device, dtype=dtype, enable_dspark=True)
    _ = engine_dspark.generate(prompt[:4], max_tokens=2)
    t0 = time.perf_counter()
    out2 = engine_dspark.generate(prompt, max_tokens=num_tokens)
    t1 = time.perf_counter()
    time_dspark = t1 - t0
    tps_dspark = len(out2) / time_dspark

    speedup = tps_dspark / tps_base

    print(f"Sequential Decode:     {time_base:8.4f}s  |  {tps_base:8.2f} tok/s")
    print(f"DSpark Decode:         {time_dspark:8.4f}s  |  {tps_dspark:8.2f} tok/s  ({speedup:.2f}x)")
    print("-------------------------------------------------------------------------")


if __name__ == "__main__":
    run_benchmark()
