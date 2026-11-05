import time
import torch
from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.engine import GemmaInferEngine


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if device == "cuda":
        config = GemmaConfig.gemma_26b_moe()
        dtype = torch.bfloat16
    else:
        config = GemmaConfig.gemma_test_small()
        dtype = torch.float32

    model = GemmaForCausalLM(config).to(device=device, dtype=dtype)
    prompt = [1, 42, 108, 99, 42, 108, 99, 50, 60, 70]
    num_tokens = 40

    baseline_engine = GemmaInferEngine(config=config, model=model, device=device, dtype=dtype, enable_dspark=False, enable_cuda_graphs=False)
    _ = baseline_engine.generate(prompt[:4], max_tokens=4)
    t0 = time.perf_counter()
    baseline_out = baseline_engine.generate(prompt, max_tokens=num_tokens)
    t1 = time.perf_counter()
    baseline_time = t1 - t0
    baseline_tps = len(baseline_out) / baseline_time

    gemmainfer_engine = GemmaInferEngine(config=config, model=model, device=device, dtype=dtype, enable_dspark=True, enable_cuda_graphs=True)
    _ = gemmainfer_engine.generate(prompt[:4], max_tokens=4)
    t0 = time.perf_counter()
    gemmainfer_out = gemmainfer_engine.generate(prompt, max_tokens=num_tokens)
    t1 = time.perf_counter()
    gemmainfer_time = t1 - t0
    gemmainfer_tps = len(gemmainfer_out) / gemmainfer_time

    speedup = gemmainfer_tps / baseline_tps

    print("-----------------------------------------------------------------")
    print(f"Device: {device} | Comparison against Sequential Baseline")
    print("-----------------------------------------------------------------")
    print(f"Sequential Baseline:  {baseline_time:8.4f}s  |  {baseline_tps:8.2f} tok/s")
    print(f"GemmaInfer (DSpark):  {gemmainfer_time:8.4f}s  |  {gemmainfer_tps:8.2f} tok/s  ({speedup:.2f}x)")
    print("-----------------------------------------------------------------")


if __name__ == "__main__":
    run_comparison()
