import os
import json
from typing import Dict, Any, Optional, Tuple
import torch
from safetensors import safe_open

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM


def load_gemma_from_hf(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[GemmaForCausalLM, GemmaConfig]:
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Missing config.json in {model_path}")

    with open(config_file, "r") as f:
        cfg = json.load(f)

    config = GemmaConfig(
        vocab_size=cfg.get("vocab_size", 256000),
        hidden_size=cfg.get("hidden_size", 4608),
        intermediate_size=cfg.get("intermediate_size", 36864),
        num_hidden_layers=cfg.get("num_hidden_layers", 46),
        num_attention_heads=cfg.get("num_attention_heads", 32),
        num_key_value_heads=cfg.get("num_key_value_heads", 16),
        head_dim=cfg.get("head_dim", 128),
        sliding_window=cfg.get("sliding_window", 4096),
        attn_logit_softcapping=cfg.get("attn_logit_softcapping", 50.0),
        final_logit_softcapping=cfg.get("final_logit_softcapping", 30.0),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        is_moe=("num_experts" in cfg or "num_routed_experts" in cfg),
    )

    model = GemmaForCausalLM(config).to(device=device, dtype=dtype)

    safetensor_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    ]

    if safetensor_files:
        state_dict = {}
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device=str(device)) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k).to(dtype)
        model.load_state_dict(state_dict, strict=False)

    return model, config
