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

    text_cfg = cfg.get("text_config", cfg)
    is_quant = "quantization_config" in cfg or text_cfg.get("quant_method") == "w4a16"

    config = GemmaConfig(
        vocab_size=text_cfg.get("vocab_size", 262144),
        hidden_size=text_cfg.get("hidden_size", 2816),
        intermediate_size=text_cfg.get("intermediate_size", 2112),
        moe_intermediate_size=text_cfg.get("moe_intermediate_size", 704),
        num_hidden_layers=text_cfg.get("num_hidden_layers", 30),
        num_attention_heads=text_cfg.get("num_attention_heads", 16),
        num_key_value_heads=text_cfg.get("num_key_value_heads", 8),
        head_dim=text_cfg.get("head_dim", 256),
        global_head_dim=text_cfg.get("global_head_dim", 512),
        num_global_key_value_heads=text_cfg.get("num_global_key_value_heads", 2),
        sliding_window=text_cfg.get("sliding_window", 1024),
        attn_logit_softcapping=text_cfg.get("attn_logit_softcapping", None),
        final_logit_softcapping=text_cfg.get("final_logit_softcapping", 30.0),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        is_moe=text_cfg.get("enable_moe_block", True),
        num_routed_experts=text_cfg.get("num_experts", 128),
        num_experts_per_tok=text_cfg.get("top_k_experts", 8),
        quant_method="w4a16" if is_quant else None,
        dtype="bfloat16" if dtype == torch.bfloat16 else "float16",
    )

    model = GemmaForCausalLM(config).to(device=device, dtype=dtype)

    safetensor_files = sorted([
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    ])

    if safetensor_files:
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if "vision" in k:
                        continue

                    clean_k = k
                    if clean_k.startswith("model.language_model."):
                        clean_k = "model." + clean_k[len("model.language_model."):]
                    elif clean_k.startswith("language_model."):
                        clean_k = "model." + clean_k[len("language_model."):]

                    t = f.get_tensor(k)

                    if clean_k == "model.embed_tokens.weight":
                        model.embed_tokens.weight.data.copy_(t.to(device=device, dtype=dtype))
                        continue

                    if clean_k == "model.norm.weight":
                        model.norm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        continue

                    parts = clean_k.split(".")
                    if len(parts) >= 4 and parts[1] == "layers":
                        layer_idx = int(parts[2])
                        layer = model.layers[layer_idx]
                        suffix = ".".join(parts[3:])

                        if suffix == "input_layernorm.weight":
                            layer.input_layernorm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "post_attention_layernorm.weight":
                            layer.post_attention_layernorm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "pre_feedforward_layernorm.weight":
                            layer.pre_feedforward_layernorm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "post_feedforward_layernorm.weight":
                            layer.post_feedforward_layernorm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "self_attn.q_norm.weight":
                            layer.self_attn.q_norm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "self_attn.k_norm.weight":
                            layer.self_attn.k_norm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "self_attn.q_proj.weight_packed":
                            layer.self_attn.q_proj.packed_weight.data.copy_(t.t().to(device=device))
                        elif suffix == "self_attn.q_proj.weight_scale":
                            layer.self_attn.q_proj.scales.data.copy_(t.t().to(device=device, dtype=dtype))
                        elif suffix == "self_attn.k_proj.weight_packed":
                            layer.self_attn.k_proj.packed_weight.data.copy_(t.t().to(device=device))
                        elif suffix == "self_attn.k_proj.weight_scale":
                            layer.self_attn.k_proj.scales.data.copy_(t.t().to(device=device, dtype=dtype))
                        elif suffix == "self_attn.v_proj.weight_packed" and layer.self_attn.v_proj is not None:
                            layer.self_attn.v_proj.packed_weight.data.copy_(t.t().to(device=device))
                        elif suffix == "self_attn.v_proj.weight_scale" and layer.self_attn.v_proj is not None:
                            layer.self_attn.v_proj.scales.data.copy_(t.t().to(device=device, dtype=dtype))
                        elif suffix == "self_attn.o_proj.weight_packed":
                            layer.self_attn.o_proj.packed_weight.data.copy_(t.t().to(device=device))
                        elif suffix == "self_attn.o_proj.weight_scale":
                            layer.self_attn.o_proj.scales.data.copy_(t.t().to(device=device, dtype=dtype))
                        elif suffix == "router.proj.weight":
                            layer.moe.gate.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "router.per_expert_scale":
                            layer.moe.expert_scale.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix == "router.scale":
                            layer.moe.gate_norm.weight.data.copy_(t.to(device=device, dtype=dtype))
                        elif suffix.startswith("experts.") and layer.moe is not None:
                            exp_parts = suffix.split(".")
                            exp_idx = int(exp_parts[1])
                            proj_name = exp_parts[2]
                            param_type = exp_parts[3]

                            if proj_name == "down_proj" and param_type == "weight_packed":
                                layer.moe.down_packed[exp_idx].copy_(t.t().to(device=device))
                            elif proj_name == "down_proj" and param_type == "weight_scale":
                                layer.moe.down_scales[exp_idx].copy_(t.t().to(device=device, dtype=dtype))
                            elif proj_name == "gate_proj" and param_type == "weight_packed":
                                layer.moe.gate_up_packed[exp_idx, :, :layer.moe.moe_intermediate_size].copy_(t.t().to(device=device))
                            elif proj_name == "up_proj" and param_type == "weight_packed":
                                layer.moe.gate_up_packed[exp_idx, :, layer.moe.moe_intermediate_size:].copy_(t.t().to(device=device))
                            elif proj_name == "gate_proj" and param_type == "weight_scale":
                                layer.moe.gate_up_scales[exp_idx, :, :layer.moe.moe_intermediate_size].copy_(t.t().to(device=device, dtype=dtype))
                            elif proj_name == "up_proj" and param_type == "weight_scale":
                                layer.moe.gate_up_scales[exp_idx, :, layer.moe.moe_intermediate_size:].copy_(t.t().to(device=device, dtype=dtype))

        model.lm_head.weight = model.embed_tokens.weight

    return model, config
