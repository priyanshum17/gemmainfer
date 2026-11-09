from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gemmainfer.models.config import GemmaConfig
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import GemmaMLP, apply_logit_softcapping
from gemmainfer.kernels.attention import GemmaAttention
from gemmainfer.kernels.quant import MarlinLinear, marlin_forward

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False


class GemmaBatchedMoE(nn.Module):
    """Batched Mixture of Experts module with Token-Union W4A16 coalesced execution."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.quant_method = config.quant_method

        self.router_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.router_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.expert_scale = nn.Parameter(torch.ones(self.num_experts))

        if self.quant_method == "w4a16":
            gu_in, gu_out = config.hidden_size, 2 * self.moe_intermediate_size
            d_in, d_out = self.moe_intermediate_size, config.hidden_size
            self.gu_group_size = config.quant_group_size
            self.d_group_size = config.quant_group_size

            dtype = getattr(torch, config.dtype, torch.bfloat16)
            self.register_buffer(
                "gate_up_packed",
                torch.zeros(self.num_experts, gu_in // 8, gu_out, dtype=torch.int32),
            )
            self.register_buffer(
                "gate_up_scales",
                torch.zeros(self.num_experts, gu_in // self.gu_group_size, gu_out, dtype=dtype),
            )
            self.register_buffer(
                "down_packed",
                torch.zeros(self.num_experts, d_in // 8, d_out, dtype=torch.int32),
            )
            self.register_buffer(
                "down_scales",
                torch.zeros(self.num_experts, d_in // self.d_group_size, d_out, dtype=dtype),
            )
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(self.num_experts, config.hidden_size, 2 * self.moe_intermediate_size)
            )
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.moe_intermediate_size, config.hidden_size)
            )
            nn.init.normal_(self.gate_up_proj, std=0.02)
            nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to top-k experts and execute Token-Union coalesced expert projections."""
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size).contiguous()
        num_tokens = x_flat.shape[0]

        x_normed = self.router_norm(x_flat) * (self.router_scale * (self.hidden_size ** -0.5))
        router_logits = self.gate(x_normed)
        top_logits, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(top_logits, dim=-1, dtype=torch.float32)
        scales = self.expert_scale[indices]
        weights = (weights * scales).type_as(x)

        out = torch.zeros_like(x_flat)

        if self.quant_method == "w4a16":
            if _HAS_CSRC and hasattr(gemmainfer_csrc, "fused_batched_moe"):
                idx_i32 = indices.to(torch.int32)
                if num_tokens <= 8:
                    gemmainfer_csrc.fused_batched_moe(
                        x_flat,
                        self.gate_up_packed,
                        self.gate_up_scales,
                        self.down_packed,
                        self.down_scales,
                        idx_i32,
                        weights,
                        out,
                        self.gu_group_size,
                    )
                else:
                    for start in range(0, num_tokens, 8):
                        end = min(start + 8, num_tokens)
                        gemmainfer_csrc.fused_batched_moe(
                            x_flat[start:end],
                            self.gate_up_packed,
                            self.gate_up_scales,
                            self.down_packed,
                            self.down_scales,
                            idx_i32[start:end],
                            weights[start:end],
                            out[start:end],
                            self.gu_group_size,
                        )
                return out.view(orig_shape)

            flat_indices = indices.view(-1)
            token_indices = (
                torch.arange(num_tokens, device=x.device)
                .unsqueeze(1)
                .expand(-1, self.top_k)
                .reshape(-1)
            )
            weights_flat = weights.view(-1, 1)

            for e_idx in torch.unique(flat_indices).tolist():
                mask = flat_indices == e_idx
                active_tokens = token_indices[mask]
                expert_in = x_flat[active_tokens]

                gu = marlin_forward(
                    expert_in,
                    self.gate_up_packed[e_idx],
                    self.gate_up_scales[e_idx],
                    self.gu_group_size,
                    2 * self.moe_intermediate_size,
                )
                gate, up = gu.chunk(2, dim=-1)
                act = F.gelu(gate, approximate="tanh") * up
                expert_out = marlin_forward(
                    act,
                    self.down_packed[e_idx],
                    self.down_scales[e_idx],
                    self.d_group_size,
                    self.hidden_size,
                )
                out.index_add_(0, active_tokens, expert_out * weights_flat[mask])

            return out.view(orig_shape)

        flat_indices = indices.view(-1)
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        weights_flat = weights.view(-1, 1)

        for e_idx in torch.unique(flat_indices).tolist():
            mask = flat_indices == e_idx
            active_tokens = token_indices[mask]
            expert_in = x_flat[active_tokens]

            gu = torch.matmul(expert_in, self.gate_up_proj[e_idx])
            gate, up = gu.chunk(2, dim=-1)
            act = F.gelu(gate, approximate="tanh") * up
            expert_out = torch.matmul(act, self.down_proj[e_idx])
            out.index_add_(0, active_tokens, expert_out * weights_flat[mask])

        return out.view(orig_shape)


class GemmaDecoderLayer(nn.Module):
    """Gemma transformer decoder layer with interleaved attention, shared MLP, and MoE blocks."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)

        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size, quant_method=config.quant_method, group_size=config.quant_group_size)

        if config.is_moe:
            self.post_feedforward_layernorm_1 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
            self.pre_feedforward_layernorm_2 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
            self.moe = GemmaBatchedMoE(config)
            self.post_feedforward_layernorm_2 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.moe = None

        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.register_buffer("layer_scalar", torch.ones(1))

    @property
    def shared_mlp(self):
        return self.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Execute decoder layer forward pass returning hidden states and updated KV cache."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        mlp_in = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(mlp_in)

        if self.moe is not None:
            hidden_states_1 = self.post_feedforward_layernorm_1(mlp_out)
            moe_in = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe(moe_in)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            ffn_combined = hidden_states_1 + hidden_states_2
        else:
            ffn_combined = mlp_out

        hidden_states = self.post_feedforward_layernorm(ffn_combined)
        hidden_states = residual + hidden_states
        hidden_states = (hidden_states * self.layer_scalar).type_as(hidden_states)

        return hidden_states, new_kv


class GemmaForCausalLM(nn.Module):
    """Gemma causal language model for autoregressive generation and speculative verification."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        if config.quant_method == "w4a16" and config.hidden_size % config.quant_group_size == 0 and config.hidden_size % 8 == 0:
            dtype = getattr(torch, config.dtype, torch.bfloat16)
            self.lm_head = MarlinLinear(
                config.hidden_size,
                config.vocab_size,
                group_size=config.quant_group_size,
                bias=False,
                dtype=dtype,
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    def init_kv_cache(
        self,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Allocate speculation-safe bounded SWA buffers and full-attention KV caches."""
        device = device or self.embed_tokens.weight.device
        dtype = dtype or self.embed_tokens.weight.dtype
        cache = []
        for layer in self.layers:
            attn = layer.self_attn
            if attn.sliding_window is not None:
                layer_cap = min(max_seq_len, attn.sliding_window + 64)
            else:
                layer_cap = max_seq_len

            k_buf = torch.zeros(
                (batch_size, attn.num_key_value_heads, layer_cap, attn.head_dim),
                device=device,
                dtype=dtype,
            )
            v_buf = torch.zeros(
                (batch_size, attn.num_key_value_heads, layer_cap, attn.head_dim),
                device=device,
                dtype=dtype,
            )
            meta = [0, 0]
            cache.append((k_buf, v_buf, meta))
        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        return_logits_only_last: bool = True,
        return_hidden_states: bool = False,
    ):
        """Execute model forward pass yielding logits and updated KV caches."""
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            if past_key_values is None or past_key_values[0] is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            else:
                if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) == 3:
                    past_len = past_key_values[0][2][0]
                else:
                    past_len = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        hidden_states = (self.embed_tokens(input_ids) * (self.config.hidden_size ** 0.5)).to(dtype=self.embed_tokens.weight.dtype)

        shared_mask = None
        if seq_len > 1:
            if past_key_values is None or past_key_values[0] is None:
                total_k_len = seq_len
            elif isinstance(past_key_values[0], tuple) and len(past_key_values[0]) == 3:
                total_k_len = past_key_values[0][2][0] + seq_len
            else:
                total_k_len = past_key_values[0][0].shape[2] + seq_len
            if self.config.sliding_window is None or total_k_len <= self.config.sliding_window:
                causal_diag = total_k_len - seq_len + 1
                shared_mask = torch.triu(
                    torch.full((seq_len, total_k_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
                    diagonal=causal_diag,
                ).unsqueeze(0).unsqueeze(0)

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=layer_past,
                mask=shared_mask,
            )
            new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)

        last_hidden = hidden_states[:, -1:, :] if return_logits_only_last else hidden_states

        if return_logits_only_last:
            logits = self.lm_head(last_hidden)
        else:
            logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            logits = apply_logit_softcapping(logits, self.config.final_logit_softcapping)

        if return_hidden_states:
            return logits, new_past_key_values, last_hidden
        return logits, new_past_key_values
