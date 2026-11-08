from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gemmainfer.models.config import GemmaConfig
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import GemmaMLP, apply_logit_softcapping
from gemmainfer.kernels.attention import GemmaAttention
from gemmainfer.kernels.quant import marlin_forward


class GemmaBatchedMoE(nn.Module):
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
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        x_normed = self.router_norm(x_flat) * self.router_scale * (self.hidden_size ** -0.5)
        router_logits = self.gate(x_normed)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        scales = self.expert_scale[indices]
        weights = (weights * scales).type_as(x)

        out = torch.zeros_like(x_flat)
        is_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

        if self.quant_method == "w4a16":
            for k in range(self.top_k):
                e_idx = indices[:, k]
                w = weights[:, k : k + 1]
                p_gu = self.gate_up_packed[e_idx]
                s_gu = self.gate_up_scales[e_idx]
                p_d = self.down_packed[e_idx]
                s_d = self.down_scales[e_idx]

                if num_tokens == 1:
                    gu = marlin_forward(x_flat, p_gu.squeeze(0), s_gu.squeeze(0), self.gu_group_size, 2 * self.moe_intermediate_size)
                    gate, up = gu.chunk(2, dim=-1)
                    act = F.gelu(gate, approximate="tanh") * up
                    expert_out = marlin_forward(act, p_d.squeeze(0), s_d.squeeze(0), self.d_group_size, self.hidden_size)
                    out = out + expert_out * w
                else:
                    for t in range(num_tokens):
                        tok_x = x_flat[t : t + 1]
                        gu = marlin_forward(tok_x, p_gu[t], s_gu[t], self.gu_group_size, 2 * self.moe_intermediate_size)
                        gate, up = gu.chunk(2, dim=-1)
                        act = F.gelu(gate, approximate="tanh") * up
                        expert_out = marlin_forward(act, p_d[t], s_d[t], self.d_group_size, self.hidden_size)
                        out[t : t + 1] = out[t : t + 1] + expert_out * w[t : t + 1]
            return out.view(orig_shape)

        if num_tokens <= 64 or is_capturing:
            for k in range(self.top_k):
                e_idx = indices[:, k]
                w = weights[:, k : k + 1]
                w_gu = self.gate_up_proj[e_idx]
                gu = torch.bmm(x_flat.unsqueeze(1), w_gu).squeeze(1)
                gate, up = gu.chunk(2, dim=-1)
                act = F.gelu(gate, approximate="tanh") * up
                w_d = self.down_proj[e_idx]
                expert_out = torch.bmm(act.unsqueeze(1), w_d).squeeze(1)
                out = out + expert_out * w
            return out.view(orig_shape)

        flat_indices = indices.view(-1)
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        weights_flat = weights.view(-1, 1)

        for e_idx in range(self.num_experts):
            mask = (flat_indices == e_idx)
            if not mask.any():
                continue
            active_tokens = token_indices[mask]
            expert_in = x_flat[active_tokens]

            w_gu = self.gate_up_proj[e_idx]
            gu = torch.matmul(expert_in, w_gu)
            gate, up = gu.chunk(2, dim=-1)
            act = F.gelu(gate, approximate="tanh") * up

            w_d = self.down_proj[e_idx]
            expert_out = torch.matmul(act, w_d)

            expert_weights = weights_flat[mask]
            out.index_add_(0, active_tokens, expert_out * expert_weights)

        return out.view(orig_shape)


class GemmaDecoderLayer(nn.Module):
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_logits_only_last: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            if past_key_values is None or past_key_values[0] is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            else:
                past_len = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        hidden_states = (self.embed_tokens(input_ids) * (self.config.hidden_size ** 0.5)).to(dtype=self.embed_tokens.weight.dtype)

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=layer_past,
            )
            new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)

        if return_logits_only_last:
            hidden_states = hidden_states[:, -1:, :]

        logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            logits = apply_logit_softcapping(logits, self.config.final_logit_softcapping)

        return logits, new_past_key_values
