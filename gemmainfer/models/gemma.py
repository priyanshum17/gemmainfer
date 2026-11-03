from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gemmainfer.models.config import GemmaConfig
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.activations import GemmaMLP, apply_logit_softcapping
from gemmainfer.kernels.attention import GemmaAttention


class GemmaBatchedMoE(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.num_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.gate_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=False)
        self.expert_scale = nn.Parameter(torch.ones(self.num_experts))

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

        x_normed = self.gate_norm(x_flat)
        router_logits = self.gate(x_normed) * (self.hidden_size ** -0.5)
        router_probs = F.softmax(router_logits, dim=-1)

        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        scales = self.expert_scale[indices]
        weights = weights * scales

        out = torch.zeros_like(x_flat)
        
        flat_indices = indices.view(-1)
        unique_experts, inverse_map = torch.unique(flat_indices, return_inverse=True)
        
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        weights_flat = weights.view(-1, 1)

        for e_idx in unique_experts:
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
        
        if config.has_shared_mlp:
            self.shared_mlp = GemmaMLP(config.hidden_size, config.intermediate_size, quant_method=config.quant_method)
            self.shared_mlp_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.shared_mlp = None

        if config.is_moe:
            self.moe = GemmaBatchedMoE(config)
            self.moe_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.moe = None

        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)

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
        normed_ffn_in = self.pre_feedforward_layernorm(hidden_states)

        ffn_out = torch.zeros_like(hidden_states)
        if self.shared_mlp is not None:
            ffn_out = ffn_out + self.shared_mlp_norm(self.shared_mlp(normed_ffn_in))
        if self.moe is not None:
            ffn_out = ffn_out + self.moe_norm(self.moe(normed_ffn_in))

        hidden_states = self.post_feedforward_layernorm(ffn_out)
        hidden_states = residual + hidden_states

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

        hidden_states = self.embed_tokens(input_ids) * (self.config.hidden_size ** 0.5)

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
