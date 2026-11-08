import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from gemmainfer.models.config import GemmaConfig
from gemmainfer.kernels.norm import GemmaRMSNorm
from gemmainfer.kernels.quant import MarlinLinear


def _make_linear(in_features: int, out_features: int, bias: bool = False, quant_method: Optional[str] = None) -> nn.Module:
    if quant_method == "w4a16":
        group_size = 64 if in_features % 128 != 0 and in_features % 64 == 0 else 128
        return MarlinLinear(in_features, out_features, group_size=group_size, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=min(max_position_embeddings, 8192))

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        device = device or self.inv_freq.device
        dtype = dtype or torch.get_default_dtype()
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self.cos_cached = self.cos_cached.to(device=x.device, dtype=x.dtype)
            self.sin_cached = self.sin_cached.to(device=x.device, dtype=x.dtype)
        if seq_len is not None and seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        if seq_len is not None:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        return self.cos_cached, self.sin_cached


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_full = config.is_full_attention(layer_idx)

        if self.is_full:
            self.head_dim = config.global_head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_global_key_value_heads
            self.sliding_window = None
            self.rope_theta = config.global_rope_theta
            self.k_eq_v = config.k_eq_v
        else:
            self.head_dim = config.head_dim
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.sliding_window = config.sliding_window
            self.rope_theta = config.rope_theta
            self.k_eq_v = False

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar or (self.head_dim ** -0.5)

        self.q_proj = _make_linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, quant_method=config.quant_method)
        self.k_proj = _make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_method=config.quant_method)
        if not self.k_eq_v:
            self.v_proj = _make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_method=config.quant_method)
        else:
            self.v_proj = None

        self.o_proj = _make_linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, quant_method=config.quant_method)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)
        self.v_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps, add_unit_offset=config.add_unit_offset)

        self.rotary_emb = GemmaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = k if self.k_eq_v else self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        cos, sin = self.rotary_emb(v)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        if self.num_key_value_groups > 1:
            k_expanded = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            k_expanded, v_expanded = k, v

        total_k_len = k_expanded.size(2)
        q = q * self.scaling
        attn_weights = torch.matmul(q, k_expanded.transpose(-1, -2))

        if mask is not None:
            attn_weights = attn_weights + mask
        elif q_len > 1:
            causal_mask = torch.triu(torch.full((q_len, total_k_len), float("-inf"), device=q.device), diagonal=1)
            if self.sliding_window is not None:
                window_mask = torch.tril(torch.full((q_len, total_k_len), float("-inf"), device=q.device), diagonal=-self.sliding_window)
                causal_mask = causal_mask + window_mask
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        output = torch.matmul(attn_probs, v_expanded)
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(output), new_kv_cache
