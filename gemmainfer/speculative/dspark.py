import logging
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False

logger = logging.getLogger("gemmainfer")

_EMPTY_TENSOR = torch.empty(0)


def _patch_assistant_rmsnorm(assistant_model: nn.Module) -> None:
    """Replace slow Python RMSNorm forward methods in HF assistant model with fused CUDA kernel."""
    if not _HAS_CSRC or not hasattr(gemmainfer_csrc, "rmsnorm"):
        return

    for mod in assistant_model.modules():
        cls_name = mod.__class__.__name__
        if "RMSNorm" in cls_name:
            eps = getattr(mod, "eps", 1e-6)
            weight = getattr(mod, "weight", None)
            with_scale = getattr(mod, "with_scale", weight is not None)

            def _make_fast_rmsnorm(w, epsilon, has_s):
                def _fast_forward(x: torch.Tensor) -> torch.Tensor:
                    if not x.is_cuda:
                        in_f = x.float()
                        normed = (in_f * torch.rsqrt(in_f.pow(2).mean(-1, keepdim=True) + epsilon)).type_as(x)
                        return normed * w if (has_s and w is not None) else normed
                    x_c = x if x.is_contiguous() else x.contiguous()
                    out = torch.empty_like(x_c)
                    w_arg = w if (has_s and w is not None and w.is_contiguous()) else _EMPTY_TENSOR
                    gemmainfer_csrc.rmsnorm(x_c, w_arg, out, epsilon, False)
                    return out
                return _fast_forward

            mod.forward = _make_fast_rmsnorm(weight, eps, with_scale)


class PromptLookupCache:
    """Fast n-gram lookup cache to propose draft tokens matching prompt and output history."""

    def __init__(self, ngram_size: int = 2, max_draft_len: int = 4):
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len

    def propose(self, token_ids: List[int]) -> List[int]:
        """Propose speculative draft tokens by searching previous occurrences of the suffix."""
        if len(token_ids) < self.ngram_size + 1:
            return []

        suffix = tuple(token_ids[-self.ngram_size:])
        hist_len = len(token_ids) - self.ngram_size
        for i in range(hist_len - 1, -1, -1):
            if tuple(token_ids[i : i + self.ngram_size]) == suffix:
                match_end = i + self.ngram_size
                draft = token_ids[match_end : match_end + self.max_draft_len]
                if draft:
                    return draft
        return []


class Gemma4AssistantDrafter:
    """
    Drafter utilizing Google's official companion assistant model
    (google/gemma-4-26B-A4B-it-assistant) with shared KV states,
    fused CUDA RMSNorm/RoPE acceleration, and zero-sync GPU drafting.
    """

    def __init__(
        self,
        assistant_model,
        embed_tokens: nn.Embedding,
        max_draft_len: int = 4,
        confidence_threshold: float = 0.25,
    ):
        self.assistant = assistant_model
        self.embed_tokens = embed_tokens
        self.max_draft_len = max_draft_len
        self.confidence_threshold = confidence_threshold
        if isinstance(assistant_model, nn.Module):
            _patch_assistant_rmsnorm(assistant_model)
        self._has_direct_layers = (
            hasattr(assistant_model, "pre_projection")
            and hasattr(assistant_model, "post_projection")
            and hasattr(assistant_model, "lm_head")
            and hasattr(assistant_model, "model")
            and hasattr(assistant_model.model, "layers")
            and hasattr(assistant_model.model, "rotary_emb")
        )

    @torch.inference_mode()
    def draft(
        self,
        last_token_id: int,
        last_hidden_state: torch.Tensor,
        shared_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        cur_pos: int,
        forced_first_token: Optional[int] = None,
    ) -> List[int]:
        """Propose draft tokens using companion model and shared KV states."""
        device = last_hidden_state.device
        start_tok = forced_first_token if forced_first_token is not None else last_token_id
        curr_token = torch.tensor([[start_tok]], dtype=torch.long, device=device)

        curr_h = last_hidden_state
        if curr_h.dim() == 2:
            curr_h = curr_h.unsqueeze(1)
        elif curr_h.shape[1] > 1:
            curr_h = curr_h[:, -1:, :]

        cropped_kv = {}
        for k, v in shared_kv.items():
            k_t = v[0][:, :, :cur_pos, :] if v[0].size(2) > cur_pos else v[0]
            v_t = v[1][:, :, :cur_pos, :] if v[1].size(2) > cur_pos else v[1]
            if k == "sliding_attention" and k_t.size(2) > 1024:
                k_t = k_t[:, :, -1024:, :]
                v_t = v_t[:, :, -1024:, :]
            cropped_kv[k] = (k_t, v_t)

        embed_scale = self.embed_tokens.weight.shape[-1] ** 0.5
        pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=device)

        if self._has_direct_layers and curr_h.is_cuda:
            return self._draft_direct_cuda(curr_token, curr_h, cropped_kv, pos_ids, embed_scale)

        drafted_tokens: List[int] = []
        for _ in range(self.max_draft_len):
            token_emb = self.embed_tokens(curr_token) * embed_scale
            inputs_embeds = torch.cat([token_emb, curr_h], dim=-1)

            out = self.assistant(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids,
                shared_kv_states=cropped_kv,
                use_cache=False,
            )

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            tok_val = next_token.item()
            drafted_tokens.append(tok_val)

            if tok_val in (1, 106):
                break

            curr_token = next_token
            curr_h = out.last_hidden_state[:, -1:, :]

        return drafted_tokens

    def _draft_direct_cuda(
        self,
        curr_token: torch.Tensor,
        curr_h: torch.Tensor,
        cropped_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        pos_ids: torch.Tensor,
        embed_scale: float,
    ) -> List[int]:
        """Zero-sync direct CUDA execution of Gemma4AssistantForCausalLM layers."""
        asst = self.assistant
        m = asst.model
        device = curr_h.device
        zero_pos = torch.zeros((1, 1), dtype=torch.long, device=device)

        pe_cache = {}
        for lt in m.unique_layer_types:
            cos, sin = m.rotary_emb(curr_h, pos_ids, lt)
            pe_cache[lt] = (cos.squeeze(0).contiguous(), sin.squeeze(0).contiguous())

        use_fused_rope = _HAS_CSRC and hasattr(gemmainfer_csrc, "fused_rope_inplace")
        drafted_tensors: List[torch.Tensor] = []

        for step in range(self.max_draft_len):
            token_emb = self.embed_tokens(curr_token) * embed_scale
            inputs_embeds = torch.cat([token_emb, curr_h], dim=-1)
            h = asst.pre_projection(inputs_embeds)

            for i, layer in enumerate(m.layers):
                lt = m.config.layer_types[i]
                cos_2d, sin_2d = pe_cache[lt]
                res = h
                h_norm = layer.input_layernorm(h)
                attn = layer.self_attn
                q = attn.q_proj(h_norm).view(1, 1, -1, attn.head_dim)
                q = attn.q_norm(q).transpose(1, 2)

                if use_fused_rope:
                    gemmainfer_csrc.fused_rope_inplace(q, _EMPTY_TENSOR, cos_2d, sin_2d, zero_pos)
                else:
                    half_d = attn.head_dim // 2
                    q1, q2 = q[..., :half_d], q[..., half_d:]
                    c = cos_2d.unsqueeze(0).unsqueeze(1)
                    s = sin_2d.unsqueeze(0).unsqueeze(1)
                    q = (q * c) + (torch.cat((-q2, q1), dim=-1) * s)

                k_shared, v_shared = cropped_kv[lt]
                attn_out = F.scaled_dot_product_attention(
                    q, k_shared, v_shared, scale=attn.scaling, enable_gqa=True
                )
                attn_out = attn_out.transpose(1, 2).reshape(1, 1, -1)
                h = res + layer.post_attention_layernorm(attn.o_proj(attn_out))

                res = h
                h = res + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
                h = h * layer.layer_scalar

            h = m.norm(h)
            if (
                getattr(asst.config, "use_ordered_embeddings", False)
                and asst.masked_embedding is not None
                and hasattr(asst.lm_head, "weight")
            ):
                logits = asst.masked_embedding(h, asst.lm_head.weight)
            else:
                logits = asst.lm_head(h)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            drafted_tensors.append(next_token)

            if step + 1 < self.max_draft_len:
                curr_token = next_token
                curr_h = asst.post_projection(h)

        all_drafted = torch.cat(drafted_tensors, dim=1)[0].tolist()
        trimmed: List[int] = []
        for tok_val in all_drafted:
            trimmed.append(tok_val)
            if tok_val in (1, 106):
                break
        return trimmed


class DSparkVerifier:
    """Speculative drafting and verification coordinator supporting companion MTP drafter and n-gram lookup."""

    def __init__(
        self,
        assistant_model=None,
        embed_tokens: Optional[nn.Embedding] = None,
        hidden_size: int = 2816,
        vocab_size: int = 262144,
        confidence_threshold: float = 0.25,
        max_draft_len: int = 4,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_draft_len = max_draft_len
        if assistant_model is not None and embed_tokens is not None:
            self.assistant_drafter = Gemma4AssistantDrafter(
                assistant_model=assistant_model,
                embed_tokens=embed_tokens,
                max_draft_len=max_draft_len,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.assistant_drafter = None
        self.fallback_cache = PromptLookupCache(max_draft_len=max_draft_len)

    def propose_draft(
        self,
        tokens: List[int],
        last_hidden_state: Optional[torch.Tensor] = None,
        shared_kv: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        cur_pos: Optional[int] = None,
        forced_first_token: Optional[int] = None,
    ) -> List[int]:
        """Propose draft continuation tokens using companion MTP drafter or prompt lookup cache."""
        if (
            self.assistant_drafter is not None
            and last_hidden_state is not None
            and shared_kv is not None
            and cur_pos is not None
        ):
            last_tok = forced_first_token if forced_first_token is not None else tokens[-1]
            draft = self.assistant_drafter.draft(
                last_token_id=last_tok,
                last_hidden_state=last_hidden_state,
                shared_kv=shared_kv,
                cur_pos=cur_pos,
                forced_first_token=forced_first_token,
            )
            if draft:
                return draft

        return self.fallback_cache.propose(tokens)

    def verify_tokens(
        self,
        draft_tokens: List[int],
        logits: torch.Tensor,
        temperature: float = 0.0,
    ) -> Tuple[List[int], int]:
        """Verify draft tokens against target model logits."""
        selected = torch.argmax(logits, dim=-1).tolist()
        accepted = []
        n_matches = 0

        for i, d in enumerate(draft_tokens):
            target_tok = selected[i]
            if target_tok == d:
                accepted.append(d)
                n_matches += 1
            else:
                accepted.append(target_tok)
                break
        else:
            bonus = selected[-1]
            accepted.append(bonus)

        logger.debug(
            "[SPEC DEBUG] draft=%s selected=%s n_matches=%d accepted=%s",
            draft_tokens,
            selected,
            n_matches,
            accepted,
        )
        return accepted, n_matches
