import logging
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("gemmainfer")


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
    teacher-forced bonus token synchronization, and confidence-gated stopping.
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
        curr_token = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
        curr_h = last_hidden_state
        if curr_h.dim() == 2:
            curr_h = curr_h.unsqueeze(1)
        elif curr_h.shape[1] > 1:
            curr_h = curr_h[:, -1:, :]

        # Crop shared KV states up to cur_pos when buffer exceeds cur_pos
        cropped_kv = {
            k: (
                v[0][:, :, :cur_pos, :] if v[0].size(2) > cur_pos else v[0],
                v[1][:, :, :cur_pos, :] if v[1].size(2) > cur_pos else v[1],
            )
            for k, v in shared_kv.items()
        }

        drafted_tokens: List[int] = []
        embed_scale = self.embed_tokens.weight.shape[-1] ** 0.5

        if forced_first_token is not None:
            # Teacher-forced step 0: synchronize assistant hidden state at (cur_pos - 1)
            # using the already-verified token (last_token_id) and its target hidden state,
            # then feed the already-known target prediction (forced_first_token) into step 1.
            pos_0 = torch.tensor([[cur_pos - 1]], dtype=torch.long, device=device)
            token_emb_0 = self.embed_tokens(curr_token) * embed_scale
            inputs_embeds_0 = torch.cat([token_emb_0, curr_h], dim=-1)
            out_0 = self.assistant(
                inputs_embeds=inputs_embeds_0,
                position_ids=pos_0,
                shared_kv_states=cropped_kv,
                use_cache=False,
            )
            curr_token = torch.tensor([[forced_first_token]], dtype=torch.long, device=device)
            curr_h = out_0.last_hidden_state[:, -1:, :]
            cum_conf = 1.0

            for step in range(1, self.max_draft_len + 1):
                pos_ids = torch.tensor([[cur_pos - 1 + step]], dtype=torch.long, device=device)
                token_emb = self.embed_tokens(curr_token) * embed_scale
                inputs_embeds = torch.cat([token_emb, curr_h], dim=-1)

                out = self.assistant(
                    inputs_embeds=inputs_embeds,
                    position_ids=pos_ids,
                    shared_kv_states=cropped_kv,
                    use_cache=False,
                )

                step_logits = out.logits[:, -1, :]
                probs = F.softmax(step_logits, dim=-1, dtype=torch.float32)
                max_prob, next_token = torch.max(probs, dim=-1, keepdim=True)
                tok_val = next_token.item()
                cum_conf *= max_prob.item()

                drafted_tokens.append(tok_val)
                if tok_val in (1, 106) or (len(drafted_tokens) >= 2 and cum_conf < self.confidence_threshold):
                    break

                curr_token = next_token
                curr_h = out.last_hidden_state[:, -1:, :]

            return drafted_tokens

        # Standalone mode (when forced_first_token is None)
        for step in range(self.max_draft_len):
            pos_ids = torch.tensor([[cur_pos - 1 + step]], dtype=torch.long, device=device)
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
            last_tok = tokens[-2] if (forced_first_token is not None and len(tokens) >= 2) else tokens[-1]
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
