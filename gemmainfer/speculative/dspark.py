from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLookupCache:
    def __init__(self, ngram_size: int = 2, max_draft_len: int = 4):
        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len

    def propose(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) < self.ngram_size:
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


class DSparkSemiAutoregressiveDrafter(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, max_draft_len: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_draft_len = max_draft_len

        self.feature_expansion = nn.Linear(hidden_size, max_draft_len * hidden_size, bias=False)
        self.markov_recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.confidence_head = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        confidence_threshold: float = 0.7,
    ) -> Tuple[List[int], List[float]]:
        if self.feature_expansion.weight.dtype != last_hidden_state.dtype or self.feature_expansion.weight.device != last_hidden_state.device:
            self.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        bsz = last_hidden_state.shape[0]
        device = last_hidden_state.device

        features = self.feature_expansion(last_hidden_state).view(bsz, self.max_draft_len, self.hidden_size)
        state = last_hidden_state.squeeze(1) if last_hidden_state.dim() == 3 else last_hidden_state

        draft_tokens = []
        confidences = []
        prefix_prob = 1.0

        for k in range(self.max_draft_len):
            feat_k = features[:, k, :]
            state = self.markov_recurrent(feat_k, state)

            conf_logit = self.confidence_head(state)
            conf = torch.sigmoid(conf_logit).item()
            prefix_prob *= conf

            if prefix_prob < confidence_threshold and len(draft_tokens) > 0:
                break

            logits = self.lm_head(state)
            token = torch.argmax(logits, dim=-1).item()
            draft_tokens.append(token)
            confidences.append(conf)

        return draft_tokens, confidences


class Gemma4AssistantDrafter:
    """
    Drafter utilizing Google's official companion assistant model
    (e.g., google/gemma-4-26B-A4B-it-assistant) with shared KV states
    for multi-token speculative decoding.
    """
    def __init__(
        self,
        assistant_model,
        embed_tokens: nn.Embedding,
        max_draft_len: int = 3,
    ):
        self.assistant = assistant_model
        self.embed_tokens = embed_tokens
        self.max_draft_len = max_draft_len

    @torch.inference_mode()
    def draft(
        self,
        last_token_id: int,
        last_hidden_state: torch.Tensor,
        shared_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        cur_pos: int,
    ) -> List[int]:
        device = last_hidden_state.device
        curr_token = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
        curr_h = last_hidden_state
        if curr_h.dim() == 2:
            curr_h = curr_h.unsqueeze(1)
        elif curr_h.shape[1] > 1:
            curr_h = curr_h[:, -1:, :]

        # Crop shared KV states to cur_pos
        cropped_kv = {
            k: (v[0][:, :, :cur_pos, :], v[1][:, :, :cur_pos, :])
            for k, v in shared_kv.items()
        }

        pos_ids = torch.tensor([[cur_pos - 1]], dtype=torch.long, device=device)
        drafted_tokens = []

        for _ in range(self.max_draft_len):
            token_emb = self.embed_tokens(curr_token)
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

            # If EOS or newline/eot, stop drafting
            if tok_val in (1, 106):
                break

            curr_token = next_token
            curr_h = out.last_hidden_state[:, -1:, :]

        return drafted_tokens


class DSparkVerifier:
    def __init__(
        self,
        assistant_model=None,
        embed_tokens: Optional[nn.Embedding] = None,
        hidden_size: int = 2816,
        vocab_size: int = 262144,
        confidence_threshold: float = 0.75,
        max_draft_len: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_draft_len = max_draft_len
        if assistant_model is not None and embed_tokens is not None:
            self.assistant_drafter = Gemma4AssistantDrafter(
                assistant_model=assistant_model,
                embed_tokens=embed_tokens,
                max_draft_len=max_draft_len,
            )
            self.neural_drafter = None
        else:
            self.assistant_drafter = None
            self.neural_drafter = DSparkSemiAutoregressiveDrafter(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                max_draft_len=max_draft_len,
            )
        self.fallback_cache = PromptLookupCache(max_draft_len=max_draft_len)

    def propose_draft(
        self,
        tokens: List[int],
        last_hidden_state: Optional[torch.Tensor] = None,
        shared_kv: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        cur_pos: Optional[int] = None,
    ) -> List[int]:
        if self.assistant_drafter is not None and last_hidden_state is not None and shared_kv is not None and cur_pos is not None:
            draft = self.assistant_drafter.draft(
                last_token_id=tokens[-1],
                last_hidden_state=last_hidden_state,
                shared_kv=shared_kv,
                cur_pos=cur_pos,
            )
            if draft:
                return draft

        if self.neural_drafter is not None and last_hidden_state is not None:
            draft, _ = self.neural_drafter(last_hidden_state, self.confidence_threshold)
            if draft:
                return draft

        return self.fallback_cache.propose(tokens)

    def verify_tokens(
        self,
        draft_tokens: List[int],
        logits: torch.Tensor,
        temperature: float = 0.7,
    ) -> Tuple[List[int], int]:
        """
        Verifies draft tokens against logits from target model.
        Returns accepted tokens and number of matches (for KV slicing).
        """
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

        return accepted, n_matches
