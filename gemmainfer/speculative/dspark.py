from typing import List, Tuple, Optional
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


class DSparkVerifier:
    def __init__(
        self,
        hidden_size: int = 2816,
        vocab_size: int = 262144,
        confidence_threshold: float = 0.75,
        max_draft_len: int = 4,
    ):
        self.confidence_threshold = confidence_threshold
        self.drafter = DSparkSemiAutoregressiveDrafter(hidden_size, vocab_size, max_draft_len)
        self.fallback_cache = PromptLookupCache(max_draft_len=max_draft_len)

    def propose_draft(
        self,
        tokens: List[int],
        last_hidden_state: Optional[torch.Tensor] = None,
    ) -> List[int]:
        if last_hidden_state is not None:
            draft, _ = self.drafter(last_hidden_state, self.confidence_threshold)
            if draft:
                return draft
        return self.fallback_cache.propose(tokens)

    def verify_tokens(
        self,
        draft_tokens: List[int],
        logits: torch.Tensor,
        temperature: float = 0.7,
    ) -> Tuple[List[int], int]:
        K = len(draft_tokens)
        accepted = [draft_tokens[0]]

        for i in range(K - 1):
            target_candidate = draft_tokens[i + 1]
            pos_logits = logits[i] / max(temperature, 1e-5)
            probs = F.softmax(pos_logits, dim=-1)

            confidence = probs[target_candidate].item()
            best_token = torch.argmax(probs).item()

            if best_token == target_candidate or confidence >= self.confidence_threshold:
                accepted.append(target_candidate)
            else:
                accepted.append(best_token)
                return accepted, len(accepted)

        last_logits = logits[K - 1] / max(temperature, 1e-5)
        bonus_token = torch.argmax(last_logits).item()
        accepted.append(bonus_token)

        return accepted, len(accepted)
