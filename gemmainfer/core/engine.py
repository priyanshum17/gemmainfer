import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Any
import torch
import torch.nn.functional as F

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.speculative.dspark import DSparkVerifier

logger = logging.getLogger("gemmainfer")


@dataclass
class Sequence:
    """State container tracking prompt tokens, generated tokens, and sampling parameters."""

    seq_id: str
    prompt_token_ids: List[int]
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    output_token_ids: List[int] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        if len(self.output_token_ids) >= self.max_tokens:
            return True
        if self.output_token_ids and self.output_token_ids[-1] in (1, 106):
            return True
        return False

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)


def _sample_token(logits_1d: torch.Tensor, temperature: float = 0.0, top_p: float = 0.95) -> int:
    """Sample a token from 1D logits using greedy decoding or top-p nucleus sampling."""
    if temperature <= 1e-5:
        return int(torch.argmax(logits_1d, dim=-1).item())
    scaled = logits_1d.float() / temperature
    probs = F.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = (cum_probs - sorted_probs) > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices[sampled_idx].item())
    return int(torch.multinomial(probs, num_samples=1).item())


class GemmaInferEngine:
    """High-performance inference engine for Gemma 4 with single-pass speculative verification."""

    def __init__(
        self,
        config: Optional[GemmaConfig] = None,
        model: Optional[GemmaForCausalLM] = None,
        assistant_model: Optional[Any] = None,
        device: Optional[str] = None,
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        enable_cuda_graphs: bool = False,
        enable_dspark: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the GemmaInfer engine with model and speculative verifier."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config or GemmaConfig.gemma_4_26b_a4b()
        self.dtype = dtype

        if model is not None:
            self.model = model.to(device=self.device, dtype=self.dtype)
        else:
            self.model = GemmaForCausalLM(self.config).to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.dspark = (
            DSparkVerifier(
                assistant_model=assistant_model,
                embed_tokens=self.model.embed_tokens,
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                confidence_threshold=0.25,
                max_draft_len=4,
            )
            if enable_dspark
            else None
        )

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.95,
        req_id: Optional[str] = None,
    ):
        """
        Generate tokens in a streaming fashion using a unified single-pass
        teacher-forced speculative verification loop (or pure autoregression).
        """
        req_id = req_id or str(uuid.uuid4())
        seq = Sequence(
            seq_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        max_cache_len = min(self.config.max_position_embeddings, max(4096, len(prompt_token_ids) + max_tokens + 64))
        past_kv = self.model.init_kv_cache(
            batch_size=1,
            max_seq_len=max_cache_len,
            device=self.device,
            dtype=self.dtype,
        )

        # Prefill prompt tokens [0 .. L-1]:
        # - past_kv now holds KV states for positions [0 .. L-1] (length L)
        # - last_h holds the target hidden state at position L-1
        # - bonus_tok is the target prediction for position L (yielded immediately)
        logits, past_kv, last_h = self.model(
            input_ids,
            past_key_values=past_kv,
            return_logits_only_last=True,
            return_hidden_states=True,
        )
        bonus_tok = _sample_token(logits[0, -1, :], temperature=temperature, top_p=top_p)
        seq.append_token(bonus_tok)
        yield bonus_tok

        eos_tokens = (1, 106)
        if bonus_tok in eos_tokens or seq.is_finished:
            return

        # Precompute last sliding_attention and full_attention layer indices for cross-model KV sharing
        sliding_idx = -1
        full_idx = -1
        for i in range(len(past_kv) - 1, -1, -1):
            if self.config.is_full_attention(i) and full_idx == -1:
                full_idx = i
            elif not self.config.is_full_attention(i) and sliding_idx == -1:
                sliding_idx = i
            if sliding_idx != -1 and full_idx != -1:
                break
        if sliding_idx == -1:
            sliding_idx = len(past_kv) - 1
        if full_idx == -1:
            full_idx = len(past_kv) - 1

        base_draft_len = self.dspark.max_draft_len if self.dspark is not None else 4
        cur_draft_len = base_draft_len

        while not seq.is_finished:
            tokens_so_far = seq.prompt_token_ids + seq.output_token_ids
            orig_logical_len = past_kv[0][2][0]

            # Draft continuation tokens [d_1, ..., d_k] that follow bonus_tok
            draft_cont: List[int] = []
            if self.dspark is not None:
                if self.dspark.assistant_drafter is not None:
                    self.dspark.assistant_drafter.max_draft_len = cur_draft_len
                    s_kv = past_kv[sliding_idx]
                    f_kv = past_kv[full_idx]
                    s_len = s_kv[2][1] if len(s_kv[2]) > 1 else s_kv[2][0]
                    f_len = f_kv[2][1] if len(f_kv[2]) > 1 else f_kv[2][0]
                    shared_kv = {
                        "sliding_attention": (s_kv[0][:, :, :s_len, :], s_kv[1][:, :, :s_len, :]),
                        "full_attention": (f_kv[0][:, :, :f_len, :], f_kv[1][:, :, :f_len, :]),
                    }
                    draft_cont = self.dspark.propose_draft(
                        tokens=tokens_so_far,
                        last_hidden_state=last_h,
                        shared_kv=shared_kv,
                        cur_pos=orig_logical_len,
                        forced_first_token=bonus_tok,
                    )
                else:
                    draft_cont = self.dspark.fallback_cache.propose(tokens_so_far)

            # Single-pass verification batch: [bonus_tok, d_1, ..., d_k]
            verify_list = [bonus_tok] + draft_cont
            verify_tensor = torch.tensor([verify_list], dtype=torch.long, device=self.device)
            num_verify = len(verify_list)

            need_hidden = self.dspark is not None and self.dspark.assistant_drafter is not None
            if need_hidden:
                all_logits, past_kv, all_hidden = self.model(
                    verify_tensor,
                    past_key_values=past_kv,
                    return_logits_only_last=(num_verify == 1),
                    return_hidden_states=True,
                )
            else:
                all_logits, past_kv = self.model(
                    verify_tensor,
                    past_key_values=past_kv,
                    return_logits_only_last=(num_verify == 1),
                    return_hidden_states=False,
                )
                all_hidden = None

            # Verify draft_cont against target logits at positions [0 .. num_verify-1]
            n_matches = 1  # verify_list[0] == bonus_tok is already guaranteed valid
            newly_accepted: List[int] = []

            if temperature <= 1e-5:
                preds = torch.argmax(all_logits[0], dim=-1).tolist()
                for i in range(len(draft_cont)):
                    pred_i = preds[i]
                    if pred_i == draft_cont[i]:
                        newly_accepted.append(draft_cont[i])
                        n_matches += 1
                    else:
                        newly_accepted.append(pred_i)
                        break
                else:
                    newly_accepted.append(preds[num_verify - 1])
            else:
                for i in range(len(draft_cont)):
                    pred_i = _sample_token(all_logits[0, i, :], temperature=temperature, top_p=top_p)
                    if pred_i == draft_cont[i]:
                        newly_accepted.append(draft_cont[i])
                        n_matches += 1
                    else:
                        newly_accepted.append(pred_i)
                        break
                else:
                    next_bonus = _sample_token(all_logits[0, num_verify - 1, :], temperature=temperature, top_p=top_p)
                    newly_accepted.append(next_bonus)

            # Roll back rejected speculative slots in KV cache metadata (zero-copy rollback)
            dropped = num_verify - n_matches
            if dropped > 0:
                for layer_kv in past_kv:
                    layer_kv[2][0] -= dropped
                    if len(layer_kv[2]) > 1:
                        layer_kv[2][1] -= dropped

            if draft_cont:
                logger.debug(
                    "[CYCLE] verify=%s matches=%d newly_accepted=%s",
                    verify_list,
                    n_matches,
                    newly_accepted,
                )

            for tok in newly_accepted:
                seq.append_token(tok)
                yield tok
                if tok in eos_tokens or seq.is_finished:
                    return

            # Adaptive draft horizon: expand to 6 on full-match streaks, reset to base_draft_len on rejection
            if num_verify > 1 and n_matches == num_verify:
                cur_draft_len = min(6, cur_draft_len + 2)
            else:
                cur_draft_len = base_draft_len

            # Synchronize state for the next cycle directly from the verification pass (ZERO extra forward passes!)
            if all_hidden is not None:
                last_h = all_hidden[:, n_matches - 1 : n_matches, :]
            bonus_tok = newly_accepted[-1]

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.95,
        req_id: Optional[str] = None,
    ) -> List[int]:
        """Generate output tokens synchronously for a given prompt token sequence."""
        return list(
            self.generate_stream(
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                req_id=req_id,
            )
        )
