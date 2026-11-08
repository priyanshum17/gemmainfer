import logging
import uuid
from typing import List, Optional, Any
import torch

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.block_manager import BlockManager
from gemmainfer.core.scheduler import AsyncScheduler, Sequence
from gemmainfer.core.cuda_graph import CUDAGraphRunner
from gemmainfer.speculative.dspark import DSparkVerifier

logger = logging.getLogger("gemmainfer")


class GemmaInferEngine:
    """High-performance inference engine for Gemma models supporting speculative decoding."""

    def __init__(
        self,
        config: Optional[GemmaConfig] = None,
        model: Optional[GemmaForCausalLM] = None,
        assistant_model: Optional[Any] = None,
        device: Optional[str] = None,
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        enable_cuda_graphs: bool = True,
        enable_dspark: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the GemmaInfer engine with model, memory manager, and schedulers."""
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

        self.block_manager = BlockManager(
            num_blocks=num_kv_blocks,
            block_size=block_size,
            sliding_window=self.config.sliding_window,
        )
        self.scheduler = AsyncScheduler(self.block_manager, device=self.device)
        self.graph_runner = CUDAGraphRunner(self.model, device=self.device)
        self.dspark = (
            DSparkVerifier(
                assistant_model=assistant_model,
                embed_tokens=self.model.embed_tokens,
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                confidence_threshold=0.75,
                max_draft_len=5,
            )
            if enable_dspark
            else None
        )

        if enable_cuda_graphs and self.device.type == "cuda":
            self.graph_runner.capture()

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        req_id: Optional[str] = None,
    ):
        """Generate tokens in a streaming fashion using autoregressive or speculative decode."""
        req_id = req_id or str(uuid.uuid4())
        seq = Sequence(
            seq_id=req_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        past_kv = self.model.init_kv_cache(batch_size=1, max_seq_len=min(self.config.max_position_embeddings, 4096), device=self.device, dtype=self.dtype)
        logits, past_kv, last_h = self.model(
            input_ids,
            past_key_values=past_kv,
            return_logits_only_last=True,
            return_hidden_states=True,
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        seq.append_token(next_token)
        yield next_token

        eos_tokens = (1, 106)
        if next_token in eos_tokens:
            return

        curr_token_tensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        # Precompute sliding and full attention layer indices for shared_kv
        sliding_idx = None
        full_idx = None
        for i in range(len(past_kv) - 1, -1, -1):
            if self.config.is_full_attention(i):
                if full_idx is None:
                    full_idx = i
            else:
                if sliding_idx is None:
                    sliding_idx = i
            if sliding_idx is not None and full_idx is not None:
                break
        if sliding_idx is None:
            sliding_idx = len(past_kv) - 1
        if full_idx is None:
            full_idx = len(past_kv) - 1

        # Run 1 step on next_token to synchronize KV cache and hidden state
        curr_token_tensor[0, 0] = next_token
        logits, past_kv, last_h = self.model(
            curr_token_tensor,
            past_key_values=past_kv,
            return_logits_only_last=True,
            return_hidden_states=True,
        )
        predicted_next = torch.argmax(logits[:, -1, :], dim=-1).item()

        while not seq.is_finished:
            tokens_so_far = seq.prompt_token_ids + seq.output_token_ids
            orig_kv_len = past_kv[0][2][0] if isinstance(past_kv[0], tuple) and len(past_kv[0]) == 3 else past_kv[0][0].shape[2]

            if self.dspark is not None and self.dspark.assistant_drafter is not None:
                shared_kv = {
                    "sliding_attention": (past_kv[sliding_idx][0], past_kv[sliding_idx][1]),
                    "full_attention": (past_kv[full_idx][0], past_kv[full_idx][1]),
                }
                draft = self.dspark.propose_draft(
                    tokens=tokens_so_far,
                    last_hidden_state=last_h,
                    shared_kv=shared_kv,
                    cur_pos=orig_kv_len,
                )
                if draft and draft[0] == predicted_next:
                    draft_tensor = torch.tensor([draft], dtype=torch.long, device=self.device)
                    all_logits, past_kv, all_hidden = self.model(
                        draft_tensor,
                        past_key_values=past_kv,
                        return_logits_only_last=False,
                        return_hidden_states=True,
                    )

                    n_matches = 1
                    accepted = [draft[0]]
                    for i in range(len(draft) - 1):
                        target_pred_i = torch.argmax(all_logits[0, i, :], dim=-1).item()
                        if target_pred_i == draft[i + 1]:
                            accepted.append(draft[i + 1])
                            n_matches += 1
                        else:
                            accepted.append(target_pred_i)
                            break
                    else:
                        bonus = torch.argmax(all_logits[0, len(draft) - 1, :], dim=-1).item()
                        accepted.append(bonus)

                    retained_len = orig_kv_len + n_matches
                    if isinstance(past_kv[0], tuple) and len(past_kv[0]) == 3:
                        for layer_kv in past_kv:
                            layer_kv[2][0] = retained_len
                    else:
                        past_kv = [
                            (k[:, :, :retained_len, :], v[:, :, :retained_len, :])
                            for k, v in past_kv
                        ]

                    logger.debug(
                        "[CYCLE] draft=%s matches=%d accepted=%s",
                        draft,
                        n_matches,
                        accepted,
                    )

                    for tok in accepted:
                        seq.append_token(tok)
                        yield tok
                        if tok in eos_tokens or seq.is_finished:
                            return

                    # Compute next step from the newly accepted tail
                    tail_tok = accepted[-1]
                    curr_token_tensor[0, 0] = tail_tok
                    logits, past_kv, last_h = self.model(
                        curr_token_tensor,
                        past_key_values=past_kv,
                        return_logits_only_last=True,
                        return_hidden_states=True,
                    )
                    predicted_next = torch.argmax(logits[:, -1, :], dim=-1).item()
                    continue

            # Pure autoregression step
            seq.append_token(predicted_next)
            yield predicted_next
            if predicted_next in eos_tokens or seq.is_finished:
                return

            curr_token_tensor[0, 0] = predicted_next
            logits, past_kv, last_h = self.model(
                curr_token_tensor,
                past_key_values=past_kv,
                return_logits_only_last=True,
                return_hidden_states=True,
            )
            predicted_next = torch.argmax(logits[:, -1, :], dim=-1).item()

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
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
