import uuid
from typing import List, Optional
import torch

from gemmainfer.models.config import GemmaConfig
from gemmainfer.models.gemma import GemmaForCausalLM
from gemmainfer.core.block_manager import BlockManager
from gemmainfer.core.scheduler import AsyncScheduler, Sequence
from gemmainfer.core.cuda_graph import CUDAGraphRunner
from gemmainfer.speculative.dspark import DSparkVerifier


class GemmaInferEngine:
    def __init__(
        self,
        config: Optional[GemmaConfig] = None,
        model: Optional[GemmaForCausalLM] = None,
        device: Optional[str] = None,
        num_kv_blocks: int = 1024,
        block_size: int = 16,
        enable_cuda_graphs: bool = True,
        enable_dspark: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
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
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
            )
            if enable_dspark
            else None
        )

        if enable_cuda_graphs and self.device.type == "cuda":
            self.graph_runner.capture()

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        req_id: Optional[str] = None,
    ) -> List[int]:
        req_id = req_id or str(uuid.uuid4())
        seq = Sequence(
            seq_id=req_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        logits, past_kv = self.model(input_ids, return_logits_only_last=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        seq.append_token(next_token)

        while not seq.is_finished:
            tokens_so_far = seq.prompt_token_ids + seq.output_token_ids

            if self.dspark is not None:
                draft = self.dspark.propose_draft(tokens_so_far)
                if draft:
                    orig_kv_len = past_kv[0][0].shape[2]
                    draft_tensor = torch.tensor(draft, dtype=torch.long, device=self.device).unsqueeze(0)
                    all_logits, past_kv = self.model(
                        draft_tensor,
                        past_key_values=past_kv,
                        return_logits_only_last=False,
                    )
                    accepted, count = self.dspark.verify_tokens(
                        draft_tokens=draft,
                        logits=all_logits[0],
                        temperature=temperature,
                    )

                    retained_len = min(orig_kv_len + count, past_kv[0][0].shape[2])
                    past_kv = [
                        (k[:, :, :retained_len, :], v[:, :, :retained_len, :])
                        for k, v in past_kv
                    ]

                    for tok in accepted:
                        seq.append_token(tok)
                        if seq.is_finished:
                            break
                    continue

            curr_token_tensor = torch.tensor([[seq.output_token_ids[-1]]], dtype=torch.long, device=self.device)
            logits, past_kv = self.model(
                curr_token_tensor,
                past_key_values=past_kv,
                return_logits_only_last=True,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            seq.append_token(next_token)

        return seq.output_token_ids
