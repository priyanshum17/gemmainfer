from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
import torch

from gemmainfer.core.block_manager import BlockManager


@dataclass
class Sequence:
    """Represents an active generation request with prompt, output tokens, and sampling configuration."""

    seq_id: str
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    is_finished: bool = False

    @property
    def total_len(self) -> int:
        """Return total sequence length including prompt and generated tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_token(self, token_id: int):
        """Append newly sampled token and update finish state if max_tokens is reached."""
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


@dataclass
class BatchDescriptor:
    """Metadata describing a scheduled batch of sequences for prefill or decode execution."""

    seqs: List[Sequence]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    is_prefill: bool


class AsyncScheduler:
    """Asynchronous request scheduler managing continuous batching for prefill and decode phases."""

    def __init__(self, block_manager: BlockManager, device: torch.device):
        self.block_manager = block_manager
        self.device = device
        self.waiting_queue: Deque[Sequence] = deque()
        self.running_batch: List[Sequence] = []
        self.h2d_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def add_request(self, seq: Sequence):
        """Enqueue an incoming sequence into the waiting queue."""
        self.waiting_queue.append(seq)

    def schedule_next_batch(self, max_batch_size: int = 8) -> Optional[BatchDescriptor]:
        """Schedule the next batch for execution, prioritizing running decodes before prefills."""
        finished = [s for s in self.running_batch if s.is_finished]
        for s in finished:
            self.block_manager.free_sequence(s.seq_id)
            self.running_batch.remove(s)

        if self.running_batch:
            batch = self.running_batch[:max_batch_size]
            tokens = [s.output_token_ids[-1] if s.output_token_ids else s.prompt_token_ids[-1] for s in batch]
            positions = [s.total_len - 1 for s in batch]

            input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            pos_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)

            for s in batch:
                self.block_manager.append_token(s.seq_id, s.total_len)

            return BatchDescriptor(seqs=batch, input_ids=input_ids, position_ids=pos_ids, is_prefill=False)

        if self.waiting_queue:
            seq = self.waiting_queue.popleft()
            self.block_manager.allocate_sequence(seq.seq_id, len(seq.prompt_token_ids))
            self.running_batch.append(seq)

            input_ids = torch.tensor(seq.prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            pos_ids = torch.arange(len(seq.prompt_token_ids), dtype=torch.long, device=self.device).unsqueeze(0)

            return BatchDescriptor(seqs=[seq], input_ids=input_ids, position_ids=pos_ids, is_prefill=True)

        return None
