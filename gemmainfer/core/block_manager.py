from typing import List, Dict, Tuple


class Block:
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0

    def reset(self):
        self.ref_count = 0


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        sliding_window: int = 4096,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.max_swa_blocks = (sliding_window + block_size - 1) // block_size

        self.free_blocks: List[int] = list(range(num_blocks))
        self.blocks = [Block(i, block_size) for i in range(num_blocks)]
        
        self.req_to_global_blocks: Dict[str, List[int]] = {}
        self.req_to_swa_blocks: Dict[str, List[int]] = {}

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        num_required = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_required

    def allocate_sequence(self, req_id: str, num_tokens: int) -> Tuple[List[int], List[int]]:
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError(f"Insufficient KV blocks: requested {num_blocks_needed}, free {len(self.free_blocks)}")

        global_blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        for b_id in global_blocks:
            self.blocks[b_id].ref_count += 1
        self.req_to_global_blocks[req_id] = global_blocks

        swa_blocks_needed = min(num_blocks_needed, self.max_swa_blocks)
        swa_blocks = [self.free_blocks.pop() for _ in range(swa_blocks_needed)]
        for b_id in swa_blocks:
            self.blocks[b_id].ref_count += 1
        self.req_to_swa_blocks[req_id] = swa_blocks

        return global_blocks, swa_blocks

    def append_token(self, req_id: str, cur_seq_len: int) -> Tuple[bool, bool]:
        allocated_global = False
        allocated_swa = False

        if cur_seq_len % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("KV block pool exhausted during step")
            new_b = self.free_blocks.pop()
            self.blocks[new_b].ref_count += 1
            self.req_to_global_blocks[req_id].append(new_b)
            allocated_global = True

        swa_blocks = self.req_to_swa_blocks[req_id]
        if cur_seq_len % self.block_size == 0:
            if len(swa_blocks) < self.max_swa_blocks:
                new_b = self.free_blocks.pop()
                self.blocks[new_b].ref_count += 1
                swa_blocks.append(new_b)
                allocated_swa = True

        return allocated_global, allocated_swa

    def free_sequence(self, req_id: str):
        if req_id in self.req_to_global_blocks:
            for b_id in self.req_to_global_blocks[req_id]:
                self.blocks[b_id].ref_count -= 1
                if self.blocks[b_id].ref_count == 0:
                    self.free_blocks.append(b_id)
            del self.req_to_global_blocks[req_id]

        if req_id in self.req_to_swa_blocks:
            for b_id in self.req_to_swa_blocks[req_id]:
                self.blocks[b_id].ref_count -= 1
                if self.blocks[b_id].ref_count == 0:
                    self.free_blocks.append(b_id)
            del self.req_to_swa_blocks[req_id]
