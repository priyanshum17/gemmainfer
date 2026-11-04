from typing import Dict, Tuple, Any
import torch


class CUDAGraphRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        batch_sizes: Tuple[int, ...] = (1, 2, 4, 8),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model
        self.batch_sizes = batch_sizes
        self.device = device
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, Dict[str, torch.Tensor]] = {}
        self.static_outputs: Dict[int, Any] = {}
        self.is_cuda = (self.device.type == "cuda")

    def capture(self, max_seq_len: int = 4096):
        if not self.is_cuda:
            return

        for bs in self.batch_sizes:
            static_input_ids = torch.zeros((bs, 1), dtype=torch.long, device=self.device)
            static_pos_ids = torch.zeros((bs, 1), dtype=torch.long, device=self.device)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(static_input_ids, position_ids=static_pos_ids)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                static_logits, _ = self.model(static_input_ids, position_ids=static_pos_ids)

            self.graphs[bs] = graph
            self.static_inputs[bs] = {
                "input_ids": static_input_ids,
                "position_ids": static_pos_ids,
            }
            self.static_outputs[bs] = static_logits

    def replay(self, batch_size: int, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if not self.is_cuda or batch_size not in self.graphs:
            logits, _ = self.model(input_ids, position_ids=position_ids)
            return logits

        self.static_inputs[batch_size]["input_ids"].copy_(input_ids)
        self.static_inputs[batch_size]["position_ids"].copy_(position_ids)
        self.graphs[batch_size].replay()
        return self.static_outputs[batch_size]
