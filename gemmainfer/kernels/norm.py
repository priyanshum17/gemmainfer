import torch
import torch.nn as nn

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim) if add_unit_offset else torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_CSRC and x.is_contiguous() and self.weight.is_contiguous():
            out = torch.empty_like(x)
            gemmainfer_csrc.rmsnorm(x, self.weight, out, self.eps, self.add_unit_offset)
            return out

        normed = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return normed * (1.0 + self.weight)
        return normed * self.weight
