import torch
import torch.nn as nn

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False

_EMPTY_WEIGHT = torch.empty(0)


class GemmaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) with optional unit offset and CUDA acceleration."""

    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.zeros(dim) if add_unit_offset else torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Internal FP32 numerical root-mean-square normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to tensor x."""
        if _HAS_CSRC and x.is_contiguous():
            if not self.with_scale or self.weight is None:
                if x.is_cuda:
                    out = torch.empty_like(x)
                    gemmainfer_csrc.rmsnorm(x, _EMPTY_WEIGHT, out, self.eps, False)
                    return out
                return self._norm(x.float()).type_as(x)
            if self.weight.is_contiguous():
                out = torch.empty_like(x)
                gemmainfer_csrc.rmsnorm(x, self.weight, out, self.eps, self.add_unit_offset)
                return out

        if not self.with_scale or self.weight is None:
            return self._norm(x.float()).type_as(x)

        normed = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            return normed * (1.0 + self.weight)
        return normed * self.weight
