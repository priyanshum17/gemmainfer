from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from gemmainfer.kernels.quant import MarlinLinear

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False


def _make_mlp_linear(in_features: int, out_features: int, quant_method: Optional[str] = None, group_size: int = 64) -> nn.Module:
    """Helper to instantiate dense or Marlin quantized Linear layer for MLP blocks."""
    if quant_method == "w4a16":
        return MarlinLinear(in_features, out_features, group_size=group_size, bias=False)
    return nn.Linear(in_features, out_features, bias=False)


class GemmaMLP(nn.Module):
    """Gemma feed-forward network with gated GeLU activation and optional W4A16 quantization."""

    def __init__(self, hidden_size: int, intermediate_size: int, quant_method: Optional[str] = None, group_size: int = 64):
        super().__init__()
        self.gate_proj = _make_mlp_linear(hidden_size, intermediate_size, quant_method=quant_method, group_size=group_size)
        self.up_proj = _make_mlp_linear(hidden_size, intermediate_size, quant_method=quant_method, group_size=group_size)
        self.down_proj = _make_mlp_linear(intermediate_size, hidden_size, quant_method=quant_method, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute gated MLP forward pass."""
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def apply_logit_softcapping(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Apply hyperbolic tangent logit soft-capping using CUDA kernel or PyTorch fallback."""
    if _HAS_CSRC and logits.is_contiguous():
        gemmainfer_csrc.softcapping(logits, cap)
        return logits
    return cap * torch.tanh(logits / cap)
