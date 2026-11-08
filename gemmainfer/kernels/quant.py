from typing import Tuple, Optional
import torch
import torch.nn as nn

try:
    import gemmainfer_csrc
    _HAS_CSRC = True
except ImportError:
    _HAS_CSRC = False


def pack_to_marlin_w4a16(
    weight: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.t().contiguous()
    K, N = w.shape
    assert K % group_size == 0
    assert K % 8 == 0

    num_groups = K // group_size
    w_reshaped = w.view(num_groups, group_size, N)

    max_val = w_reshaped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = (max_val / 7.0).squeeze(1).to(weight.dtype)

    scales_expanded = scales.unsqueeze(1).expand_as(w_reshaped)
    w_quant = torch.clamp(torch.round(w_reshaped / scales_expanded) + 8, 0, 15).to(torch.int32)
    w_quant = w_quant.view(K, N)

    packed_K = K // 8
    packed_weight = torch.zeros((packed_K, N), dtype=torch.int32, device=weight.device)

    for i in range(8):
        packed_weight |= (w_quant[i::8, :] & 0x0F) << (i * 4)

    return packed_weight, scales


def marlin_forward(
    x_2d: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out_features: int,
) -> torch.Tensor:
    M = x_2d.size(0)
    out = torch.empty((M, out_features), dtype=x_2d.dtype, device=x_2d.device)
    if _HAS_CSRC and x_2d.is_cuda:
        gemmainfer_csrc.marlin_w4a16(
            x_2d,
            packed_weight,
            scales,
            out,
            group_size,
        )
    else:
        K = x_2d.size(1)
        w_unpacked = torch.empty((K, out_features), dtype=x_2d.dtype, device=x_2d.device)
        for i in range(8):
            val = ((packed_weight >> (i * 4)) & 0x0F) - 8
            group_idx = (torch.arange(K // 8, device=x_2d.device) * 8 + i) // group_size
            scale = scales[group_idx, :]
            w_unpacked[i::8, :] = (val.to(x_2d.dtype) * scale)
        out = torch.matmul(x_2d, w_unpacked)
    return out


class MarlinLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dtype = dtype

        self.register_buffer(
            "packed_weight",
            torch.zeros((in_features // 8, out_features), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.full((in_features // group_size, out_features), 0.02, dtype=dtype),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "MarlinLinear":
        marlin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )
        packed, scales = pack_to_marlin_w4a16(linear.weight, group_size=group_size)
        marlin.packed_weight.copy_(packed)
        marlin.scales.copy_(scales)
        if linear.bias is not None:
            marlin.bias.data.copy_(linear.bias.data)
        return marlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features).contiguous()
        out = marlin_forward(x_2d, self.packed_weight, self.scales, self.group_size, self.out_features)
        if self.bias is not None:
            out += self.bias
        return out.view(*orig_shape[:-1], self.out_features)
