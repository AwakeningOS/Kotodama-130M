"""RMSNorm and the SwiGLU feed-forward network.

The product feed-forward path uses the established SwiGLU equation:

    FFN(x) = W_down [ SiLU(W_gate x) * W_up x ]

There is deliberately no rotary embedding here.  This architecture is strict
NoPE -- order information reaches the attention blocks through the KDA
recurrence, in the residual stream, as content.  Keeping an unused
`RotaryEmbedding` class around "just in case" is the pattern the specification
forbids, so it is gone rather than disabled.
"""

from __future__ import annotations

import torch
from torch import nn


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """At least float32, but never downcast a higher-precision input.

    Plain `.float()` silently makes float64 reference tests lossy, which hides
    real algebraic errors behind what looks like rounding noise.
    """
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        # A norm gain is a scale, not a weight to be shrunk towards zero.
        self.weight._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x.dtype
        work = accumulation_dtype(original)
        xw = x.to(work)
        xw = xw * torch.rsqrt(xw.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xw * self.weight.to(work)).to(original)


class SwiGLUMLP(nn.Module):
    """768 -> 2272 -> 768, no biases, 5,234,688 parameters.

    `down_proj.weight` is zero-initialised by the model's initialisation pass,
    so a freshly built block is an exact identity rather than an approximate
    one.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "accumulation_dtype",
]
