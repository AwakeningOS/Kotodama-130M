"""Stable diagonal input injection for the recurrent core.

Parcae (arXiv:2604.12946) traces looped-model instability to large spectral
norms in the injection parameters, and constrains them by discretising a
negative diagonal parameterisation.  That is what this module implements:

    dt = softplus(dt_bias)              elementwise, strictly positive
    A  = exp(log_A)                     elementwise, strictly positive
    a  = exp(-dt * A)                   therefore strictly inside (0, 1)

    J(h, e) = a * h + dt * (e @ B.T)

`a` is the state carry-over.  Because `dt > 0` and `A > 0` by construction,
`0 < a_i < 1` holds for every channel and every parameter value the optimiser
can reach -- there is no clamp and no way to leave the stable region.

**What this does not guarantee.**  `rho(Diag(a)) < 1` bounds the *carry-over*
operator only.  The full per-iteration map is

    F(h, e) = Core(J(h, e))

and its contraction depends on the Jacobian of the shared Core, which is
unconstrained.  A stable injection does not make the loop stable.  Never write
"Parcae-style, therefore safe to iterate deeply" into a comment, a document or
an assertion: the loop can still diverge, which is why per-iteration state RMS,
relative state change, core update RMS and the validation depth sweep are
mandatory telemetry rather than optional diagnostics.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# rho_0 = sqrt(1/5): the specification's initial carry-over ratio.
INITIAL_DECAY_TARGET = math.sqrt(1.0 / 5.0)
# dt_0 = -log(rho_0)
INITIAL_DT = -math.log(INITIAL_DECAY_TARGET)
# inverse-softplus(dt_0) = dt_0 + log(-expm1(-dt_0))
INITIAL_DT_BIAS = INITIAL_DT + math.log(-math.expm1(-INITIAL_DT))


class StableDiagonalInjection(nn.Module):
    """`J(h, e) = a * h + dt * (e @ B.T)` with `a` pinned inside (0, 1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.log_A = nn.Parameter(torch.zeros(hidden_size))
        self.dt_bias = nn.Parameter(torch.zeros(hidden_size))
        self.B = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        # These parameterise a dynamical system, not a linear map to be shrunk.
        for parameter in (self.log_A, self.dt_bias, self.B):
            parameter._no_weight_decay = True
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.log_A.zero_()
        self.dt_bias.fill_(INITIAL_DT_BIAS)
        self.B.copy_(torch.eye(self.hidden_size, dtype=self.B.dtype))

    def get_dt(self) -> torch.Tensor:
        return nn.functional.softplus(self.dt_bias.float())

    def get_decay(self) -> torch.Tensor:
        """The carry-over vector `a`, always strictly inside (0, 1)."""
        return torch.exp(-self.get_dt() * self.log_A.float().exp())

    def get_spectral_radius(self) -> torch.Tensor:
        """`rho(Diag(a)) = max_i a_i`.  Bounds the carry-over only -- see the
        module docstring; it says nothing about the composed loop map."""
        return self.get_decay().max()

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        decay = self.get_decay()
        dt = self.get_dt()
        # The mix is computed in fp32: it is the one place where a small
        # per-iteration bias compounds over up to eight iterations.
        injected = nn.functional.linear(e, self.B).float()
        mixed = h.float() * decay.view(1, 1, -1) + injected * dt.view(1, 1, -1)
        return mixed.to(h.dtype)

    def closed_form_identity_core(self, e: torch.Tensor, iterations: int) -> torch.Tensor:
        """`h_n = dt * (1 - a^n) / (1 - a) * e`, for `Core = identity`, `B = I`,
        `h_0 = 0`.  Used by the unit tests to pin the implementation against an
        analytic solution rather than against itself."""
        decay = self.get_decay()
        dt = self.get_dt()
        factor = dt * (1.0 - decay**iterations) / (1.0 - decay)
        return (e.float() * factor.view(1, 1, -1)).to(e.dtype)


__all__ = ["INITIAL_DECAY_TARGET", "INITIAL_DT_BIAS", "StableDiagonalInjection"]
