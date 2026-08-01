"""Kimi Delta Attention.

Two paths over one set of parameters:

    ReferenceKDA  explicit torch, written to read like the specification
    FastKDA       the same projections, recurrence delegated to FLA

They share every `nn.Parameter`, so the equivalence tests compare two
implementations of one model rather than two models.

The one thing to get right
--------------------------
The state is decayed *before* it is read:

    S_bar = Diag(alpha_t) S_{t-1}
    v_hat = S_bar^T k_t                <- from the decayed state, not S_{t-1}
    eps   = v_t - v_hat
    S_t   = S_bar + beta_t k_t eps^T
    o_t   = (1/sqrt(64)) S_t^T q_t

Reading `v_hat` from the undecayed `S_{t-1}` gives a model that still trains and
still produces a finite loss.  It is simply a different architecture.  This is
the easiest error to make here and the hardest to notice afterwards.

Division of labour with the kernel
----------------------------------
`FastKDA` hands *pre-activation* tensors to the bridge.  Q/K L2 normalisation,
the safe-gate activation and the beta sigmoid each happen exactly once, inside
the kernel.  So this module never normalises Q/K, never applies sigmoid to
beta, never adds `dt_bias` and never multiplies by `1/sqrt(64)` on the fast
path -- doing any of them here as well would apply them twice.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kotodama.config import KDAConfig
from kotodama.fla_bridge import call_chunk_kda_checked
from kotodama.layers import RMSNorm, accumulation_dtype
from kotodama.segments import DocumentInfo


class KDACache:
    """Recurrent state plus the short-convolution history.

    Both reset together at a document start.  Clearing the state while keeping
    the convolution history leaks up to three tokens across the boundary, and
    no state-norm telemetry would show it.
    """

    def __init__(
        self,
        state: torch.Tensor | None = None,
        conv_history: tuple | None = None,
        state_layout: str = "value_key",
    ):
        self.state = state
        self.conv_history = conv_history
        # FLA with `state_v_first=True` stores [B, H, V, K]; the reference
        # stores [B, H, K, V].  Identical shapes, different axes, so the layout
        # is recorded rather than inferred from `.shape`.
        self.state_layout = state_layout

    def reset(self) -> None:
        self.state = None
        self.conv_history = None

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Reset selected batch rows at an EOD boundary.

        State and all three convolution histories must move together.  Keeping
        either one would leak information from the previous document.
        """
        rows = rows.to(dtype=torch.bool, device=(self.state.device
                                                 if self.state is not None
                                                 else rows.device))
        if not bool(rows.any()):
            return
        if bool(rows.all()):
            self.reset()
            return
        if self.state is not None:
            self.state[rows] = 0
        if self.conv_history is not None:
            self.conv_history = tuple(
                history.masked_fill(rows.view(-1, 1, 1), 0)
                for history in self.conv_history)

    def for_reference(self) -> torch.Tensor | None:
        if self.state is None:
            return None
        return self.state.transpose(-1, -2) if self.state_layout == "value_key" else self.state

    def for_fast(self) -> torch.Tensor | None:
        if self.state is None:
            return None
        return self.state.transpose(-1, -2) if self.state_layout == "key_value" else self.state


def causal_depthwise_convolution(
    x: torch.Tensor,
    weight: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution that also stops at document boundaries.

    `x` is [B, L, C], `weight` is [C, 1, K].  Position `t` may read
    `t, t-1, ..., t-K+1`, and only those inside its own document.

    Written as an explicit sum over the K taps rather than `F.conv1d`, because
    each tap needs its own document mask and K is 4.
    """
    kernel = weight.shape[-1]
    taps = weight[:, 0, :]

    offset = 0
    if history is not None:
        x = torch.cat([history, x], dim=1)
        if segment_ids is not None:
            leading = segment_ids[:, :1].expand(-1, history.shape[1])
            segment_ids = torch.cat([leading, segment_ids], dim=1)
        offset = history.shape[1]

    batch, total, channels = x.shape
    padded = torch.nn.functional.pad(x, (0, 0, kernel - 1, 0))
    padded_ids = None
    if segment_ids is not None:
        padded_ids = torch.nn.functional.pad(segment_ids, (kernel - 1, 0), value=-1)

    output = x.new_zeros((batch, total, channels))
    for tap in range(kernel):
        # tap == kernel - 1 is the current position; lower taps reach further back.
        window = padded[:, tap : tap + total, :]
        if padded_ids is not None:
            window_ids = padded_ids[:, tap : tap + total]
            window = window * (window_ids == segment_ids).unsqueeze(-1)
        output = output + window * taps[:, tap]

    keep = max(x.shape[1] - (kernel - 1), 0)
    return output[:, offset:, :], x[:, keep:, :].detach()


def reference_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    document_start_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule, on already-activated tensors.

    Separated from the projections so it can be compared directly against
    `fla.ops.kda.naive.naive_recurrent_kda`, which is the only check that
    actually establishes the algebra.  Matching gradients is supporting
    evidence, not proof: a forward pass can differ while local gradients stay
    close.

    `q`, `k`, `v` are [B, L, H, D] and post-activation; `log_decay` is
    [B, L, H, D] in log space; `beta` is [B, L, H] post-sigmoid.  State is
    [B, H, K, V].
    """
    batch, length, heads, head_dim = v.shape
    alpha = log_decay.exp()
    state = initial_state
    if state is None:
        state = torch.zeros(batch, heads, q.shape[-1], head_dim,
                            device=v.device, dtype=v.dtype)

    outputs = []
    for t in range(length):
        if document_start_mask is not None:
            state = state * (~document_start_mask[:, t]).to(state.dtype).view(batch, 1, 1, 1)
        decayed = state * alpha[:, t].unsqueeze(-1)              # 1. decay first
        predicted = (decayed * k[:, t].unsqueeze(-1)).sum(-2)    # 2. read decayed state
        error = v[:, t] - predicted                              # 3. error
        state = decayed + beta[:, t].view(batch, heads, 1, 1) * (
            k[:, t].unsqueeze(-1) * error.unsqueeze(-2))         # 4. write
        outputs.append(scale * (state * q[:, t].unsqueeze(-1)).sum(-2))  # 5. query read
    return torch.stack(outputs, dim=1), state


def safe_gate_log_decay(
    raw_gate: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """`lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`."""
    heads = raw_gate.shape[-2]
    shifted = raw_gate + dt_bias.view(heads, -1).to(raw_gate.dtype)
    return lower_bound * torch.sigmoid(
        A_log.view(heads, 1).to(raw_gate.dtype).exp() * shifted)


class KDAProjections(nn.Module):
    """Everything either path needs before the recurrence.

    2,575,948 parameters:
        q/k/v          1,769,472
        short convs        9,216
        decay low rank    98,304
        beta               9,216
        A_log + dt_bias      780
        output gate       99,072
        output norm           64
        output proj      589,824
    """

    def __init__(self, hidden_size: int, config: KDAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        heads, head_dim = config.num_heads, config.head_dim
        inner = heads * head_dim
        assert inner == hidden_size, (inner, hidden_size)

        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)

        kernel = config.short_conv_kernel_size
        self.q_conv_weight = nn.Parameter(torch.zeros(inner, 1, kernel))
        self.k_conv_weight = nn.Parameter(torch.zeros(inner, 1, kernel))
        self.v_conv_weight = nn.Parameter(torch.zeros(inner, 1, kernel))

        self.decay_down = nn.Linear(hidden_size, config.decay_rank, bias=False)
        self.decay_up = nn.Linear(config.decay_rank, inner, bias=False)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(inner))

        self.beta_proj = nn.Linear(hidden_size, heads, bias=False)

        self.gate_down = nn.Linear(hidden_size, config.decay_rank, bias=False)
        self.gate_up = nn.Linear(config.decay_rank, inner, bias=True)
        self.output_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)

        for parameter in (self.A_log, self.dt_bias, self.q_conv_weight,
                          self.k_conv_weight, self.v_conv_weight):
            parameter._no_weight_decay = True
        self.reset_kda_parameters()

    @torch.no_grad()
    def reset_kda_parameters(self) -> None:
        """Initialisations the generic pass must run before, and not overwrite."""
        self.A_log.zero_()
        # dt ~ LogUniform(1e-3, 1e-1), stored through the inverse of softplus.
        uniform = torch.rand_like(self.dt_bias)
        span = math.log(0.1) - math.log(0.001)
        dt = torch.exp(uniform * span + math.log(0.001)).clamp_min(1.0e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        for weight in (self.q_conv_weight, self.k_conv_weight, self.v_conv_weight):
            weight.zero_()
            weight[:, 0, -1] = 1.0  # identity: read only the current position
        nn.init.zeros_(self.gate_up.bias)

    def project(self, x: torch.Tensor, document_info: DocumentInfo | None,
                cache: KDACache | None):
        heads, head_dim = self.config.num_heads, self.config.head_dim
        segment_ids = document_info.segment_ids if document_info is not None else None
        histories = (cache.conv_history if cache is not None and cache.conv_history
                     else (None, None, None))

        q, q_history = causal_depthwise_convolution(
            self.q_proj(x), self.q_conv_weight, segment_ids, histories[0])
        k, k_history = causal_depthwise_convolution(
            self.k_proj(x), self.k_conv_weight, segment_ids, histories[1])
        v, v_history = causal_depthwise_convolution(
            self.v_proj(x), self.v_conv_weight, segment_ids, histories[2])
        q, k, v = (torch.nn.functional.silu(t) for t in (q, k, v))

        shape = (x.shape[0], x.shape[1], heads, head_dim)
        raw_gate = self.decay_up(self.decay_down(x)).view(shape)
        raw_beta = self.beta_proj(x)
        return (q.view(shape), k.view(shape), v.view(shape), raw_gate, raw_beta,
                (q_history, k_history, v_history))

    def finish(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Head-wise RMSNorm, sigmoid output gate, then the output projection."""
        gate = self.gate_up(self.gate_down(x))
        normalised = self.output_norm(o).reshape(x.shape[0], x.shape[1], -1)
        return self.o_proj(normalised * torch.sigmoid(gate))


class ReferenceKDA(nn.Module):
    """Explicit torch recurrence.  Written to be checkable, not fast."""

    def __init__(self, hidden_size: int, config: KDAConfig,
                 projections: KDAProjections | None = None):
        super().__init__()
        self.config = config
        self.projections = projections or KDAProjections(hidden_size, config)

    def forward(self, x: torch.Tensor, document_info: DocumentInfo | None = None,
                cache: KDACache | None = None):
        config = self.config
        work = accumulation_dtype(x.dtype)
        projections = self.projections
        q, k, v, raw_gate, raw_beta, conv_history = projections.project(
            x, document_info, cache)

        # The activations FLA would fuse, spelled out here instead.
        q = torch.nn.functional.normalize(q.to(work), dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k.to(work), dim=-1, eps=1e-6)
        v = v.to(work)
        log_decay = safe_gate_log_decay(
            raw_gate.to(work), projections.A_log.to(work),
            projections.dt_bias.to(work), config.lower_bound)
        beta = torch.sigmoid(raw_beta.to(work))

        initial_state = None
        if cache is not None and cache.state is not None:
            initial_state = cache.for_reference().to(work)
        starts = document_info.document_start_mask if document_info is not None else None
        o, state = reference_recurrence(
            q, k, v, log_decay, beta, config.scale,
            initial_state=initial_state, document_start_mask=starts)
        o = o.to(x.dtype)
        if cache is not None:
            cache.state = state
            cache.conv_history = conv_history
            cache.state_layout = "key_value"
        return projections.finish(o, x), cache


class FastKDA(nn.Module):
    """Production path.  Same parameters, recurrence inside FLA."""

    def __init__(self, hidden_size: int, config: KDAConfig,
                 projections: KDAProjections | None = None):
        super().__init__()
        self.config = config
        self.projections = projections or KDAProjections(hidden_size, config)

    def forward(self, x: torch.Tensor, document_info: DocumentInfo | None = None,
                cache: KDACache | None = None):
        projections = self.projections
        q, k, v, raw_gate, raw_beta, conv_history = projections.project(
            x, document_info, cache)

        use_cache = cache is not None
        cu_seqlens = None
        if document_info is not None and not use_cache:
            cu_seqlens = document_info.cu_seqlens.to(torch.int32)
            # The varlen kernel takes the batch flattened into one row.
            q, k, v = (t.reshape(1, -1, *t.shape[2:]) for t in (q, k, v))
            raw_gate = raw_gate.reshape(1, -1, *raw_gate.shape[2:])
            raw_beta = raw_beta.reshape(1, -1, raw_beta.shape[-1])

        output, final_state = call_chunk_kda_checked(
            q=q, k=k, v=v,
            raw_gate=raw_gate.float(), raw_beta=raw_beta.float(),
            A_log=projections.A_log.float(),
            dt_bias=projections.dt_bias.float(),
            initial_state=cache.for_fast() if use_cache else None,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        if cu_seqlens is not None:
            output = output.reshape(x.shape[0], x.shape[1], *output.shape[2:])

        if use_cache:
            cache.state = final_state
            cache.conv_history = conv_history
            cache.state_layout = "value_key"
        return projections.finish(output.to(x.dtype), x), cache


class KDALayer(nn.Module):
    """Both paths over one parameter set, selected at runtime."""

    def __init__(self, hidden_size: int, config: KDAConfig, fast: bool = True):
        super().__init__()
        self.projections = KDAProjections(hidden_size, config)
        self.reference = ReferenceKDA(hidden_size, config, self.projections)
        self.fast = FastKDA(hidden_size, config, self.projections)
        self.use_fast = fast

    def reset_kda_parameters(self) -> None:
        self.projections.reset_kda_parameters()

    def forward(self, x, document_info=None, cache=None):
        # The FLA cached kernel accepts one initial state but cannot reset it
        # inside a chunk.  An internal document boundary therefore uses the
        # explicit recurrence for that chunk.  A boundary at the first token
        # can stay on the fast path after the affected cache rows are cleared.
        internal_boundary = False
        if cache is not None and document_info is not None:
            starts = document_info.document_start_mask
            cache.reset_rows(starts[:, 0])
            internal_boundary = starts.shape[1] > 1 and bool(starts[:, 1:].any())
        path = self.fast if self.use_fast and not internal_boundary else self.reference
        return path(x, document_info=document_info, cache=cache)


__all__ = [
    "FastKDA",
    "KDACache",
    "KDALayer",
    "KDAProjections",
    "ReferenceKDA",
    "causal_depthwise_convolution",
    "reference_recurrence",
    "safe_gate_log_decay",
]
