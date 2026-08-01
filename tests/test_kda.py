"""KDA verification, in two tiers that must not be conflated.

**A0. Independent algebraic oracle**, float64, tolerance 1e-10.  The equations
are transcribed again inside this file, per head, using `outer` and matrix
products rather than the production broadcast-and-sum form, and closed forms
for the first two steps are written out by hand.  It never calls
`reference_recurrence`, because a function compared against itself proves
nothing.  This is the gate that establishes the algebra.

**A1. FLA-naive compatibility gate**, tolerances 2e-6 / 2e-5.  This is *not* an
algebraic gate.  `naive_recurrent_kda` hard-casts q/k/v/g/beta to float32
regardless of input dtype, casts only the output back, and returns an
FP32-accumulated final state -- so it cannot resolve below float32 rounding no
matter what is fed to it.  These numbers measure agreement with a pinned
third-party implementation, nothing more.

Matching gradients is *supporting* evidence, not proof: a forward pass can
differ while local gradients stay close.

**B. Production-kernel precision gate.**  FLA's naive against FLA's chunked
Triton kernel, and then the full `ReferenceKDA` layer against `FastKDA`.  These
disagree by more than the algebraic tolerance because of the numerical path
inside the chunked kernel -- a TF32/TensorCore route is a confirmed contributing
factor, but the observed difference is not established as attributable to TF32
alone; chunking, WY transforms, `exp2` and intermediate round-trips all sit on
that path too.  So this tier records an error *envelope* rather than a value.

Envelopes are not bitwise snapshots: the last few bits move with GPU, driver,
CUDA, Triton and FLA.  When the environment fingerprint changes, both tiers are
re-run rather than trusted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.config import KDAConfig  # noqa: E402
from kotodama.fla_bridge import call_chunk_kda_checked  # noqa: E402
from kotodama.kda import (  # noqa: E402
    KDALayer,
    KDAProjections,
    causal_depthwise_convolution,
    reference_recurrence,
    safe_gate_log_decay,
)
from kotodama.segments import build_document_info  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEADS, HEAD_DIM = 12, 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)
LOWER_BOUND = -5.0

SEQUENCE_LENGTHS = (16, 64, 256, 512)
SEEDS = (3, 11, 29)
GATE_REGIMES = ("normal", "near_zero", "near_lower_bound")

INITIAL_STATES = ("none", "nonzero")

# Tier B envelopes.  Widening these requires a section 26 report, not an edit.
ENVELOPE_REL_L2 = 5.0e-3
ENVELOPE_SLOPE = 5.0e-3
# Not a concession for large tensors: a floor so the ratio cannot explode when
# the reference is almost entirely zero.
ENVELOPE_FLOOR = 1.0e-5
GRAD_MAX_ABS = 5.0e-4
GRAD_REL_L2 = 5.0e-3

# Tier A0: an oracle written independently in this file.  float64 throughout.
ORACLE_MAX_ABS = 1.0e-10

# Tier A1.  FLA `naive_recurrent_kda` is not an FP64 oracle: it hard-casts
# q/k/v/g/beta to torch.float32 internally.  The output is cast back to the
# input dtype, but the returned final state stays the FP32-accumulated one.
#
# These tolerances therefore measure compatibility with FLA's pinned FP32 naive
# implementation, not the algebraic accuracy of our FP64 recurrence.  Exact
# algebra is checked by Tier A0.  Observed maxima over the 72 conditions are
# 3.73e-7 (output) and 3.36e-6 (state), so the margin is 5.4x and 6.0x, while a
# genuine ordering bug moves the result by more than 1e-4.
#
# Re-measure -- do not reuse -- if any of these change:
#   FLA source SHA256, naive.py SHA256, torch version, CUDA version, GPU
#   architecture.  If FLA ever preserves float64, rebuild this gate tighter
#   rather than keeping these numbers.
FLA_NAIVE_OUTPUT_MAX_ABS = 2.0e-6
FLA_NAIVE_STATE_MAX_ABS = 2.0e-5
FLA_NAIVE_GRAD_MAX_ABS = 1.0e-6


def assert_kernel_envelope(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Scale-invariant acceptance for the production kernel.

    A fixed absolute bound measures the amplitude of the reference tensor, not
    the accuracy of the kernel.  Under a near-zero decay gate the recurrent
    state grows about threefold and the absolute error grows with it, while the
    relative error stays at 1.1e-3 in every regime -- so the fixed bound failed
    on a kernel that had not got any worse.  During training the state
    amplitude moves continuously, which would make a fixed bound drift between
    false alarms and blind spots.
    """
    difference = actual.float() - reference.float()
    max_abs = difference.abs().max()
    reference_max_abs = reference.float().abs().max()
    rms = difference.square().mean().sqrt()
    rel_l2 = (torch.linalg.vector_norm(difference)
              / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-12))
    normalized_max_abs = max_abs / reference_max_abs.clamp_min(1.0e-12)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(reference).all()
    assert rel_l2.item() < ENVELOPE_REL_L2, rel_l2.item()
    assert max_abs.item() <= ENVELOPE_FLOOR + ENVELOPE_SLOPE * reference_max_abs.item(), (
        max_abs.item(), reference_max_abs.item())

    return {
        "max_abs": max_abs.item(),
        "rms": rms.item(),
        "rel_l2": rel_l2.item(),
        "reference_max_abs": reference_max_abs.item(),
        "normalized_max_abs": normalized_max_abs.item(),
    }


def error_metrics(fast: torch.Tensor, reference: torch.Tensor) -> dict:
    """Recorded for gradients, where the envelope contract is unchanged."""
    difference = (fast.float() - reference.float())
    denominator = max(reference.float().norm().item(), 1.0e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "rel_l2": difference.norm().item() / denominator,
    }


def make_activated_inputs(length: int, seed: int, regime: str, batch: int = 1,
                          device: str = "cuda", dtype: torch.dtype = torch.float32):
    """Post-activation tensors, shared by both implementations.

    Tier A runs in float64.  In float32 the two pure-torch implementations
    differ by up to 2.4e-7 on the final state -- which is 2**-22, the float32
    epsilon, produced by a different reduction order rather than by different
    algebra.  Raising the tolerance would hide real errors of the same size;
    raising the precision removes the confound instead, and the specification
    already provides for a float64 reference tier.
    """
    torch.manual_seed(seed)
    shape = (batch, length, HEADS, HEAD_DIM)
    q = torch.nn.functional.normalize(
        torch.randn(shape, device=device, dtype=dtype), dim=-1, eps=1e-6)
    k = torch.nn.functional.normalize(
        torch.randn(shape, device=device, dtype=dtype), dim=-1, eps=1e-6)
    v = torch.randn(shape, device=device, dtype=dtype)
    raw_gate = torch.randn(shape, device=device, dtype=dtype)
    if regime == "near_zero":
        # sigmoid saturates high -> log_decay near lower_bound is avoided;
        # large negative pushes log_decay towards 0 (almost no forgetting).
        raw_gate = raw_gate - 8.0
    elif regime == "near_lower_bound":
        raw_gate = raw_gate + 8.0
    A_log = torch.zeros(HEADS, device=device, dtype=dtype)
    dt_bias = torch.zeros(HEADS * HEAD_DIM, device=device, dtype=dtype)
    log_decay = safe_gate_log_decay(raw_gate, A_log, dt_bias, LOWER_BOUND)
    beta = torch.sigmoid(torch.randn((batch, length, HEADS), device=device, dtype=dtype))
    return q, k, v, raw_gate, log_decay, beta, A_log, dt_bias


def make_initial_state(name: str, batch: int, seed: int, device: str = "cuda",
                       dtype: torch.dtype = torch.float32):
    if name == "none":
        return None
    torch.manual_seed(seed + 1000)
    return torch.randn(batch, HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=dtype) * 0.1


# 4 lengths x 3 seeds x 3 gate regimes x 2 initial states = 72 conditions.
ALL_CONDITIONS = [
    (length, seed, regime, state)
    for length in SEQUENCE_LENGTHS
    for seed in SEEDS
    for regime in GATE_REGIMES
    for state in INITIAL_STATES
]


# --------------------------------------------------------------------------
# Tier A0: independent oracle.  Written from the equations, not from the code.
# --------------------------------------------------------------------------


def oracle_recurrence(q, k, v, log_decay, beta, scale, initial_state=None):
    """The delta rule again, deliberately in a different shape.

    Per head, with `S` held as an explicit [K, V] matrix and updates written as
    `torch.outer` and `S.t() @ k`.  The production version broadcasts and sums
    over a batched axis; agreement between the two forms is evidence, whereas
    agreement between one form and itself is not.
    """
    batch, length, heads, head_dim = v.shape
    alpha = log_decay.exp()
    output = torch.zeros_like(v)
    finals = torch.zeros(batch, heads, q.shape[-1], head_dim,
                         device=v.device, dtype=v.dtype)
    for b in range(batch):
        for h in range(heads):
            state = (torch.zeros(q.shape[-1], head_dim, device=v.device, dtype=v.dtype)
                     if initial_state is None else initial_state[b, h].clone())
            for t in range(length):
                decayed = torch.diag(alpha[b, t, h]) @ state
                predicted = decayed.t() @ k[b, t, h]
                error = v[b, t, h] - predicted
                state = decayed + beta[b, t, h] * torch.outer(k[b, t, h], error)
                output[b, t, h] = scale * (state.t() @ q[b, t, h])
            finals[b, h] = state
    return output, finals


@CUDA
@pytest.mark.parametrize("regime", GATE_REGIMES)
@pytest.mark.parametrize("state_name", INITIAL_STATES)
def test_production_recurrence_matches_independent_oracle(regime, state_name):
    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        12, 3, regime, dtype=torch.float64)
    state = make_initial_state(state_name, q.shape[0], 3, dtype=torch.float64)
    expected, expected_state = oracle_recurrence(
        q, k, v, log_decay, beta, SCALE, initial_state=state)
    actual, actual_state = reference_recurrence(
        q, k, v, log_decay, beta, SCALE, initial_state=state)
    assert (actual - expected).abs().max().item() < ORACLE_MAX_ABS
    assert (actual_state - expected_state).abs().max().item() < ORACLE_MAX_ABS


@CUDA
def test_single_step_closed_form():
    """From a zero state: `o_0 = scale * beta_0 * (k_0 . q_0) * v_0`."""
    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        1, 3, "normal", dtype=torch.float64)
    actual, actual_state = reference_recurrence(q, k, v, log_decay, beta, SCALE)
    for h in range(HEADS):
        expected = SCALE * beta[0, 0, h] * torch.dot(k[0, 0, h], q[0, 0, h]) * v[0, 0, h]
        assert (actual[0, 0, h] - expected).abs().max().item() < ORACLE_MAX_ABS
        expected_state = beta[0, 0, h] * torch.outer(k[0, 0, h], v[0, 0, h])
        assert (actual_state[0, h] - expected_state).abs().max().item() < ORACLE_MAX_ABS


@CUDA
def test_two_step_closed_form():
    """Second step written out by hand, including the decayed read-back."""
    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        2, 11, "normal", dtype=torch.float64)
    actual, _ = reference_recurrence(q, k, v, log_decay, beta, SCALE)
    alpha = log_decay.exp()
    for h in range(HEADS):
        k0, k1, v0, v1, q1 = (k[0, 0, h], k[0, 1, h], v[0, 0, h], v[0, 1, h], q[0, 1, h])
        b0, b1, a1 = beta[0, 0, h], beta[0, 1, h], alpha[0, 1, h]
        # S_1 = b0 * outer(k0, v0); decayed row-wise by a1 before being read.
        predicted = b0 * torch.dot(a1 * k0, k1) * v0
        error = v1 - predicted
        expected = SCALE * (b0 * torch.dot(a1 * k0, q1) * v0
                            + b1 * torch.dot(k1, q1) * error)
        assert (actual[0, 1, h] - expected).abs().max().item() < ORACLE_MAX_ABS


@CUDA
def test_first_step_uses_the_initial_state_after_decay():
    """A nonzero initial state must be decayed before it is read back."""
    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        1, 29, "normal", dtype=torch.float64)
    state = make_initial_state("nonzero", 1, 29, dtype=torch.float64)
    actual, _ = reference_recurrence(q, k, v, log_decay, beta, SCALE, initial_state=state)
    alpha = log_decay.exp()
    for h in range(HEADS):
        decayed = torch.diag(alpha[0, 0, h]) @ state[0, h]
        error = v[0, 0, h] - decayed.t() @ k[0, 0, h]
        updated = decayed + beta[0, 0, h] * torch.outer(k[0, 0, h], error)
        expected = SCALE * (updated.t() @ q[0, 0, h])
        assert (actual[0, 0, h] - expected).abs().max().item() < ORACLE_MAX_ABS


# --------------------------------------------------------------------------
# Tier A1: compatibility with FLA's pinned FP32 naive implementation.
# --------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("length,seed,regime,state_name", ALL_CONDITIONS)
def test_fla_naive_fp32_compatibility_gate(length, seed, regime, state_name,
                                                record_property):
    """Compatibility with FLA's naive path, which computes in float32.

    float64 is fed in so our side contributes no rounding of its own.
    Production never runs in float64 to satisfy this test.
    """
    from fla.ops.kda.naive import naive_recurrent_kda

    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        length, seed, regime, dtype=torch.float64)
    state = make_initial_state(state_name, q.shape[0], seed, dtype=torch.float64)
    expected, expected_state = naive_recurrent_kda(
        q=q, k=k, v=v, g=log_decay, beta=beta, scale=SCALE,
        initial_state=state, output_final_state=True)
    actual, actual_state = reference_recurrence(
        q, k, v, log_decay, beta, SCALE, initial_state=state)

    output_error = (actual - expected).abs().max().item()
    state_error = (actual_state - expected_state).abs().max().item()
    record_property("tier_a_output_max_abs", output_error)
    record_property("tier_a_state_max_abs", state_error)
    assert output_error < FLA_NAIVE_OUTPUT_MAX_ABS
    assert state_error < FLA_NAIVE_STATE_MAX_ABS


@CUDA
def test_fla_naive_fp32_compatibility_gradients():
    from fla.ops.kda.naive import naive_recurrent_kda

    for length, seed, regime in [(64, 3, "normal"), (256, 11, "near_lower_bound")]:
        q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
            length, seed, regime, dtype=torch.float64)
        tensors = [t.clone().requires_grad_(True) for t in (q, k, v, log_decay, beta)]
        others = [t.clone().requires_grad_(True) for t in (q, k, v, log_decay, beta)]

        expected, _ = naive_recurrent_kda(
            q=others[0], k=others[1], v=others[2], g=others[3], beta=others[4],
            scale=SCALE, output_final_state=True)
        actual, _ = reference_recurrence(*tensors, SCALE)
        expected.square().mean().backward()
        actual.square().mean().backward()

        for name, mine, theirs in zip(("q", "k", "v", "log_decay", "beta"), tensors, others):
            assert mine.grad is not None and theirs.grad is not None, name
            assert torch.isfinite(mine.grad).all(), name
            assert mine.grad.abs().max().item() > 0.0, name
            assert (mine.grad - theirs.grad).abs().max().item() < FLA_NAIVE_GRAD_MAX_ABS, name


@CUDA
def test_reference_reads_the_decayed_state_not_the_previous_one():
    """The single most consequential ordering in this file."""
    q, k, v, _, log_decay, beta, _, _ = make_activated_inputs(
        8, 3, "normal", dtype=torch.float64)
    correct, _ = reference_recurrence(q, k, v, log_decay, beta, SCALE)

    def decay_last(q, k, v, log_decay, beta, scale):
        batch, length, heads, dim = v.shape
        alpha = log_decay.exp()
        state = torch.zeros(batch, heads, dim, dim, device=v.device, dtype=v.dtype)
        outputs = []
        for t in range(length):
            predicted = (state * k[:, t].unsqueeze(-1)).sum(-2)  # undecayed: wrong
            error = v[:, t] - predicted
            state = state * alpha[:, t].unsqueeze(-1) + beta[:, t].view(
                batch, heads, 1, 1) * (k[:, t].unsqueeze(-1) * error.unsqueeze(-2))
            outputs.append(scale * (state * q[:, t].unsqueeze(-1)).sum(-2))
        return torch.stack(outputs, dim=1)

    wrong = decay_last(q, k, v, log_decay, beta, SCALE)
    assert (correct - wrong).abs().max().item() > 1.0e-4, (
        "the decay-first ordering makes no difference here, so this test proves nothing"
    )


# --------------------------------------------------------------------------
# Tier B: production-kernel precision envelope.
# --------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("length,seed,regime,state_name", ALL_CONDITIONS)
def test_fla_chunk_kernel_output_and_state_error_envelope(
    length, seed, regime, state_name, record_property
):
    """Output and final state are judged separately: a good result on one must
    not offset a bad one on the other."""
    from fla.ops.kda.naive import naive_recurrent_kda

    q, k, v, raw_gate, log_decay, beta, A_log, dt_bias = make_activated_inputs(
        length, seed, regime)
    state = make_initial_state(state_name, q.shape[0], seed)
    expected, expected_state = naive_recurrent_kda(
        q=q, k=k, v=v, g=log_decay, beta=beta, scale=SCALE,
        initial_state=state, output_final_state=True)
    actual, actual_state = call_chunk_kda_checked(
        q=q, k=k, v=v, raw_gate=raw_gate, raw_beta=torch.logit(beta.clamp(1e-6, 1 - 1e-6)),
        A_log=A_log, dt_bias=dt_bias,
        initial_state=state.transpose(-1, -2).contiguous() if state is not None else None,
        output_final_state=True)

    output_metrics = assert_kernel_envelope(actual, expected)
    # `state_v_first=True` stores [B, H, V, K]; the reference is [B, H, K, V].
    state_metrics = assert_kernel_envelope(actual_state.transpose(-1, -2), expected_state)
    for prefix, metrics in (("output", output_metrics), ("state", state_metrics)):
        for key, value in metrics.items():
            record_property(f"{prefix}.{key}", value)


@CUDA
@pytest.mark.parametrize("length", SEQUENCE_LENGTHS)
def test_reference_layer_vs_fast_layer_error_envelope(length, record_property):
    torch.manual_seed(11)
    layer = KDALayer(768, KDAConfig()).cuda().float()
    ids = torch.randint(5, 49152, (2, length), device="cuda")
    ids[0, length // 3] = 4
    info = build_document_info(ids)
    x = torch.randn(2, length, 768, device="cuda")

    layer.use_fast = False
    reference, _ = layer(x, document_info=info)
    layer.use_fast = True
    fast, _ = layer(x, document_info=info)

    metrics = assert_kernel_envelope(fast, reference)
    for key, value in metrics.items():
        record_property(f"layer.{key}", value)


@CUDA
def test_reference_layer_vs_fast_layer_grad_envelope(record_property):
    torch.manual_seed(11)
    layer = KDALayer(768, KDAConfig()).cuda().float()
    ids = torch.randint(5, 49152, (2, 128), device="cuda")
    ids[0, 40] = 4
    info = build_document_info(ids)
    x = torch.randn(2, 128, 768, device="cuda", requires_grad=True)

    layer.use_fast = False
    reference, _ = layer(x, document_info=info)
    layer.use_fast = True
    fast, _ = layer(x, document_info=info)

    names = ("projections.A_log", "projections.dt_bias",
             "projections.q_proj.weight", "projections.o_proj.weight")
    parameters = [dict(layer.named_parameters())[name] for name in names]
    reference_grads = torch.autograd.grad(reference.square().mean(),
                                          [x, *parameters], retain_graph=True)
    fast_grads = torch.autograd.grad(fast.square().mean(),
                                     [x, *parameters], retain_graph=True)

    for name, mine, theirs in zip(("input", *names), fast_grads, reference_grads):
        metrics = error_metrics(mine, theirs)
        record_property(f"grad_{name}_rel_l2", metrics["rel_l2"])
        assert torch.isfinite(mine).all(), name
        assert metrics["max_abs"] < GRAD_MAX_ABS, (name, metrics)
        assert metrics["rel_l2"] < GRAD_REL_L2, (name, metrics)


# --------------------------------------------------------------------------
# Sensitivity: the `**kwargs` hazard, at both initial-state settings.
# --------------------------------------------------------------------------


@CUDA
@pytest.mark.parametrize("state_name", ["none", "nonzero"])
def test_A_log_changes_chunk_output(state_name):
    q, k, v, raw_gate, _, beta, A_log, dt_bias = make_activated_inputs(64, 3, "normal")
    state = make_initial_state(state_name, 1, 3)
    raw_beta = torch.logit(beta.clamp(1e-6, 1 - 1e-6))
    common = dict(q=q, k=k, v=v, raw_gate=raw_gate, raw_beta=raw_beta,
                  dt_bias=dt_bias, initial_state=state, output_final_state=True)
    a, _ = call_chunk_kda_checked(A_log=A_log, **common)
    b, _ = call_chunk_kda_checked(A_log=torch.full_like(A_log, 2.0), **common)
    assert (a - b).abs().max().item() > 1.0e-4, "A_log was ignored"


@CUDA
@pytest.mark.parametrize("state_name", ["none", "nonzero"])
def test_dt_bias_changes_chunk_output(state_name):
    q, k, v, raw_gate, _, beta, A_log, dt_bias = make_activated_inputs(64, 3, "normal")
    state = make_initial_state(state_name, 1, 3)
    raw_beta = torch.logit(beta.clamp(1e-6, 1 - 1e-6))
    common = dict(q=q, k=k, v=v, raw_gate=raw_gate, raw_beta=raw_beta,
                  A_log=A_log, initial_state=state, output_final_state=True)
    a, _ = call_chunk_kda_checked(dt_bias=dt_bias, **common)
    b, _ = call_chunk_kda_checked(dt_bias=torch.full_like(dt_bias, 1.5), **common)
    assert (a - b).abs().max().item() > 1.0e-4, "dt_bias was ignored"


# --------------------------------------------------------------------------
# Structure, initialisation, convolution and document isolation.
# --------------------------------------------------------------------------


def test_exact_projection_parameter_count():
    assert sum(t.numel() for t in KDAProjections(768, KDAConfig()).parameters()) == 2_575_948


def test_specified_initialisation():
    projections = KDAProjections(768, KDAConfig())
    assert projections.A_log.shape == (HEADS,) and projections.A_log.dtype == torch.float32
    assert projections.dt_bias.shape == (HEADS * HEAD_DIM,)
    assert torch.equal(projections.A_log, torch.zeros(HEADS))
    assert torch.equal(projections.gate_up.bias, torch.zeros(768))
    dt = torch.nn.functional.softplus(projections.dt_bias)
    assert 1.0e-3 * 0.9 < dt.min().item() and dt.max().item() < 1.0e-1 * 1.1
    for weight in (projections.q_conv_weight, projections.k_conv_weight,
                   projections.v_conv_weight):
        assert torch.equal(weight[:, 0, -1], torch.ones(768))
        assert torch.equal(weight[:, 0, :-1], torch.zeros(768, 3))


def test_convolution_is_causal_and_resets_at_documents():
    projections = KDAProjections(768, KDAConfig())
    x = torch.randn(1, 6, 768)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int32)

    identity, _ = causal_depthwise_convolution(x, projections.q_conv_weight, segment_ids)
    assert torch.allclose(identity, x, atol=1e-6)

    three_back = torch.zeros(768, 1, 4)
    three_back[:, 0, 0] = 1.0
    output, _ = causal_depthwise_convolution(x, three_back, segment_ids)
    assert output[0, 3].abs().max().item() == 0.0  # first token of document 1
    assert output[0, 2].abs().max().item() == 0.0  # would need t = -1
    assert output[0, 5].abs().max().item() == 0.0  # t = 2 is in document 0


@CUDA
def test_document_state_and_convolution_reset():
    """Changing document 0 must not move a single logit in document 1."""
    torch.manual_seed(11)
    layer = KDALayer(768, KDAConfig()).cuda().float()
    embedding = torch.nn.Embedding(49152, 768).cuda()
    ids = torch.randint(5, 49152, (1, 64), device="cuda")
    ids[0, 20] = 4

    def run(token_ids):
        info = build_document_info(token_ids)
        output, _ = layer(embedding(token_ids), document_info=info)
        return output

    original = run(ids)
    changed = ids.clone()
    changed[0, :20] = torch.randint(5, 49152, (20,), device="cuda")
    after = run(changed)
    assert (original[0, 21:] - after[0, 21:]).abs().max().item() < 1.0e-5
