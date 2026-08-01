"""Architecture contracts for kotodama_stable_loop_130m_v2."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.blocks import DecoderBlock  # noqa: E402
from kotodama.cache import build_model_cache  # noqa: E402
from kotodama.config import EXACT_PARAMETER_COUNT, KotodamaConfig  # noqa: E402
from kotodama.loop_injection import (  # noqa: E402
    INITIAL_DECAY_TARGET,
    StableDiagonalInjection,
)
from kotodama.layers import SwiGLUMLP  # noqa: E402
from kotodama.model import build_model  # noqa: E402
from kotodama.segments import build_document_info  # noqa: E402

FORBIDDEN = ("mudd", "delta_bank", "attn_res", "mtp", "moe", "halting",
             "loop_embedding", "round_embedding", "rope", "rotary")


@pytest.fixture(scope="module")
def model():
    return build_model(fast_kda=False)


# ---------------------------------------------------------------- architecture


def test_exact_parameter_count(model):
    assert model.parameter_count() == EXACT_PARAMETER_COUNT


def test_exact_unique_block_count(model):
    assert len(model.prelude) == 2
    assert len(model.recurrent_core) == 8
    assert len(model.coda) == 2
    kinds = [block.kind for block in
             [*model.prelude, *model.recurrent_core, *model.coda]]
    assert kinds.count("KDA") == 9 and kinds.count("MLA") == 3


def test_no_parameter_duplication_by_loop_depth(model):
    keys = set(model.state_dict().keys())
    count = model.parameter_count()
    ids = torch.randint(5, 49152, (1, 8))
    for depth in (1, 12):
        model(ids, loop_depths=torch.tensor([depth]))
        assert set(model.state_dict().keys()) == keys
        assert model.parameter_count() == count


def test_tied_embedding_pointer_identity(model):
    assert model.lm_head_weight is model.embed_tokens.weight


def test_forbidden_modules_absent(model):
    for key in model.state_dict():
        for token in FORBIDDEN:
            assert token not in key.lower(), key


# ------------------------------------------------------------- loop injection


def test_decay_strictly_between_zero_and_one():
    decay = StableDiagonalInjection(768).get_decay()
    assert decay.min().item() > 0.0 and decay.max().item() < 1.0


def test_initial_decay_matches_sqrt_one_fifth():
    decay = StableDiagonalInjection(768).get_decay()
    assert abs(decay.mean().item() - INITIAL_DECAY_TARGET) < 1e-6


def test_B_is_exact_identity():
    assert torch.equal(StableDiagonalInjection(768).B, torch.eye(768))


def test_closed_form_identity_core():
    injection = StableDiagonalInjection(64)
    e = torch.randn(2, 5, 64)
    h = torch.zeros_like(e)
    for _ in range(6):
        h = injection(h, e)
    expected = injection.closed_form_identity_core(e, 6)
    assert (h - expected).abs().max().item() < 1e-5


def test_spectral_radius_below_one():
    assert StableDiagonalInjection(768).get_spectral_radius().item() < 1.0


def test_no_external_loop_residual(model):
    """`h <- Core(J(h, e))`, not `h + Core(J(h, e))`."""
    injection = model.loop_injection
    h = torch.randn(1, 4, 768) * 0.1
    e = torch.randn(1, 4, 768) * 0.1

    original = model.recurrent_core
    try:
        model.recurrent_core = torch.nn.ModuleList()

        def doubling(current, e_, document_info, caches=None):
            return 2.0 * model.loop_injection(current, e_)

        actual = doubling(h, e, None)
        expected = 2.0 * injection(h, e)
        assert (actual - expected).abs().max().item() < 1e-6
        assert (actual - (h + expected)).abs().max().item() > 1e-3
    finally:
        model.recurrent_core = original


def test_prelude_gradient_is_not_detached(model):
    ids = torch.randint(5, 49152, (1, 8))
    result = model.loss_from_ids(ids, loop_depths=torch.tensor([3]))
    result.loss.backward()
    weight = model.prelude[0].mixer.projections.q_proj.weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    model.zero_grad(set_to_none=True)


def test_core_exit_starts_as_identity(model):
    assert torch.equal(model.core_exit.weight, torch.eye(768))
    assert model.core_exit.bias is None


# --------------------------------------------------------------------- blocks


def test_block_is_an_exact_identity_at_initialisation():
    config = KotodamaConfig()
    model = build_model(fast_kda=False)
    x = torch.randn(1, 6, 768, dtype=torch.float32)
    for block in [*model.prelude, *model.recurrent_core, *model.coda]:
        output, _ = block(x, document_info=build_document_info(
            torch.randint(5, 49152, (1, 6))))
        assert torch.equal(output, x), block.kind


def test_v2_uses_swiglu_everywhere(model):
    for block in [*model.prelude, *model.recurrent_core, *model.coda]:
        assert isinstance(block.ffn, SwiGLUMLP)


def test_like_init_state_is_training_only(model):
    e = torch.zeros(2, 4, 768)
    model.train()
    noisy = model.initial_loop_state(e, labels=torch.zeros(2, 4, dtype=torch.long))
    assert noisy.std().item() > 0.5
    assert noisy.abs().max().item() <= 3.0 * model.config.loop.state_init_std

    model.eval()
    assert torch.equal(model.initial_loop_state(e, labels=torch.zeros(2, 4, dtype=torch.long)), e)
    model.train()
    assert torch.equal(model.initial_loop_state(e, labels=None), e)


# -------------------------------------------------------------- variable depth


def test_depth_range_and_per_sequence_masking(model):
    ids = torch.randint(5, 49152, (3, 8))
    depths = torch.tensor([2, 5, 8])
    output = model(ids, loop_depths=depths)
    assert output.logits.shape == (3, 8, 49152)


def test_inactive_sequence_state_is_bitwise_unchanged(model):
    """A sequence that stopped at depth 2 must be identical to one run at 2."""
    ids = torch.randint(5, 49152, (2, 8))
    mixed = model(ids, loop_depths=torch.tensor([2, 6])).final_state
    alone = model(ids[:1], loop_depths=torch.tensor([2])).final_state
    assert torch.equal(mixed[0], alone[0])


def test_no_loop_index_reaches_the_core(model):
    """The core must be time invariant, or extrapolating past the trained
    depth range is meaningless."""
    import inspect
    source = inspect.getsource(model.recurrent_step)
    for token in ("loop_index", "iteration", "total_loops", "step_index"):
        assert token not in source, token


def test_all_active_loops_receive_gradient(model):
    ids = torch.randint(5, 49152, (1, 8))
    model.zero_grad(set_to_none=True)
    model.loss_from_ids(ids, loop_depths=torch.tensor([4])).loss.backward()
    for name in ("log_A", "dt_bias", "B"):
        grad = getattr(model.loop_injection, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().max().item() > 0.0, name
    model.zero_grad(set_to_none=True)


def test_no_detach_in_the_recurrent_path():
    """The loop state must never be *reassigned* from a detached value.

    A substring search cannot tell truncated BPTT from a diagnostic copy: the
    telemetry hook legitimately does `loop_states.append(h.detach())`, which
    reads a value without cutting the graph.  What must not happen is `h`
    itself being rebound to something detached, so the check is on assignments
    to `h`.
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "kotodama" / "model.py").read_text()
    assert "stop_gradient" not in source

    tree = ast.parse(source)
    detaching = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "h" not in targets:
            continue
        for inner in ast.walk(node.value):
            if isinstance(inner, ast.Attribute) and inner.attr in ("detach", "data"):
                detaching.append(node.lineno)
    assert detaching == [], f"loop state rebound from a detached value at {detaching}"


# ------------------------------------------------------- document isolation


@pytest.mark.parametrize("depth", [1, 4, 8])
def test_document_isolation(model, depth):
    """Changing document 0 must not move document 1 at any loop depth."""
    ids = torch.randint(5, 49152, (1, 24))
    ids[0, 11] = 4
    depths = torch.tensor([depth])
    original = model(ids, loop_depths=depths).logits

    changed = ids.clone()
    changed[0, :11] = torch.randint(5, 49152, (11,))
    after = model(changed, loop_depths=depths).logits
    assert (original[0, 12:] - after[0, 12:]).abs().max().item() < 1e-4


# ------------------------------------------------------------------- cache


def test_cache_counts_at_depth_eight(model):
    cache = build_model_cache(model, loop_depth=8)
    assert cache.count("KDA") == 51
    assert cache.count("MLA") == 17


def test_loop_caches_are_distinct_objects(model):
    cache = build_model_cache(model, loop_depth=4)
    identities = [id(entry) for iteration in cache.loop_iterations
                  for entry in iteration.core_layers]
    assert len(identities) == len(set(identities)) == 32


def test_changing_depth_with_a_live_cache_is_refused(model):
    cache = build_model_cache(model, loop_depth=4)
    with pytest.raises(RuntimeError, match="re-prefill"):
        cache.require_depth(8)
    cache.require_depth(4)
