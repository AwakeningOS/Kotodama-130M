"""Depth sampling is checked by its distribution, not just its range.

The sampler once returned a realised mean of 4.43 where 6.89 was correct: a
sign-extension bug made half the hashed uniforms negative, so those sequences
exited Knuth's loop immediately and were clamped up to the minimum.  Every
value still sat inside `[2, 8]`, so a range test passed.  Only the histogram
showed it.  Hence the exact snapshots below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.depth_sampler import (  # noqa: E402
    poisson_lambda,
    realized_mean_depth,
    sample_depths,
    uniform_stream,
)

SEED = 11
IDENTIFIERS = torch.arange(4096)

# Exact histograms for (seed=11, ids=arange(4096)).  Any change to the hash,
# the sampling loop or the ramp moves these.
EXPECTED_COUNTS = {
    0: {2: 2640, 3: 655, 4: 352, 5: 149, 6: 42, 7: 11, 8: 247},
    100_000_000: {2: 2001, 3: 852, 4: 529, 5: 302, 6: 105, 7: 42, 8: 265},
    1_000_000_000: {2: 53, 3: 116, 4: 231, 5: 311, 6: 491, 7: 548, 8: 2346},
}


def histogram(depths: torch.Tensor) -> dict[int, int]:
    return {depth: int((depths == depth).sum()) for depth in range(2, 9)}


def test_exact_histogram_snapshot():
    for consumed_tokens, expected in EXPECTED_COUNTS.items():
        depths = sample_depths(consumed_tokens, IDENTIFIERS, run_seed=SEED)
        assert histogram(depths) == expected, f"distribution changed at {consumed_tokens} tokens"


def test_uniform_stream_is_a_real_uniform():
    """Checked directly, because its failure hid inside the clamp."""
    values = uniform_stream(torch.arange(200_000))
    assert values.min().item() >= 0.0
    assert values.max().item() < 1.0
    assert abs(values.mean().item() - 0.5) < 0.005


def test_realized_mean_tracks_the_poisson_rate_and_max_depth_tail():
    """The fixed max-depth tail raises the realised mean as specified."""
    depths = sample_depths(1_000_000_000, IDENTIFIERS, run_seed=SEED)
    rate = poisson_lambda(1_000_000_000, 1_000_000_000, 2, 8)
    assert rate == 8.0
    assert 6.8 < realized_mean_depth(depths) < 7.1


def test_max_depth_is_present_at_the_100m_gate():
    depths = sample_depths(100_000_000, IDENTIFIERS, run_seed=SEED)
    fraction = float((depths == 8).float().mean())
    assert 0.05 < fraction < 0.08
    assert 3.0 < realized_mean_depth(depths) < 3.5


def test_depths_are_always_inside_the_training_range():
    for consumed_tokens in (0, 100_000_000, 500_000_000, 1_000_000_000, 10_000_000_000):
        depths = sample_depths(consumed_tokens, IDENTIFIERS, run_seed=SEED)
        assert int(depths.min()) >= 2
        assert int(depths.max()) <= 8


def test_resume_reproduces_the_same_depths():
    arguments = dict(consumed_tokens=1_000_000_000, batch_global_sequence_ids=IDENTIFIERS,
                     run_seed=SEED, optimizer_step=7, micro_step=3)
    assert torch.equal(sample_depths(**arguments), sample_depths(**arguments))


def test_depth_depends_on_every_key_component():
    base = dict(consumed_tokens=1_000_000_000, batch_global_sequence_ids=IDENTIFIERS,
                run_seed=SEED, optimizer_step=7, micro_step=3)
    reference = sample_depths(**base)
    for field, value in (("run_seed", 12), ("optimizer_step", 8), ("micro_step", 4)):
        assert not torch.equal(reference, sample_depths(**{**base, field: value})), field


def test_sampler_uses_no_global_rng():
    """A global-RNG dependency would break resume without changing any output
    that a loss curve could reveal."""
    torch.manual_seed(1)
    first = sample_depths(1_000_000_000, IDENTIFIERS, run_seed=SEED)
    torch.manual_seed(999)
    _ = torch.randn(1000)
    second = sample_depths(1_000_000_000, IDENTIFIERS, run_seed=SEED)
    assert torch.equal(first, second)
