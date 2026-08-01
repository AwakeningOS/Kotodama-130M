"""Deterministic per-sequence loop-depth sampling.

Depth is drawn per sequence, not per batch, from a bounded Poisson whose rate
ramps with consumed tokens.  v2 also reserves a deterministic high-depth tail:

    lambda(n) = 2 + 6 * min(n / depth_ramp_tokens, 1)
    T_i       = 8                         with probability p_max
                clamp(Poisson(lambda(n)), 2, 8) otherwise

`lambda` is the *rate of the Poisson before clamping*, not the mean depth that
actually occurs.  At `lambda = 8` the realised mean is about 6.89, because the
clamp caps the upper tail at 8 while lifting 0 and 1 up to 2.  Recording that
number as "mean_depth = 8" would invite a later reader to "fix" a shortfall
that does not exist, so the two quantities are named separately everywhere.

The tail exposes the shared core to the inference depth from the beginning,
while the bulk of the batch still follows a cheap curriculum.

**Determinism is a resume contract, not a nicety.**  The depth for a given
sequence is a pure function of

    (run_seed, optimizer_step, micro_step, global_sequence_index)

and nothing else.  No global RNG, no module state, no call ordering.  A run
resumed from a checkpoint must assign the same depth to the same sequence, or
the "same tokens in the same order" guarantee is broken in a way that no loss
curve will reveal.
"""

from __future__ import annotations

import torch

# A 63-bit splitmix64 variant.  Torch has no unsigned 64-bit integer, and the
# reference constants exceed 2**63, so everything is masked into the
# non-negative range after each step.  This matters for correctness, not
# tidiness: on a signed int64, `>>` sign-extends, which made half the hashed
# keys negative, produced negative "uniforms", and collapsed the Poisson draw
# to zero for those sequences -- the sampler returned mean depth 4.4 where 7.0
# was expected, entirely inside the clamp so nothing looked wrong.
_MASK63 = 0x7FFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15 & _MASK63
_MIX_A = 0xBF58476D1CE4E5B9 & _MASK63
_MIX_B = 0x94D049BB133111EB & _MASK63
# 2**52, the span of `key >> 11` once the key is confined to 63 bits.
_UNIFORM_SCALE = 1.0 / float(1 << 52)


def _splitmix64(value: torch.Tensor) -> torch.Tensor:
    """Vectorised splitmix-style hash confined to [0, 2**63)."""
    z = (value + _GOLDEN) & _MASK63
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK63
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK63
    return (z ^ (z >> 31)) & _MASK63


def _uniform(keys: torch.Tensor) -> torch.Tensor:
    """Map hashed keys into [0, 1) as doubles."""
    return (keys >> 11).double() * _UNIFORM_SCALE


def poisson_lambda(consumed_tokens: int, ramp_tokens: int, minimum: int, maximum: int) -> float:
    """The Poisson rate before clamping: `min + (max - min) * min(n / ramp, 1)`.

    This is not the mean depth that will be observed.  Use
    `realized_mean_depth` for that.
    """
    progress = min(consumed_tokens / max(ramp_tokens, 1), 1.0)
    return minimum + (maximum - minimum) * progress


def realized_mean_depth(depths: torch.Tensor) -> float:
    """The mean depth actually produced, after the clamp."""
    return float(depths.float().mean())


def uniform_stream(keys: torch.Tensor) -> torch.Tensor:
    """The hashed uniform draws, exposed so tests can check them directly.

    A sign-extension bug here produced negative "uniforms" and silently halved
    the realised mean depth while every value still sat inside the clamp range,
    so this stream is checked on its own rather than only through its effects.
    """
    return _uniform(_splitmix64(keys.to(torch.int64)))


def sample_depths(
    consumed_tokens: int,
    batch_global_sequence_ids: torch.Tensor,
    run_seed: int,
    optimizer_step: int = 0,
    micro_step: int = 0,
    minimum_depth: int = 2,
    maximum_depth: int = 8,
    ramp_tokens: int = 1_000_000_000,
    maximum_depth_probability: float = 1.0 / 16.0,
) -> torch.Tensor:
    """Bounded-Poisson depths, one per sequence, reproducible from the inputs.

    Poisson is sampled by Knuth's method against a hash-derived uniform stream,
    so the whole draw stays a pure function of the key.  The rate is at most 8
    and the result is clamped, so the loop count is bounded and short.
    """
    device = batch_global_sequence_ids.device
    ids = batch_global_sequence_ids.to(torch.int64)

    key = torch.full_like(ids, run_seed & _MASK63)
    key = _splitmix64(key ^ _splitmix64(torch.full_like(ids, optimizer_step)))
    key = _splitmix64(key ^ _splitmix64(torch.full_like(ids, micro_step)))
    key = _splitmix64(key ^ _splitmix64(ids))

    rate = poisson_lambda(consumed_tokens, ramp_tokens, minimum_depth, maximum_depth)
    threshold = torch.tensor(-float(rate), dtype=torch.float64, device=device).exp()

    counts = torch.zeros_like(ids)
    product = torch.ones(ids.shape, dtype=torch.float64, device=device)
    active = torch.ones_like(ids, dtype=torch.bool)
    # Knuth's Poisson: multiply uniforms until the product falls below e^-mu.
    # Bounded explicitly -- the clamp discards anything above `maximum_depth`
    # anyway, so there is no reason to let the loop run long.
    for iteration in range(4 * maximum_depth + 16):
        if not bool(active.any()):
            break
        stream = _splitmix64(key ^ _splitmix64(torch.full_like(ids, iteration)))
        product = torch.where(active, product * _uniform(stream), product)
        active = product > threshold
        counts = counts + active.to(counts.dtype)

    depths = counts.clamp_(minimum_depth, maximum_depth)
    if maximum_depth_probability > 0.0:
        if not 0.0 <= maximum_depth_probability < 1.0:
            raise ValueError("maximum_depth_probability must be in [0, 1)")
        anchor_key = _splitmix64(key ^ _splitmix64(torch.full_like(ids, 0x4D41584445505448)))
        anchored = _uniform(anchor_key) < maximum_depth_probability
        depths = torch.where(anchored, torch.full_like(depths, maximum_depth), depths)
    return depths


__all__ = ["poisson_lambda", "realized_mean_depth", "sample_depths", "uniform_stream"]
