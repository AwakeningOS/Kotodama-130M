"""100M-gate recommended inference entry point.

Kept outside ``kotodama/`` so selecting an inference compute budget does not
change the training source fingerprint or invalidate resumable checkpoints.
"""

from __future__ import annotations

from kotodama.generate import generate

RECOMMENDED_INFERENCE_DEPTH = 2
RECOMMENDATION_GATE_TOKENS = 100_007_936


def generate_recommended(model, input_ids, **kwargs):
    """Generate at the 100M quality/speed Pareto depth unless overridden."""
    kwargs.setdefault("loop_depth", RECOMMENDED_INFERENCE_DEPTH)
    return generate(model, input_ids, **kwargs)


__all__ = [
    "RECOMMENDATION_GATE_TOKENS",
    "RECOMMENDED_INFERENCE_DEPTH",
    "generate_recommended",
]
