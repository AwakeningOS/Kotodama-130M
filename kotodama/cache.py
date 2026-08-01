"""Generation cache, separated by logical loop depth.

The core's weights are shared across iterations.  Its *history* is not.
Iteration 3 has seen a different sequence of states than iteration 4, so its
recurrent state and its keys and values are different tensors that happen to
have been produced by the same parameters.  Sharing one cache between them
would silently mix them, and full-forward would stop matching cached decode.

At T = 8 that means:

    KDA   prelude 1 + core 6*8 + coda 2 = 51
    MLA   prelude 1 + core 2*8 + coda 0 = 17

The loop depth is fixed for the lifetime of a generation session.  Changing it
mid-session would invalidate every cached iteration at once, so it raises
instead: the caller must drop the cache and re-prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kotodama.blocks import MixerCache


@dataclass
class LoopIterationCache:
    """One logical pass through the shared core."""

    core_layers: list[MixerCache] = field(default_factory=list)

    def reset(self) -> None:
        for cache in self.core_layers:
            cache.reset()


@dataclass
class ModelCache:
    prelude_layers: list[MixerCache] = field(default_factory=list)
    loop_iterations: list[LoopIterationCache] = field(default_factory=list)
    coda_layers: list[MixerCache] = field(default_factory=list)
    fixed_loop_depth: int = 0
    position: int = 0
    current_segment_id: int = 0

    def require_depth(self, requested_depth: int) -> None:
        if requested_depth != self.fixed_loop_depth:
            raise RuntimeError(
                f"this cache was built for loop depth {self.fixed_loop_depth}, "
                f"not {requested_depth}. Changing loop depth requires cache reset "
                "and full prompt re-prefill"
            )

    def reset(self) -> None:
        """Called at a document boundary: every level, not just the core."""
        for cache in self.prelude_layers:
            cache.reset()
        for iteration in self.loop_iterations:
            iteration.reset()
        for cache in self.coda_layers:
            cache.reset()
        self.position = 0

    def reset_rows(self, rows) -> None:
        """Reset only sessions whose next token begins a new document."""
        for cache in self.prelude_layers:
            cache.reset_rows(rows)
        for iteration in self.loop_iterations:
            for cache in iteration.core_layers:
                cache.reset_rows(rows)
        for cache in self.coda_layers:
            cache.reset_rows(rows)

    def count(self, kind: str) -> int:
        caches = [*self.prelude_layers, *self.coda_layers]
        for iteration in self.loop_iterations:
            caches.extend(iteration.core_layers)
        attribute = "kda" if kind == "KDA" else "mla"
        return sum(1 for cache in caches if getattr(cache, attribute) is not None)


def build_model_cache(model, loop_depth: int) -> ModelCache:
    """One cache per block per logical iteration -- never one shared set."""
    return ModelCache(
        prelude_layers=[block.new_cache() for block in model.prelude],
        loop_iterations=[
            LoopIterationCache([block.new_cache() for block in model.recurrent_core])
            for _ in range(loop_depth)
        ],
        coda_layers=[block.new_cache() for block in model.coda],
        fixed_loop_depth=loop_depth,
    )


__all__ = ["LoopIterationCache", "ModelCache", "build_model_cache"]
