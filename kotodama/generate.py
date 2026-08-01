"""Incremental generation.

Two things here are specific to a looped model.

**The loop depth is fixed for the session.**  The core's weights are shared
across iterations but its history is not: iteration 3 has seen a different
sequence of states than iteration 4.  Changing the depth mid-generation would
leave every cached iteration describing a trajectory that no longer exists, so
it raises and the caller has to re-prefill.  Depth is a knob you turn between
generations, not during one.

**Document info is sliced, never rebuilt.**  `build_document_info` on a
one-token slice marks that token as the start of a document, which resets the
KDA recurrent state on every step.  The output stays finite and reads
plausibly; the model has simply forgotten everything before the current token.
This is why `slice_document_info` exists and why nothing here calls
`build_document_info` on a slice.
"""

from __future__ import annotations

import torch

from kotodama.cache import ModelCache, build_model_cache
from kotodama.config import EOD_TOKEN_ID
from kotodama.segments import build_document_info, slice_document_info


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None,
            top_p: float | None, generator: torch.Generator | None) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        ordered, indices = logits.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        # Keep the first token that crosses the threshold, drop the rest.
        remove = cumulative - ordered.softmax(dim=-1) > top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, ordered)

    probabilities = logits.softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    loop_depth: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = None,
    stop_at_eod: bool = True,
    seed: int | None = None,
    cache: ModelCache | None = None,
) -> torch.Tensor:
    """Prefill the prompt, then decode one token at a time from the cache.

    `loop_depth` is the compute budget per token.  Because the core is time
    invariant -- no iteration index reaches it -- it may be set higher than any
    depth seen in training, though nothing guarantees that helps.
    """
    was_training = model.training
    model.eval()
    device = input_ids.device
    depth = loop_depth if loop_depth is not None else model.config.loop.inference_depth

    if cache is None:
        cache = build_model_cache(model, loop_depth=depth)
    else:
        cache.require_depth(depth)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    tokens = input_ids
    depths = torch.full((input_ids.shape[0],), depth, dtype=torch.int64, device=device)

    info = build_document_info(tokens)
    output = model(tokens, document_info=slice_document_info(info, 0, tokens.shape[1]),
                   loop_depths=depths, cache=cache)
    next_token = _sample(output.logits[:, -1], temperature, top_k, top_p, generator)

    generated = [next_token]
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    for _ in range(max_new_tokens - 1):
        tokens = torch.cat([tokens, next_token], dim=1)
        if stop_at_eod:
            finished |= next_token.squeeze(-1) == EOD_TOKEN_ID
            if bool(finished.all()):
                break
        # Rebuilt over the whole sequence so the boundary mask stays correct,
        # then sliced -- never built from the one-token window.
        info = build_document_info(tokens)
        step = slice_document_info(info, tokens.shape[1] - 1, tokens.shape[1])
        output = model(tokens[:, -1:], document_info=step, loop_depths=depths, cache=cache)
        next_token = _sample(output.logits[:, -1], temperature, top_k, top_p, generator)
        generated.append(next_token)

    if was_training:
        model.train()
    return torch.cat([input_ids, *generated], dim=1)


@torch.no_grad()
def generate_at_depths(model, input_ids: torch.Tensor, depths: list[int], **kwargs):
    """The same prompt decoded at several compute budgets.

    Each depth gets its own cache and its own prefill: reusing one cache across
    depths is the thing `require_depth` refuses.
    """
    return {depth: generate(model, input_ids, loop_depth=depth, **kwargs)
            for depth in depths}


__all__ = ["generate", "generate_at_depths"]
