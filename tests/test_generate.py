"""Generation: cache correctness, determinism, and the depth contract.

Runs on the CPU with `ReferenceKDA`, and on a model whose blocks have been
perturbed away from the identity -- a freshly initialised model returns zero
from every block, so any comparison between two paths would pass vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.cache import build_model_cache  # noqa: E402
from kotodama.config import EOD_TOKEN_ID  # noqa: E402
from kotodama.generate import generate, generate_at_depths  # noqa: E402
from kotodama.model import build_model  # noqa: E402
from kotodama.segments import build_document_info  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(11)
    built = build_model(fast_kda=False).float().eval()
    with torch.no_grad():
        for block in [*built.prelude, *built.recurrent_core, *built.coda]:
            block.ffn.down_proj.weight.normal_(std=0.02)
            projection = (block.mixer.projections.o_proj if block.kind == "KDA"
                          else block.mixer.o_proj)
            projection.weight.normal_(std=0.02)
    return built


@pytest.fixture
def prompt():
    torch.manual_seed(3)
    return torch.randint(5, 49152, (1, 6))


def test_output_length_and_prompt_preserved(model, prompt):
    out = generate(model, prompt, max_new_tokens=5, loop_depth=2, temperature=0.0,
                   stop_at_eod=False)
    assert out.shape == (1, 11)
    assert torch.equal(out[:, :6], prompt)


def test_greedy_generation_is_deterministic(model, prompt):
    first = generate(model, prompt, max_new_tokens=6, loop_depth=2, temperature=0.0,
                     stop_at_eod=False)
    second = generate(model, prompt, max_new_tokens=6, loop_depth=2, temperature=0.0,
                      stop_at_eod=False)
    assert torch.equal(first, second)


def test_sampling_is_reproducible_from_a_seed(model, prompt):
    first = generate(model, prompt, max_new_tokens=6, loop_depth=2, seed=7,
                     stop_at_eod=False)
    second = generate(model, prompt, max_new_tokens=6, loop_depth=2, seed=7,
                      stop_at_eod=False)
    assert torch.equal(first, second)


def test_cached_generation_matches_full_recomputation(model, prompt):
    """The point of the cache: every generated token must be the one a full
    forward over the whole prefix would have produced."""
    generated = generate(model, prompt, max_new_tokens=5, loop_depth=2,
                         temperature=0.0, stop_at_eod=False)
    depths = torch.full((1,), 2, dtype=torch.int64)
    for position in range(prompt.shape[1], generated.shape[1]):
        prefix = generated[:, :position]
        with torch.no_grad():
            logits = model(prefix, document_info=build_document_info(prefix),
                           loop_depths=depths).logits
        assert int(logits[0, -1].argmax()) == int(generated[0, position]), position


def test_stops_at_end_of_document(model, prompt):
    """A forced EOD ends generation rather than running past the boundary."""
    with torch.no_grad():
        model.embed_tokens.weight[EOD_TOKEN_ID] += 50.0
    try:
        out = generate(model, prompt, max_new_tokens=20, loop_depth=1,
                       temperature=0.0, stop_at_eod=True)
        assert out.shape[1] < prompt.shape[1] + 20
        assert EOD_TOKEN_ID in out[0, prompt.shape[1]:].tolist()
    finally:
        with torch.no_grad():
            model.embed_tokens.weight[EOD_TOKEN_ID] -= 50.0


def test_depth_may_exceed_the_training_range(model, prompt):
    """The core is time invariant, so nothing structurally prevents it."""
    out = generate(model, prompt, max_new_tokens=3, loop_depth=12,
                   temperature=0.0, stop_at_eod=False)
    assert out.shape == (1, 9)


def test_reusing_a_cache_at_a_different_depth_is_refused(model, prompt):
    cache = build_model_cache(model, loop_depth=2)
    generate(model, prompt, max_new_tokens=2, loop_depth=2, temperature=0.0,
             stop_at_eod=False, cache=cache)
    with pytest.raises(RuntimeError, match="re-prefill"):
        generate(model, prompt, max_new_tokens=2, loop_depth=4, temperature=0.0,
                 stop_at_eod=False, cache=cache)


def test_generate_at_depths_gives_each_depth_a_fresh_cache(model, prompt):
    results = generate_at_depths(model, prompt, depths=[1, 2, 4],
                                 max_new_tokens=3, temperature=0.0, stop_at_eod=False)
    assert set(results) == {1, 2, 4}
    for depth, output in results.items():
        assert output.shape == (1, 9), depth
        assert torch.equal(output[:, :6], prompt)


def test_generation_leaves_training_mode_unchanged(model, prompt):
    model.train()
    try:
        generate(model, prompt, max_new_tokens=2, loop_depth=1, temperature=0.0,
                 stop_at_eod=False)
        assert model.training
    finally:
        model.eval()
