"""Incremental decoding must equal recomputing the whole sequence.

**These tests must never run on a freshly initialised model.**  Every residual
output projection starts at exactly zero, so a KDA block returns zero whatever
its recurrence computed.  A cache test on that model compares two zeros and
passes while testing nothing at all -- which is exactly what happened here: the
first version of this check reported PASS on an untrained model, and the same
check on trained weights failed by 3.4e-2 relative.

So `perturbed_model` gives every zero-initialised projection a small non-zero
value first.  It is the same warning the Deltaxis notes give about testing with
zero-initialised routers, arrived at the hard way.

Everything here runs on the CPU: `ReferenceKDA` and the MLA are plain torch, so
wiring can be checked without a GPU.  Wiring is what these tests are for --
kernel precision is Tier B's job in test_kda.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.cache import build_model_cache  # noqa: E402
from kotodama.config import KotodamaConfig  # noqa: E402
from kotodama.model import build_model  # noqa: E402
from kotodama.segments import build_document_info, slice_document_info  # noqa: E402

LENGTH = 10
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def perturbed_model():
    """A model whose blocks are not the identity.

    Without this every block output is exactly zero and any comparison between
    two code paths succeeds trivially.
    """
    torch.manual_seed(11)
    model = build_model(fast_kda=False).float().eval()
    with torch.no_grad():
        for block in [*model.prelude, *model.recurrent_core, *model.coda]:
            block.ffn.down_proj.weight.normal_(std=0.02)
            if block.kind == "KDA":
                block.mixer.projections.o_proj.weight.normal_(std=0.02)
            else:
                block.mixer.o_proj.weight.normal_(std=0.02)
        model.loop_injection.B.add_(torch.randn_like(model.loop_injection.B) * 0.01)
        model.core_exit.weight.add_(torch.randn_like(model.core_exit.weight) * 0.01)
    return model


@pytest.fixture(scope="module")
def sequence():
    torch.manual_seed(3)
    ids = torch.randint(5, 49152, (1, LENGTH))
    return ids, build_document_info(ids)


def test_the_fixture_actually_perturbs_the_model(perturbed_model):
    """If this fails, every other test in this file is vacuous."""
    for block in perturbed_model.recurrent_core:
        projection = (block.mixer.projections.o_proj if block.kind == "KDA"
                      else block.mixer.o_proj)
        assert projection.weight.abs().max().item() > 0.0
        assert block.ffn.down_proj.weight.abs().max().item() > 0.0


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_single_token_decode_matches_full_forward(perturbed_model, sequence, depth):
    ids, info = sequence
    depths = torch.full((1,), depth, dtype=torch.int64)
    with torch.no_grad():
        full = perturbed_model(ids, document_info=info, loop_depths=depths).logits
        cache = build_model_cache(perturbed_model, loop_depth=depth)
        steps = [perturbed_model(ids[:, t:t + 1],
                                 document_info=slice_document_info(info, t, t + 1),
                                 loop_depths=depths, cache=cache).logits
                 for t in range(LENGTH)]
    assert (full - torch.cat(steps, dim=1)).abs().max().item() < TOLERANCE


def test_prefill_then_continuation_matches_full_forward(perturbed_model, sequence):
    ids, info = sequence
    depths = torch.full((1,), 4, dtype=torch.int64)
    with torch.no_grad():
        full = perturbed_model(ids, document_info=info, loop_depths=depths).logits
        cache = build_model_cache(perturbed_model, loop_depth=4)
        head = perturbed_model(ids[:, :6], document_info=slice_document_info(info, 0, 6),
                               loop_depths=depths, cache=cache).logits
        tail = perturbed_model(ids[:, 6:], document_info=slice_document_info(info, 6, LENGTH),
                               loop_depths=depths, cache=cache).logits
    assert (full - torch.cat([head, tail], dim=1)).abs().max().item() < TOLERANCE


def test_slicing_document_info_is_not_the_same_as_rebuilding_it():
    """The trap this whole file exists to catch.

    `build_document_info` on a slice marks its first position as a document
    start unconditionally.  During decoding that resets the KDA state on every
    token, silently, with finite and plausible-looking output.
    """
    ids = torch.tensor([[7, 8, 9, 4, 11, 12]])
    info = build_document_info(ids)
    for start in (1, 2, 4):
        sliced = slice_document_info(info, start, start + 1)
        rebuilt = build_document_info(ids[:, start:start + 1])
        assert bool(sliced.document_start_mask[0, 0]) == bool(info.document_start_mask[0, start])
        if not info.document_start_mask[0, start]:
            assert not bool(sliced.document_start_mask[0, 0])
            assert bool(rebuilt.document_start_mask[0, 0]), "the trap is gone; update this test"


def test_state_resets_at_a_document_boundary_during_decode(perturbed_model):
    """After an EOD, the next document must not see the previous one."""
    torch.manual_seed(5)
    ids = torch.randint(5, 49152, (1, 12))
    ids[0, 5] = 4
    info = build_document_info(ids)
    depths = torch.full((1,), 2, dtype=torch.int64)

    with torch.no_grad():
        cache = build_model_cache(perturbed_model, loop_depth=2)
        decoded = [perturbed_model(ids[:, t:t + 1],
                                   document_info=slice_document_info(info, t, t + 1),
                                   loop_depths=depths, cache=cache).logits
                   for t in range(12)]
        decoded = torch.cat(decoded, dim=1)

        changed = ids.clone()
        changed[0, :5] = torch.randint(5, 49152, (5,))
        changed_info = build_document_info(changed)
        cache = build_model_cache(perturbed_model, loop_depth=2)
        after = [perturbed_model(changed[:, t:t + 1],
                                 document_info=slice_document_info(changed_info, t, t + 1),
                                 loop_depths=depths, cache=cache).logits
                 for t in range(12)]
        after = torch.cat(after, dim=1)

    assert (decoded[0, 6:] - after[0, 6:]).abs().max().item() < TOLERANCE


def test_cache_reset_clears_every_level(perturbed_model, sequence):
    ids, info = sequence
    depths = torch.full((1,), 2, dtype=torch.int64)
    with torch.no_grad():
        cache = build_model_cache(perturbed_model, loop_depth=2)
        perturbed_model(ids, document_info=info, loop_depths=depths, cache=cache)
        cache.reset()
    for entry in cache.prelude_layers + cache.coda_layers:
        assert entry.kda is None or entry.kda.state is None
        assert entry.mla is None or entry.mla.latent is None
    for iteration in cache.loop_iterations:
        for entry in iteration.core_layers:
            assert entry.kda is None or entry.kda.state is None
            assert entry.mla is None or entry.mla.latent is None


def test_depth_and_cache_counts(perturbed_model):
    cache = build_model_cache(perturbed_model, loop_depth=8)
    assert cache.count("KDA") == 51
    assert cache.count("MLA") == 17
    with pytest.raises(RuntimeError, match="re-prefill"):
        cache.require_depth(4)
