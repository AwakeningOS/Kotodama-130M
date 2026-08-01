"""Strict NoPE MLA: structure, causality, document isolation, cache identity."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.config import MLAConfig  # noqa: E402
from kotodama.mla import GatedNoPEMLA, MLACache  # noqa: E402
from kotodama.segments import build_document_info  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 768


def build(dtype=torch.float32, device="cpu"):
    torch.manual_seed(11)
    return GatedNoPEMLA(HIDDEN, MLAConfig()).to(device).to(dtype)


def test_exact_parameter_count():
    assert sum(t.numel() for t in build().parameters()) == 1_671_552


def test_qk_shared_dimension_is_zero():
    assert MLAConfig().qk_shared_head_dim == 0
    assert MLAConfig().use_rope is False


def test_no_rope_module_or_buffer():
    """A disabled rotary embedding still left in place is what the
    specification forbids, so check the object graph, not a config flag."""
    layer = build()
    names = [name.lower() for name, _ in layer.named_modules()]
    names += [name.lower() for name, _ in layer.named_buffers()]
    names += [name.lower() for name, _ in layer.named_parameters()]
    for forbidden in ("rope", "rotary", "inverse_frequency", "alibi", "position"):
        assert not any(forbidden in name for name in names), forbidden

    source = (REPOSITORY_ROOT / "kotodama" / "mla.py").read_text()
    tree = ast.parse(source)
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "apply_rotary" not in called


def test_future_token_causality():
    layer = build()
    ids = torch.randint(5, 49152, (1, 16))
    info = build_document_info(ids)
    x = torch.randn(1, 16, HIDDEN)

    original, _ = layer(x, document_info=info)
    changed = x.clone()
    changed[0, 8:] = torch.randn(8, HIDDEN)
    after, _ = layer(changed, document_info=info)
    assert (original[0, :8] - after[0, :8]).abs().max().item() < 1e-6


def test_cross_document_attention_is_blocked():
    layer = build()
    ids = torch.randint(5, 49152, (1, 16))
    ids[0, 7] = 4  # EOD: positions 8.. open a new document
    info = build_document_info(ids)
    x = torch.randn(1, 16, HIDDEN)

    original, _ = layer(x, document_info=info)
    changed = x.clone()
    changed[0, :8] = torch.randn(8, HIDDEN)
    after, _ = layer(changed, document_info=info)
    assert (original[0, 8:] - after[0, 8:]).abs().max().item() < 1e-6


def test_output_gate_is_applied_to_the_attention_output():
    """Zeroing the gate must zero the block, which is only true if the gate
    multiplies the output rather than the queries, keys or logits."""
    layer = build()
    with torch.no_grad():
        layer.output_gate.weight.zero_()
        layer.o_proj.weight.copy_(torch.eye(HIDDEN))
    x = torch.randn(1, 8, HIDDEN)
    output, _ = layer(x)
    # sigmoid(0) = 0.5, so the output is exactly half the ungated attention.
    with torch.no_grad():
        layer.output_gate.weight.fill_(0.0)
    assert torch.isfinite(output).all()

    query, key, value = layer.project(x)
    reference = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, scale=1.0 / 8.0)
    reference = reference.transpose(1, 2).reshape(1, 8, -1) * 0.5
    assert (output - reference).abs().max().item() < 1e-5


def test_full_forward_matches_single_token_decode():
    layer = build()
    ids = torch.randint(5, 49152, (1, 12))
    info = build_document_info(ids)
    x = torch.randn(1, 12, HIDDEN)
    full, _ = layer(x, document_info=info)

    cache = MLACache()
    outputs = []
    for t in range(12):
        step_info = build_document_info(ids[:, t : t + 1])
        step_info.segment_ids = info.segment_ids[:, t : t + 1]
        step, cache = layer(x[:, t : t + 1], document_info=step_info, cache=cache)
        outputs.append(step)
    decoded = torch.cat(outputs, dim=1)
    assert (full - decoded).abs().max().item() < 1e-5


def test_prefill_then_continuation_matches_full_forward():
    layer = build()
    ids = torch.randint(5, 49152, (1, 12))
    info = build_document_info(ids)
    x = torch.randn(1, 12, HIDDEN)
    full, _ = layer(x, document_info=info)

    cache = MLACache()
    prefill_info = build_document_info(ids[:, :8])
    prefill_info.segment_ids = info.segment_ids[:, :8]
    head, cache = layer(x[:, :8], document_info=prefill_info, cache=cache)

    tail_info = build_document_info(ids[:, 8:])
    tail_info.segment_ids = info.segment_ids[:, 8:]
    tail, cache = layer(x[:, 8:], document_info=tail_info, cache=cache)

    assert (full - torch.cat([head, tail], dim=1)).abs().max().item() < 1e-5


def test_cache_stores_latent_not_expanded_keys_and_values():
    layer = build()
    ids = torch.randint(5, 49152, (2, 12))
    x = torch.randn(2, 12, HIDDEN)
    cache = MLACache()
    layer(x, document_info=build_document_info(ids), cache=cache)
    assert cache.latent.shape == (2, 12, 128)
    assert cache.key_inv_rms.shape == (2, 12, 12)
    assert not hasattr(cache, "keys") and not hasattr(cache, "values")
    cached_per_token = cache.latent.shape[-1] + cache.key_inv_rms.shape[-1]
    expanded_per_token = 2 * 12 * 64
    assert cached_per_token == 140
    assert cached_per_token / expanded_per_token < 0.10


def test_cache_keeps_document_boundaries():
    """A cached key from a finished document must stay unreachable."""
    layer = build()
    ids = torch.randint(5, 49152, (1, 10))
    ids[0, 4] = 4
    info = build_document_info(ids)
    x = torch.randn(1, 10, HIDDEN)

    cache = MLACache()
    first_info = build_document_info(ids[:, :5])
    first_info.segment_ids = info.segment_ids[:, :5]
    layer(x[:, :5], document_info=first_info, cache=cache)

    second_info = build_document_info(ids[:, 5:])
    second_info.segment_ids = info.segment_ids[:, 5:]
    tail, _ = layer(x[:, 5:], document_info=second_info, cache=cache)

    reference_info = build_document_info(ids[:, 5:])
    reference_info.segment_ids = info.segment_ids[:, 5:]
    standalone, _ = layer(x[:, 5:], document_info=reference_info, cache=None)
    assert (tail - standalone).abs().max().item() < 1e-5


@pytest.mark.parametrize("length", [1, 5, 64])
def test_shapes(length):
    layer = build()
    x = torch.randn(2, length, HIDDEN)
    output, _ = layer(x)
    assert output.shape == (2, length, HIDDEN)
