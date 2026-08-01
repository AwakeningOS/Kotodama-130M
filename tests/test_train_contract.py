"""Product launcher contracts that must not drift between milestone runs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402
from kotodama.config import KotodamaConfig  # noqa: E402
from kotodama.model import build_model  # noqa: E402


def test_default_gate_is_the_first_whole_step_beyond_100m():
    assert train.DEFAULT_TARGET_TOKENS == 100_000_000
    assert train.target_steps(100_000_000) == 1_526
    assert train.target_steps(100_000_000) * train.TOKENS_PER_OPTIMIZER_STEP == 100_007_936


def test_later_milestones_have_stable_absolute_steps():
    assert train.target_steps(500_000_000) == 7_630
    assert train.target_steps(1_000_000_000) == 15_259
    assert train.target_steps(16_000_000_000) == 244_141


def test_intermediate_gate_has_no_cooldown():
    assert train.learning_rate(0) < train.PEAK_LR
    assert train.learning_rate(train.DEFAULT_WARMUP_STEPS - 1) == train.PEAK_LR
    assert train.learning_rate(1_526) == train.PEAK_LR


def test_product_source_fingerprint_is_sha256():
    value = train.source_fingerprint()
    assert len(value) == 64
    assert int(value, 16) >= 0


def test_source_migration_must_be_explicitly_recorded():
    assert train.checkpoint_source_is_compatible(
        {"source_fingerprint": "same"}, "same")
    assert train.checkpoint_source_is_compatible(
        {
            "source_fingerprint": "original",
            "resume_compatible_source_fingerprints": ["cleaned"],
        },
        "cleaned",
    )
    assert not train.checkpoint_source_is_compatible(
        {"source_fingerprint": "original"}, "cleaned")


def test_adamw_parameter_order_matches_the_saved_checkpoint_contract():
    with torch.device("meta"):
        model = build_model(KotodamaConfig())
    decay, no_decay = train.adamw_parameter_groups(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    decay_names = [names[id(parameter)] for parameter in decay]
    no_decay_names = [names[id(parameter)] for parameter in no_decay]

    assert len(decay_names) == 136
    assert sum(parameter.numel() for parameter in decay) == 128_664_576
    assert hashlib.sha256("\n".join(decay_names).encode()).hexdigest() == (
        "a603cfa14defe60d4db64f427fcef3ba046192c5cfc27cad8dbab07bc19a6273"
    )
    assert len(no_decay_names) == 105
    assert sum(parameter.numel() for parameter in no_decay) == 1_299_756
    assert hashlib.sha256("\n".join(no_decay_names).encode()).hexdigest() == (
        "b38fda232fb1c1e7b85265e32deedf1d7747223a5d03168ba37a142fab6d9e07"
    )
    assert len(set(decay_names + no_decay_names)) == len(decay_names) + len(no_decay_names)


def test_wall_clock_budget_is_exposed_by_the_product_launcher():
    source = (Path(__file__).resolve().parents[1] / "train.py").read_text()
    assert 'parser.add_argument("--max-minutes"' in source
    assert "time.monotonic() < deadline" in source
