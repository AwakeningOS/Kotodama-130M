"""The FLA boundary must keep working, and must be the only boundary.

`A_log` and `dt_bias` reach `chunk_kda` through its `**kwargs`, so nothing in
the type system or the call itself would complain if they stopped being read.
These tests fail loudly if that happens: each parameter is varied on its own,
with every other input held fixed, and the output has to move.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kotodama.fla_bridge import (  # noqa: E402
    KDA_HEAD_DIM,
    KDA_HEADS,
    KDA_LOWER_BOUND,
    call_chunk_kda_checked,
    fla_source_fingerprint,
    reference_log_decay,
    verify_fla_fingerprint,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def fixed_inputs(length: int = 64, batch: int = 1):
    """Identical tensors on every call: no dropout, no RNG, no state carried."""
    torch.manual_seed(0)
    shape = (batch, length, KDA_HEADS, KDA_HEAD_DIM)
    return {
        "q": torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(shape, device="cuda", dtype=torch.bfloat16),
        "raw_gate": torch.randn(shape, device="cuda", dtype=torch.float32),
        "raw_beta": torch.randn((batch, length, KDA_HEADS), device="cuda", dtype=torch.float32),
    }


def run_kda(A_log, dt_bias):
    output, _ = call_chunk_kda_checked(
        **fixed_inputs(), A_log=A_log, dt_bias=dt_bias, output_final_state=True
    )
    return output.float()


def zeros_A():
    return torch.zeros(KDA_HEADS, device="cuda", dtype=torch.float32)


def zeros_dt():
    return torch.zeros(KDA_HEADS * KDA_HEAD_DIM, device="cuda", dtype=torch.float32)


@CUDA
def test_fla_consumes_A_log():
    out_a = run_kda(A_log=zeros_A(), dt_bias=zeros_dt())
    out_b = run_kda(A_log=torch.full_like(zeros_A(), 2.0), dt_bias=zeros_dt())
    assert (out_a - out_b).abs().max().item() > 1.0e-4, "A_log was ignored by FLA"


@CUDA
def test_fla_consumes_dt_bias():
    out_a = run_kda(A_log=zeros_A(), dt_bias=zeros_dt())
    out_b = run_kda(A_log=zeros_A(), dt_bias=torch.full_like(zeros_dt(), 1.5))
    assert (out_a - out_b).abs().max().item() > 1.0e-4, "dt_bias was ignored by FLA"


@CUDA
def test_A_log_and_dt_bias_receive_finite_nonzero_gradients():
    """Being read on the forward pass is not enough; they must also train."""
    A_log = zeros_A().requires_grad_(True)
    dt_bias = zeros_dt().requires_grad_(True)
    output, _ = call_chunk_kda_checked(
        **fixed_inputs(), A_log=A_log, dt_bias=dt_bias, output_final_state=True
    )
    output.float().square().mean().backward()
    for name, parameter in (("A_log", A_log), ("dt_bias", dt_bias)):
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
        assert parameter.grad.abs().max().item() > 0.0, f"{name} gradient is all zero"


@CUDA
def test_log_decay_stays_inside_the_lower_bound():
    """Safe-gate output is `log_decay`, and lives in `[lower_bound, 0)`."""
    inputs = fixed_inputs()
    log_decay = reference_log_decay(inputs["raw_gate"], zeros_A(), zeros_dt())
    assert log_decay.min().item() >= KDA_LOWER_BOUND
    assert log_decay.max().item() < 0.0


def find_fla_imports(root: Path) -> list[tuple[str, int, str]]:
    """Real import statements only.

    A grep also matches comments, docstrings and audit notes, so it is a useful
    log but a bad gate.  The parse tree cannot be fooled that way.
    """
    found: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "fla" or alias.name.startswith("fla."):
                        found.append((str(path), node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "fla" or module.startswith("fla."):
                    found.append((str(path), node.lineno, module))
    return found


def test_fla_is_imported_in_exactly_one_file():
    """The `**kwargs` hazard is contained only while the boundary is."""
    imports = find_fla_imports(REPOSITORY_ROOT / "kotodama")
    offenders = {Path(path).name for path, _, _ in imports}
    assert offenders == {"fla_bridge.py"}, (
        "FLA may only be imported in fla_bridge.py; found "
        + ", ".join(f"{Path(p).name}:{line} ({module})" for p, line, module in imports)
    )


def test_fingerprint_pins_fla_and_not_torch():
    fingerprint = fla_source_fingerprint()
    for key in ("chunk_kda_source_path", "selected_backend_source_path"):
        assert "/fla/" in fingerprint[key], f"{key} does not point at FLA: {fingerprint[key]}"
        assert "_dynamo" not in fingerprint[key], f"{key} pinned the torch wrapper"
    assert len(fingerprint["chunk_kda_source_sha256"]) == 64


def test_fingerprint_mismatch_refuses_to_start():
    tampered = dict(fla_source_fingerprint())
    tampered["chunk_kda_source_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint"):
        verify_fla_fingerprint(tampered)
    verify_fla_fingerprint(fla_source_fingerprint())
