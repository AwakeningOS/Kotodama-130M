"""The single point of contact with `flash-linear-attention`.

Why this is its own module
--------------------------
`chunk_kda` in FLA 0.5.2 ends its signature with `**kwargs`, and `A_log` and
`dt_bias` -- the two parameters that decide the entire forget-gate -- arrive
through it rather than as named arguments.  They were verified to take effect
on this install (changing `A_log` moves the output by 5.8e-2, `dt_bias` by
2.3e-2), but `**kwargs` means a misspelling or an upstream rename would be
silently absorbed and the gate would quietly revert to its default.  No
exception, no warning, a slightly different model.

So the boundary is confined here.  `chunk_kda` is imported in exactly one file,
which makes the guarantee greppable:

    grep -rn "from fla" kotodama/ | grep -v fla_bridge.py

must return nothing.  `call_chunk_kda_checked` takes keyword-only arguments,
accepts no `**kwargs` of its own, and asserts the shapes and dtypes that the
kernel will not check.  Sensitivity tests cover the other direction: that FLA
is still reading what we pass.

Naming
------
The safe-gate output is `log_decay`, not `gate`.  It is `log alpha`, lives in
`[lower_bound, 0)`, and calling it a "gate" invites confusion with the KDA
*output* gate, which is a different tensor doing a different job.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
from fla.ops.kda import chunk_kda

# Frozen call contract, from the implementation specification section 8.10.
KDA_SCALE = 0.125
KDA_LOWER_BOUND = -5.0
KDA_HEADS = 12
KDA_HEAD_DIM = 64


def call_chunk_kda_checked(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """The only permitted call into FLA.

    With `safe_gate=True` and `lower_bound=-5.0`, FLA switches the gate
    activation from `-exp(A_log) * softplus(g + dt_bias)` to
    `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, which is the
    specification's formula and clamps `log_decay` into `[-5, 0)`.

    `raw_gate` and `raw_beta` are pre-activation: the kernel applies the gate
    activation and the beta sigmoid itself, and Q/K L2 normalisation too.
    Applying any of them outside as well would double them.
    """
    assert A_log.shape == (KDA_HEADS,), A_log.shape
    assert dt_bias.shape == (KDA_HEADS * KDA_HEAD_DIM,), dt_bias.shape
    assert A_log.dtype == torch.float32, A_log.dtype
    assert dt_bias.dtype == torch.float32, dt_bias.dtype

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=KDA_SCALE,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=KDA_LOWER_BOUND,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )


def reference_log_decay(
    raw_gate: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> torch.Tensor:
    """`log_decay = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.

    The torch-side statement of what the kernel is supposed to compute, so the
    two can be compared instead of trusting the kernel's own documentation.
    """
    heads = raw_gate.shape[-2]
    gate = raw_gate.float() + dt_bias.view(heads, -1)
    return KDA_LOWER_BOUND * torch.sigmoid(A_log.view(heads, 1).float().exp() * gate)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fla_source_fingerprint() -> dict:
    """Identify the source that will actually run, not just the version string.

    A version number does not distinguish a stock wheel from a locally edited
    one, so the audit contract is the SHA-256 of the executing files.
    """
    import fla
    from fla.ops.kda import chunk as kda_chunk

    package_root = Path(fla.__file__).resolve().parent
    # `chunk_kda` is wrapped by a torch.compile decorator, so asking it for its
    # source file yields `torch/_dynamo/eval_frame.py` -- pinning torch's
    # dispatcher instead of the KDA kernel, which would leave an FLA change
    # invisible to the fingerprint.  Unwrap before asking.
    unwrapped = inspect.unwrap(chunk_kda)
    chunk_source = Path(inspect.getsourcefile(unwrapped)).resolve()
    backend_source = Path(kda_chunk.__file__).resolve()
    if "_dynamo" in chunk_source.parts or "torch" in chunk_source.parts:
        # Unwrapping did not reach FLA; fall back to the module that defines it
        # rather than silently fingerprinting the wrong file.
        chunk_source = backend_source

    # A fingerprint of the wrong file is worse than none: it would keep passing
    # while the kernel underneath changed.
    for source in (chunk_source, backend_source):
        assert source.name == "chunk.py", source
        assert "/torch/" not in source.as_posix(), source
        posix = source.as_posix()
        assert "/fla/ops/kda/" in posix or posix.startswith(package_root.as_posix()), source
    return {
        "fla_package_version": fla.__version__,
        "fla_package_root": str(package_root),
        "chunk_kda_qualified_name": f"{chunk_kda.__module__}.{chunk_kda.__qualname__}",
        "chunk_kda_source_path": str(chunk_source),
        "chunk_kda_source_sha256": _sha256(chunk_source),
        "selected_backend_source_path": str(backend_source),
        "selected_backend_source_sha256": _sha256(backend_source),
        "chunk_kda_signature": str(inspect.signature(chunk_kda)),
    }


def verify_fla_fingerprint(expected: dict) -> None:
    """Refuse to start when the executing FLA source has changed.

    Called at training start-up.  A silently upgraded dependency changes the
    model without changing the code, so continuing is worse than stopping.
    """
    current = fla_source_fingerprint()
    mismatched = [
        key
        for key in ("chunk_kda_source_sha256", "selected_backend_source_sha256",
                    "chunk_kda_qualified_name")
        if expected.get(key) != current.get(key)
    ]
    if mismatched:
        lines = [f"  {key}:\n    recorded={expected.get(key)}\n    current ={current.get(key)}"
                 for key in mismatched]
        raise RuntimeError(
            "FLA source fingerprint does not match the audited environment.\n"
            + "\n".join(lines)
            + "\n\nThis changes the model without changing this repository. "
            "Update the audit record deliberately before continuing."
        )


__all__ = [
    "KDA_HEADS",
    "KDA_HEAD_DIM",
    "KDA_LOWER_BOUND",
    "KDA_SCALE",
    "call_chunk_kda_checked",
    "fla_source_fingerprint",
    "reference_log_decay",
    "verify_fla_fingerprint",
]
