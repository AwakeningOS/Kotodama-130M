"""Telemetry shared by training and the phase gates.

Two things are watched that a loss curve will not show.

**Whether the model is still echoing its input.**  Identity-initialised blocks
plus a tied output projection make the freshly built model reproduce the
current token, which puts it above the uniform-distribution loss and means the
first thing it has to learn is to stop copying.  `copy_rate` and `copy_margin`
say whether that has happened; the loss alone cannot distinguish "learning the
language" from "unlearning the copy".

**Whether the loop is doing anything.**  A stable injection bounds the
carry-over operator, not the composed map `Core(J(h, e))`, so neither
divergence nor collapse is ruled out by construction.  At initialisation the
relative change per iteration falls off geometrically -- 0.45, 0.14, 0.06,
0.02, ... -- so the state has effectively stopped moving by iteration six and
the deeper iterations are doing no work.  Whether training gives them work is
the open question about this architecture, and it is answered by watching this
series over the run, not by assuming it.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def copy_diagnostics(logits: torch.Tensor, input_ids: torch.Tensor,
                     labels: torch.Tensor) -> dict:
    """How strongly the model still just echoes its input.

    `copy_margin` is the current token's logit minus the best competing one, so
    a positive value means the model's top prediction is the token it was just
    given.  `target_margin` is the same quantity for the token that should
    actually come next.
    """
    flat = logits.float()
    current = flat.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    copy_rate = (flat.argmax(-1) == input_ids).float().mean()

    without_current = flat.scatter(-1, input_ids.unsqueeze(-1), float("-inf"))
    copy_margin = (current - without_current.max(-1).values).mean()

    valid = labels != -100
    safe = labels.clamp_min(0)
    target = flat.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    without_target = flat.scatter(-1, safe.unsqueeze(-1), float("-inf"))
    target_margin = ((target - without_target.max(-1).values) * valid).sum() / valid.sum()

    return {
        "copy_rate": copy_rate.item(),
        "mean_current_token_logit": current.mean().item(),
        "mean_target_token_logit": (target * valid).sum().item() / valid.sum().item(),
        "copy_margin": copy_margin.item(),
        "target_margin": target_margin.item(),
    }


@torch.no_grad()
def loop_telemetry(loop_states: list[torch.Tensor]) -> dict:
    """Per-iteration state RMS, relative change and cosine similarity."""
    rms, relative, cosine = [], [], []
    previous = None
    for state in loop_states:
        value = state.float()
        current_rms = value.pow(2).mean().sqrt().item()
        rms.append(current_rms)
        if previous is not None:
            previous_rms = previous.pow(2).mean().sqrt().item()
            relative.append((value - previous).pow(2).mean().sqrt().item()
                            / max(previous_rms, 1e-8))
            cosine.append(torch.nn.functional.cosine_similarity(
                value.flatten(), previous.flatten(), dim=0).item())
        previous = value
    return {
        "state_rms_per_loop": rms,
        "relative_state_change_per_loop": relative,
        "state_cosine_per_loop": cosine,
    }


@torch.no_grad()
def injection_telemetry(injection) -> dict:
    decay = injection.get_decay().detach().float()
    percentiles = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=decay.device)
    return {
        "injection_decay_min": decay.min().item(),
        "injection_decay_max": decay.max().item(),
        "injection_decay_percentiles": torch.quantile(decay, percentiles).tolist(),
        "injection_spectral_radius": decay.max().item(),
        "B_spectral_norm": torch.linalg.matrix_norm(
            injection.B.detach().float(), ord=2).item(),
    }


__all__ = ["copy_diagnostics", "injection_telemetry", "loop_telemetry"]
