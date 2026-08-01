"""Pretraining for kotodama_stable_loop_130m_v2.

    .venv/bin/python train.py --run-dir runs/kotodama_stable_loop_seed11 --allow-gpu

What is deliberate here, and why
-------------------------------
**Extendable WSD plateau, no milestone cooldown.**  The 100M, 500M and 1B
checkpoints are continuation points.  Cooling down at each gate would make the
next phase restart from a deliberately weakened learning rate, so v2 warms up
once and remains on the plateau until a final release cooldown is requested.

**Loss on the final iteration only.**  Full BPTT already carries the gradient
through every executed iteration.  Attaching a loss to each one instead pulls
every intermediate state towards answering immediately.

**Depth is sampled per sequence and deterministically**, from
(run_seed, optimizer_step, micro_step, global_sequence_index).  A resumed run
must hand the same sequence the same depth, or "same tokens in the same order"
is quietly false.

**Sequences are bucketed by loop depth.**  The core iterates `max(depths)`
times for a micro-batch. Sorting one optimizer step by depth before forming
micro-batches reduces masked-out recurrent work. Shallow buckets run without
activation checkpointing; deeper buckets retain it to stay inside the product
memory bound. The live run log is the authority for actual throughput and VRAM.

**AdamW only.**  The product launcher has one optimizer recipe. Keeping a
failed optimizer behind a dormant flag would make manifests and handoffs
ambiguous, so there is no alternative optimizer path in this file.

**Both of the above are cost, not quality.**  The loss is identical across all
bucketing arms to four decimals, which is the point: regrouping rows must not
change the objective.  It is only exact because the loss is summed and divided by the optimizer
step's total valid-token count.  Averaging per-micro-batch means -- the
obvious thing -- weights each micro-batch equally regardless of how many
valid positions it holds, so regrouping rows would quietly change the
objective.

**The FLA fingerprint is checked before the first kernel call.**  A dependency
that changed underneath the run changes the model without changing this
repository.

**Checkpoints are hourly by wall clock, and the newest two are kept.**  This
run happens a few hours at a time, so the interval that matters is "how much
work does a kill cost", not "how many steps have passed" -- and the step rate
varies by a factor of two across the depth ramp, so a step interval would mean
two different answers at either end.  Two are kept rather than one because the
second exists to be fallen back on: an overwrite-in-place scheme has nothing to
resume from if the newest state turns out to hold a diverged optimiser.  A
non-finite dump is never pruned.  Resuming picks the
highest-numbered `step_*.pt`, so restarting is always the same command with
`--resume`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch

from kotodama.cache import build_model_cache  # noqa: F401  (audit surface)
from kotodama.config import ARCHITECTURE_ID, KotodamaConfig
from kotodama.data import LoaderState, PackedCorpus
from kotodama.depth_sampler import poisson_lambda, sample_depths
from kotodama.fla_bridge import fla_source_fingerprint, verify_fla_fingerprint
from kotodama.model import build_model
from kotodama.observe import copy_diagnostics, injection_telemetry, loop_telemetry
from kotodama.segments import build_document_info, build_labels

TOKENS_PER_OPTIMIZER_STEP = 65_536
# The live tournament stops first at 100M tokens.  Later gates are selected by
# resuming the same run with --target-tokens 500000000, 1000000000, and finally
# 16000000000.  The plateau has no milestone cooldown, because the 100M/500M/1B
# checkpoints are continuation trunks rather than terminal release weights.
DEFAULT_TARGET_TOKENS = 100_000_000
DEFAULT_WARMUP_STEPS = 100
PEAK_LR = 3.0e-4


def source_fingerprint() -> str:
    """Hash the product source that defines a checkpoint's forward and update."""
    root = Path(__file__).resolve().parent
    files = [root / "train.py", *sorted((root / "kotodama").glob("*.py"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def target_steps(target_tokens: int) -> int:
    """First whole optimizer step at or beyond the requested token gate."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    return math.ceil(target_tokens / TOKENS_PER_OPTIMIZER_STEP)


def learning_rate(step: int, warmup_steps: int = DEFAULT_WARMUP_STEPS) -> float:
    """Short warmup followed by a WSD plateau that can be extended safely."""
    if warmup_steps > 0 and step < warmup_steps:
        return PEAK_LR * (step + 1) / warmup_steps
    return PEAK_LR


def checkpoints(run_dir: Path) -> list[Path]:
    """Rolling checkpoints, oldest first.

    `trunk_step*.pt` and `non_finite.pt` are deliberately not matched: the trunk
    is the checkpoint the next phase grows from and the non-finite dump is
    evidence, so neither may be pruned by the rotation.
    """
    return sorted(run_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def newest_checkpoint(run_dir: Path) -> Path | None:
    """The newest numbered checkpoint, with no historical-layout fallback."""
    rolling = checkpoints(run_dir)
    return rolling[-1] if rolling else None


def adamw_parameter_groups(model):
    """Return decay/no-decay lists in the checkpoint-stable product order.

    Dynamical-system parameters, norm gains and biases are not decayed.  The
    decay list intentionally places directly classified matrices (the tied
    embedding) before the remaining matrix weights.  This is the parameter
    order used by the existing AdamW checkpoint, so changing it would silently
    attach saved moment tensors to the wrong parameters on resume.
    """
    direct_decay, matrix_decay, no_decay = [], [], []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))

        loop_parameter = "loop_injection" in name or "core_exit" in name
        matrix_weight = not (
            loop_parameter
            or parameter.ndim < 2
            or getattr(parameter, "_no_weight_decay", False)
            or "embed_tokens" in name
            or "lm_head" in name
            or "norm" in name
            or name.endswith(".bias")
        )
        if matrix_weight:
            matrix_decay.append(parameter)
        elif (parameter.ndim >= 2 and not loop_parameter
              and not getattr(parameter, "_no_weight_decay", False)):
            direct_decay.append(parameter)
        else:
            no_decay.append(parameter)
    return direct_decay + matrix_decay, no_decay


def build_optimizer(model, weight_decay: float):
    decay, no_decay = adamw_parameter_groups(model)
    return [torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )]


def checkpoint_source_is_compatible(payload: dict, current_fingerprint: str) -> bool:
    """Strict equality, or an explicitly recorded behaviour-neutral migration."""
    return (
        payload.get("source_fingerprint") == current_fingerprint
        or current_fingerprint in payload.get("resume_compatible_source_fingerprints", [])
    )


@torch.no_grad()
def validate(model, corpus, batches, batch_size, length, device, depths):
    """Held-out NLL at a set of fixed depths, each in its own forward pass.

    Never by attaching a readout to intermediate states of one pass: that is
    the supervision this architecture deliberately does not have.
    """
    model.eval()
    results = {}
    for depth in depths:
        state = LoaderState(cursor=0)
        total, count = 0.0, 0
        for _ in range(batches):
            ids, _, _ = corpus.batch(state, batch_size, length, device)
            info = build_document_info(ids)
            labels = build_labels(ids, info.ntp_loss_mask)
            output = model(ids, labels=labels, document_info=info,
                           loop_depths=torch.full((ids.shape[0],), depth,
                                                  dtype=torch.int64, device=device))
            total += float(output.loss)
            count += 1
        results[f"validation_nll_T{depth}"] = total / max(count, 1)
    model.train()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="packed corpus made by scripts/prepare_data.py",
    )
    parser.add_argument("--allow-gpu", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help="automatic stop gate (default: 100M tokens)")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="optional earlier absolute step stop for a bounded run")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="wall-clock budget including first-step compilation")
    parser.add_argument("--telemetry-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-depths", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--checkpoint-minutes", type=float, default=60.0)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--bucket-by-depth", dest="bucket", action="store_true",
                        default=True)
    parser.add_argument("--no-bucket-by-depth", dest="bucket", action="store_false")
    # Buckets no deeper than this run without activation checkpointing.  At 3
    # the measured peak is 19.45 GiB against the 22.0 gate; raising it puts a
    # depth-4 bucket in the batch at 22.21 GiB, which is over.
    parser.add_argument("--checkpoint-free-depth", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", dest="compile_step", action="store_true",
                        default=True, help="compile the recurrent step (default)")
    parser.add_argument("--no-compile", dest="compile_step", action="store_false")
    arguments = parser.parse_args()
    if arguments.max_minutes is not None and arguments.max_minutes <= 0.0:
        parser.error("--max-minutes must be positive")
    requested_target_steps = target_steps(arguments.target_tokens)
    stop_step = (requested_target_steps if arguments.max_steps is None
                 else min(requested_target_steps, arguments.max_steps))

    device = "cuda"
    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_dir = arguments.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = fla_source_fingerprint()
    product_source = source_fingerprint()

    config = KotodamaConfig(context_length_train=arguments.length)
    model = build_model(config).to(device).to(torch.bfloat16)
    model.train()
    if arguments.compile_step:
        # The loop stays in Python and only the step is compiled: measured at
        # +62.8% with peak reserved falling from 13.88 to 12.37 GiB.  Compiling
        # `forward` instead would unroll `range(max_depth)` once per depth and
        # recompile as the sampler moves between 2 and 8.
        model.recurrent_step = torch.compile(model.recurrent_step)
    optimizers = build_optimizer(model, arguments.weight_decay)

    corpus = PackedCorpus(arguments.data_dir, "train")
    validation = PackedCorpus(arguments.data_dir, "validation")
    loader_state = LoaderState()

    accumulation = TOKENS_PER_OPTIMIZER_STEP // (arguments.micro_batch * arguments.length)
    assert accumulation * arguments.micro_batch * arguments.length == TOKENS_PER_OPTIMIZER_STEP

    step, seen_tokens, sequences_seen = 0, 0, 0
    resume_from = newest_checkpoint(run_dir)
    if arguments.resume and resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        if payload["architecture_id"] != ARCHITECTURE_ID:
            raise RuntimeError(f"checkpoint is {payload['architecture_id']}")
        if not checkpoint_source_is_compatible(payload, product_source):
            raise RuntimeError(
                "checkpoint source does not match the current v2 product source")
        if payload.get("config") != asdict(config):
            raise RuntimeError("checkpoint config does not match the current v2 config")
        verify_fla_fingerprint(payload["fla_fingerprint"])
        model.load_state_dict(payload["model"])
        saved_optimizers = payload["optimizer"]
        # The earliest AdamW-only checkpoints stored one state dict directly;
        # current checkpoints always store a list, even when it has one item.
        if isinstance(saved_optimizers, dict):
            saved_optimizers = [saved_optimizers]
        if len(saved_optimizers) != len(optimizers):
            raise RuntimeError(
                f"checkpoint has {len(saved_optimizers)} optimizer states, "
                f"current run needs {len(optimizers)}")
        for optimizer, saved in zip(optimizers, saved_optimizers):
            optimizer.load_state_dict(saved)
        loader_state.cursor = payload["cursor"]
        step, seen_tokens = payload["step"], payload["seen_tokens"]
        sequences_seen = payload["global_sequences_seen"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]])
        print(f"[resume] {resume_from.name}: step={step} tokens={seen_tokens:,}",
              flush=True)
    else:
        verify_fla_fingerprint(fingerprint)

    manifest = {
        "architecture_id": ARCHITECTURE_ID,
        "parameters": model.parameter_count(),
        "source_fingerprint": product_source,
        "config": asdict(config),
        "optimizer_contract": "adamw_wsd_v2",
        "compiled_recurrent_step": arguments.compile_step,
        "bucket_by_depth": arguments.bucket,
        "checkpoint_free_depth": arguments.checkpoint_free_depth,
        "loop_contract": "stable_diagonal_like_init_full_bptt_v2",
        "depth_sampler_contract": "bounded_poisson_ramp_with_max_depth_tail_v2",
        "tokens_per_optimizer_step": TOKENS_PER_OPTIMIZER_STEP,
        "target_tokens_requested": arguments.target_tokens,
        "target_tokens_effective": requested_target_steps * TOKENS_PER_OPTIMIZER_STEP,
        "target_steps": requested_target_steps,
        "stop_step": stop_step,
        "max_minutes": arguments.max_minutes,
        "warmup_steps": arguments.warmup_steps,
        "micro_batch": arguments.micro_batch,
        "gradient_accumulation": accumulation,
        "fla_fingerprint": fingerprint,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[start] {manifest['parameters']:,} params, "
          f"{TOKENS_PER_OPTIMIZER_STEP:,} tokens/step, micro {arguments.micro_batch} "
          f"x accumulation {accumulation}", flush=True)

    stopping = {"now": False}

    def request_stop(*_):
        stopping["now"] = True
        print("[signal] finishing this step, then saving", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # SIGHUP too: the interpreter has to survive its launching shell going away,
    # or a nightly stop kills it between checkpoints and the graceful save never
    # runs.  That is not hypothetical -- it is how the first v3 attempt lost its
    # state, because the signal reached the wrapping shell and not this process.
    signal.signal(signal.SIGHUP, request_stop)

    # The PID to signal, written where the run lives.  `kill -TERM $(cat ...)`
    # is then the whole stop procedure, and it reaches the interpreter rather
    # than whatever launched it.
    pid_file = run_dir / "train.pid"
    pid_file.write_text(f"{os.getpid()}\n")

    def save(path: Path, stats=None) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save({
            "architecture_id": ARCHITECTURE_ID,
            "source_fingerprint": product_source,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": [optimizer.state_dict() for optimizer in optimizers],
            "cursor": loader_state.cursor,
            "step": step,
            "seen_tokens": seen_tokens,
            "global_sequences_seen": sequences_seen,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "fla_fingerprint": fingerprint,
            "stats": stats,
        }, temporary)
        os.replace(temporary, path)

    def save_rolling(stats=None) -> Path:
        """Write `step_<n>.pt` and keep only the newest `--keep-checkpoints`.

        Numbered rather than overwriting one file, so that a resume always has a
        second, older state to fall back on if the newest one turns out to hold
        a diverged optimiser.  Pruning happens strictly after the new file is in
        place: a crash during pruning costs disk, a crash during an overwrite
        would cost the run.
        """
        path = run_dir / f"step_{step:07d}.pt"
        save(path, stats)
        for stale in checkpoints(run_dir)[:-max(arguments.keep_checkpoints, 1)]:
            stale.unlink(missing_ok=True)
        return path

    log_file = (run_dir / "train.jsonl").open("a", buffering=1)
    started = time.perf_counter()
    last_checkpoint = time.monotonic()
    deadline = (None if arguments.max_minutes is None
                else time.monotonic() + arguments.max_minutes * 60.0)

    while (step < stop_step and not stopping["now"]
           and (deadline is None or time.monotonic() < deadline)):
        current_lr = learning_rate(step, arguments.warmup_steps)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)

        step_start = time.perf_counter()
        totals = torch.zeros(1, device=device)
        first_state, telemetry = None, None

        # Read the whole optimizer step, then regroup by depth.  Depth is a
        # function of the global sequence index, so reordering rows does not
        # change which depth any sequence receives.
        rows, depth_rows = [], []
        for micro in range(accumulation):
            ids, _, _ = corpus.batch(loader_state, arguments.micro_batch,
                                     arguments.length, device)
            sequence_ids = torch.arange(
                sequences_seen, sequences_seen + ids.shape[0], device=device)
            depth_rows.append(sample_depths(
                seen_tokens, sequence_ids, run_seed=arguments.seed,
                optimizer_step=step, micro_step=micro,
                minimum_depth=config.loop.train_min_depth,
                maximum_depth=config.loop.train_max_depth,
                ramp_tokens=config.loop.depth_ramp_tokens,
                maximum_depth_probability=config.loop.max_depth_probability))
            sequences_seen += ids.shape[0]
            rows.append(ids)
        all_ids = torch.cat(rows, dim=0)
        all_depths = torch.cat(depth_rows, dim=0)
        if arguments.bucket:
            order = torch.argsort(all_depths, stable=True)
            all_ids, all_depths = all_ids[order], all_depths[order]
        depth_histogram = torch.bincount(all_depths.cpu(), minlength=9)

        groups = [(all_ids[i * arguments.micro_batch:(i + 1) * arguments.micro_batch],
                   all_depths[i * arguments.micro_batch:(i + 1) * arguments.micro_batch])
                  for i in range(accumulation)]
        infos = [build_document_info(ids) for ids, _ in groups]
        labels = [build_labels(ids, info.ntp_loss_mask)
                  for (ids, _), info in zip(groups, infos)]
        # Exact step mean: sum the losses, divide by the step's own token count.
        total_labels = sum(int((label != -100).sum()) for label in labels)

        for micro, ((ids, depths), info, label) in enumerate(zip(groups, infos, labels)):
            # Shallow buckets fit without checkpointing; deep ones do not.
            model.gradient_checkpointing = (
                int(depths.max()) > arguments.checkpoint_free_depth)
            wants_telemetry = micro == 0 and step % arguments.telemetry_every == 0
            output = model(ids, labels=label, document_info=info, loop_depths=depths,
                           return_loop_states=wants_telemetry)
            (output.loss_sum / total_labels).backward()
            totals += output.loss_sum.detach()
            if wants_telemetry:
                first_state = output.final_state.detach()
                telemetry = {
                    **copy_diagnostics(output.logits.detach(), ids, label),
                    **loop_telemetry(output.loop_states),
                }

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.grad_clip)
        for optimizer in optimizers:
            optimizer.step()

        step += 1
        seen_tokens += TOKENS_PER_OPTIMIZER_STEP
        duration = time.perf_counter() - step_start
        loss_value = float(totals) / max(total_labels, 1)

        record = {
            "step": step,
            "seen_tokens": seen_tokens,
            "global_sequences_seen": sequences_seen,
            "loss": loss_value,
            "grad_norm": float(norm),
            "lr": current_lr,
            "step_seconds": duration,
            "tokens_per_second": TOKENS_PER_OPTIMIZER_STEP / duration,
            "elapsed_hours": (time.perf_counter() - started) / 3600.0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            "poisson_lambda": poisson_lambda(
                seen_tokens, config.loop.depth_ramp_tokens,
                config.loop.train_min_depth, config.loop.train_max_depth),
            "realized_mean_depth": float(
                (depth_histogram * torch.arange(9)).sum() / depth_histogram.sum()),
            "depth_histogram": depth_histogram[2:9].tolist(),
            "executed_loop_iterations": sum(
                int(d.max()) * d.numel() for _, d in groups),
            "required_loop_iterations": int(all_depths.sum()),
        }
        if telemetry is not None:
            record["loop"] = {
                "state_rms": float(first_state.float().pow(2).mean().sqrt()),
                **injection_telemetry(model.loop_injection),
                **telemetry,
            }

        if not math.isfinite(loss_value):
            log_file.write(json.dumps({"step": step, "fatal": "non_finite_loss"}) + "\n")
            save(run_dir / "non_finite.pt", record)
            print("[fatal] non-finite loss", flush=True)
            return 1

        if step % arguments.validate_every == 0:
            record.update(validate(
                model, validation, arguments.validation_batches,
                arguments.micro_batch, arguments.length, device,
                arguments.validation_depths))

        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step % 10 == 0 or step <= 5:
            extra = "".join(
                f" T{d}:{record[f'validation_nll_T{d}']:.3f}"
                for d in arguments.validation_depths
                if f"validation_nll_T{d}" in record)
            print(f"step {step:>6} | {seen_tokens/1e6:8.1f}M tok | "
                  f"loss {loss_value:.4f} | gnorm {record['grad_norm']:6.2f} | "
                  f"T~{record['realized_mean_depth']:.2f} | "
                  f"{record['tokens_per_second']:7.0f} tok/s{extra}", flush=True)

        # Wall-clock, not step count: the run is a few hours a night and the
        # step rate varies by a factor of two across the depth ramp, so a step
        # interval means a different amount of lost work at either end.
        if time.monotonic() - last_checkpoint >= arguments.checkpoint_minutes * 60:
            written = save_rolling(record)
            last_checkpoint = time.monotonic()
            print(f"[checkpoint] {written.name}", flush=True)
    # Stopping for the night is the normal case, so this is the checkpoint that
    # matters most: it costs whatever was done since the last hourly one.
    print(f"[checkpoint] {save_rolling().name}", flush=True)
    pid_file.unlink(missing_ok=True)
    log_file.close()
    print(f"[done] step={step} tokens={seen_tokens:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
