#!/usr/bin/env python3
"""Generate a text continuation from a Kotodama checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kotodama.config import KDAConfig, KotodamaConfig, LoopConfig, MLAConfig
from kotodama.generate import generate
from kotodama.model import build_model


def config_from_dict(raw: dict) -> KotodamaConfig:
    values = dict(raw)
    values["kda"] = KDAConfig(**values["kda"])
    values["mla"] = MLAConfig(**values["mla"])
    values["loop"] = LoopConfig(**values["loop"])
    for key in ("prelude_pattern", "recurrent_core_pattern", "coda_pattern"):
        values[key] = tuple(values[key])
    return KotodamaConfig(**values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(payload["config"])
    model = build_model(config, fast_kda=device.type == "cuda")
    model.load_state_dict(payload["model"], strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    if tokenizer.vocab_size() != config.vocab_size:
        parser.error("tokenizer vocabulary does not match the checkpoint")
    ids = tokenizer.encode(args.prompt, out_type=int)
    if not ids:
        ids = [tokenizer.bos_id()]
    if len(ids) > config.context_length_train:
        parser.error(f"prompt exceeds {config.context_length_train} tokens")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        loop_depth=args.depth,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(tokenizer.decode(output[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
