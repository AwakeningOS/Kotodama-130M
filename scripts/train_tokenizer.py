#!/usr/bin/env python3
"""Train a SentencePiece tokenizer compatible with Kotodama-130M."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

VOCAB_SIZE = 49_152
EOD_PIECE = "<|eod|>"
EOD_ID = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--model-prefix", type=Path, required=True)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    args = parser.parse_args()

    for path in args.input:
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=[str(path) for path in args.input],
        model_prefix=str(args.model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=args.character_coverage,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=[EOD_PIECE],
        hard_vocab_limit=True,
    )

    tokenizer = spm.SentencePieceProcessor(model_file=f"{args.model_prefix}.model")
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise RuntimeError(f"expected {VOCAB_SIZE} pieces, got {tokenizer.vocab_size()}")
    if tokenizer.piece_to_id(EOD_PIECE) != EOD_ID:
        raise RuntimeError(f"{EOD_PIECE} must have ID {EOD_ID}")
    print(f"wrote {args.model_prefix}.model ({tokenizer.vocab_size():,} pieces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
