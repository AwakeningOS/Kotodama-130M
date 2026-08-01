#!/usr/bin/env python3
"""Pack text or JSONL documents into Kotodama's uint16 corpus format."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import sentencepiece as spm

VOCAB_SIZE = 49_152
EOD_ID = 4
EOD_PIECE = "<|eod|>"


def iter_documents(paths: Iterable[Path], text_field: str = "text") -> Iterable[str]:
    """Yield one document per non-empty text line or JSONL record."""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get(text_field)
                if not isinstance(text, str):
                    raise TypeError(f"{path}:{line_number}: missing string field {text_field!r}")
                if text.strip():
                    yield text.strip()
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield line.strip()


def validate_tokenizer(tokenizer: spm.SentencePieceProcessor) -> None:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(f"Kotodama requires {VOCAB_SIZE:,} pieces")
    if tokenizer.piece_to_id(EOD_PIECE) != EOD_ID or tokenizer.id_to_piece(EOD_ID) != EOD_PIECE:
        raise ValueError(f"Kotodama requires {EOD_PIECE} at token ID {EOD_ID}")


def pack_split(
    tokenizer: spm.SentencePieceProcessor,
    paths: list[Path],
    output_dir: Path,
    split: str,
    text_field: str,
) -> dict:
    token_chunks: list[np.ndarray] = []
    offsets = [0]
    for document in iter_documents(paths, text_field=text_field):
        ids = tokenizer.encode(document, out_type=int)
        if not ids:
            continue
        ids.append(EOD_ID)
        chunk = np.asarray(ids, dtype=np.uint16)
        token_chunks.append(chunk)
        offsets.append(offsets[-1] + len(chunk))
    if not token_chunks:
        raise ValueError(f"{split} contains no non-empty documents")

    stem = f"{split}-00000"
    bin_path = output_dir / f"{stem}.bin"
    idx_path = output_dir / f"{stem}.idx"
    np.concatenate(token_chunks).tofile(bin_path)
    np.asarray(offsets, dtype=np.int64).tofile(idx_path)
    return {
        "tokens": offsets[-1],
        "documents": len(token_chunks),
        "shards": [
            {
                "path": bin_path.name,
                "index": idx_path.name,
                "tokens": offsets[-1],
                "documents": len(token_chunks),
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    validate_tokenizer(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "kotodama-packed-v1",
        "tokenizer_model": args.tokenizer_model.name,
        "vocab_size": VOCAB_SIZE,
        "eod_token_id": EOD_ID,
        "splits": {
            "train": pack_split(
                tokenizer, args.train, args.output_dir, "train", args.text_field
            ),
            "validation": pack_split(
                tokenizer, args.validation, args.output_dir, "validation", args.text_field
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
