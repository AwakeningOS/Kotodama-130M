"""Packed-corpus loader for the pool-v2-30b document-indexed shards.

Format, read from the manifest and verified against the files:

  .bin   uint16 token stream
  .idx   int64 document start offsets, length documents+1, last entry = tokens

Two properties this loader is responsible for:

**Sources are mixed from step 1.**  Both predecessor projects were burned by
reading one giant shard at a time: the entire warmup then sees a single
distribution, which biases early representation formation and, worse,
contaminates any later interpretation of what the architecture learned.  This
loader interleaves shards from every source deterministically from the first
batch.

**Document boundaries are preserved.**  Sequences are packed contiguously, but
each token carries a document id so attention and the KDA recurrence can be
told where one document ends.  A recurrent state that carries across a document
boundary is a silent correctness bug, not a rounding error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EOD_DEFAULT = 4


@dataclass
class Shard:
    source: str
    bin_path: Path
    idx_path: Path
    tokens: int

    def open_tokens(self) -> np.ndarray:
        return np.memmap(self.bin_path, dtype=np.uint16, mode="r")


@dataclass
class LoaderState:
    """Everything needed to resume the stream exactly."""

    cursor: int = 0

    def to_json(self) -> dict:
        return {"cursor": self.cursor}


def source_of(name: str) -> str:
    return name.split("-")[0]


class PackedCorpus:
    def __init__(self, root: str | Path, split: str = "train", eod_token_id: int = EOD_DEFAULT):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.eod_token_id = manifest.get("eod_token_id", eod_token_id)
        entries = manifest["splits"][split]["shards"]
        self.total_tokens = manifest["splits"][split]["tokens"]

        shards = [
            Shard(
                source=source_of(entry["path"]),
                bin_path=self.root / entry["path"],
                idx_path=self.root / entry["index"],
                tokens=entry["tokens"],
            )
            for entry in entries
        ]
        self.shards = self._interleave(shards)
        self._offsets = np.cumsum([0] + [shard.tokens for shard in self.shards])
        self._open: dict[int, np.ndarray] = {}

    @staticmethod
    def _interleave(shards: list[Shard]) -> list[Shard]:
        """Round-robin across sources so every source appears from the start.

        Deterministic and stateless: the same shard list always produces the
        same order, so a resumed run reads the same tokens.
        """
        by_source: dict[str, list[Shard]] = {}
        for shard in sorted(shards, key=lambda s: s.bin_path.name):
            by_source.setdefault(shard.source, []).append(shard)
        order = sorted(by_source)
        merged: list[Shard] = []
        position = 0
        while any(position < len(by_source[name]) for name in order):
            for name in order:
                if position < len(by_source[name]):
                    merged.append(by_source[name][position])
            position += 1
        return merged

    def _tokens_for(self, index: int) -> np.ndarray:
        if index not in self._open:
            # Keep the map cache small; shards are large and memmap costs address
            # space, not resident memory.
            if len(self._open) > 8:
                self._open.pop(next(iter(self._open)))
            self._open[index] = self.shards[index].open_tokens()
        return self._open[index]

    def read(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens from the concatenated stream, wrapping at the end."""
        out = np.empty(length, dtype=np.uint16)
        written = 0
        position = start % int(self._offsets[-1])
        while written < length:
            shard_index = int(np.searchsorted(self._offsets, position, side="right") - 1)
            local = position - int(self._offsets[shard_index])
            tokens = self._tokens_for(shard_index)
            take = min(length - written, len(tokens) - local)
            out[written : written + take] = tokens[local : local + take]
            written += take
            position += take
            if position >= int(self._offsets[-1]):
                position = 0
        return out

    def batch(
        self,
        state: LoaderState,
        batch_size: int,
        length: int,
        device: str = "cuda",
    ):
        """One packed batch, advancing `state` in place.

        Returns input ids, labels and document ids.  `length + 1` tokens are read
        per row so the labels are the inputs shifted by one without borrowing a
        token from the next batch.
        """
        rows_in = np.empty((batch_size, length), dtype=np.int64)
        rows_out = np.empty((batch_size, length), dtype=np.int64)
        rows_doc = np.empty((batch_size, length), dtype=np.int32)

        for row in range(batch_size):
            window = self.read(state.cursor, length + 1).astype(np.int64)
            rows_in[row] = window[:-1]
            rows_out[row] = window[1:]
            # A new document starts on the token *after* an EOD.
            boundary = np.zeros(length, dtype=np.int32)
            boundary[1:] = (window[:-2] == self.eod_token_id).astype(np.int32)
            rows_doc[row] = np.cumsum(boundary)
            state.cursor += length

        return (
            torch.from_numpy(rows_in).to(device, non_blocking=True),
            torch.from_numpy(rows_out).to(device, non_blocking=True),
            torch.from_numpy(rows_doc).to(device, non_blocking=True),
        )


def cumulative_sequence_lengths(document_id: torch.Tensor) -> torch.Tensor:
    """`cu_seqlens` for the variable-length KDA kernel.

    The kernel wants the batch flattened to one row, with a boundary wherever a
    document changes *and* at every row boundary, so no recurrent state survives
    across either.
    """
    batch, length = document_id.shape
    flat = document_id.reshape(-1)
    changed = torch.zeros_like(flat, dtype=torch.bool)
    changed[1:] = flat[1:] != flat[:-1]
    row_start = torch.arange(batch * length, device=flat.device) % length == 0
    starts = torch.nonzero(changed | row_start, as_tuple=False).flatten()
    total = torch.tensor([batch * length], device=flat.device, dtype=starts.dtype)
    return torch.cat([starts, total]).to(torch.int32)


__all__ = [
    "LoaderState",
    "PackedCorpus",
    "Shard",
    "cumulative_sequence_lengths",
]
