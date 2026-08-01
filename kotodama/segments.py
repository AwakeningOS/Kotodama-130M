"""Document boundaries in a packed token stream.

Four different things are derived here, and they are deliberately separate
tensors because they answer four different questions about the same position:

    segment_ids          which document does this token belong to
    document_start_mask  is this token the first of a new document
    ntp_loss_mask        may this token predict the next one
    cu_seqlens           where every document begins, for the varlen kernels

Collapsing any two of them is a silent correctness bug.  In particular
`document_start_mask` is about position `t` beginning a document, while
`ntp_loss_mask` is about position `t` being allowed to predict `t+1` -- these
are offset from each other by one, and swapping them leaks one token across
every document boundary in the corpus.

Definitions, fixed by the specification:

    start(b, t)      = (t == 0) or (x[b, t-1] == EOD)
    loss_valid(b, t) = (x[b, t] != EOD)      and t is not the last position

An EOD token predicts the first token of the *next* document, which is not a
continuation of anything, so that position is excluded from the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kotodama.config import EOD_TOKEN_ID


@dataclass
class DocumentInfo:
    """Everything the blocks need to keep documents apart.

    The same object is passed to the prelude, to *every* core iteration and to
    the coda.  Dropping it after the first iteration would let the second
    iteration attend across documents that the first one kept separate.
    """

    segment_ids: torch.Tensor          # [B, L] int32, monotonic within a row
    document_start_mask: torch.Tensor  # [B, L] bool
    ntp_loss_mask: torch.Tensor        # [B, L] bool
    cu_seqlens: torch.Tensor           # [N+1] int32, flattened batch


def build_document_info(
    input_ids: torch.Tensor, eod_token_id: int = EOD_TOKEN_ID
) -> DocumentInfo:
    batch, length = input_ids.shape
    device = input_ids.device

    is_eod = input_ids == eod_token_id

    # A document starts at position 0, or immediately after an EOD.
    document_start_mask = torch.zeros((batch, length), dtype=torch.bool, device=device)
    document_start_mask[:, 0] = True
    if length > 1:
        document_start_mask[:, 1:] = is_eod[:, :-1]

    segment_ids = document_start_mask.to(torch.int32).cumsum(dim=1) - 1

    # An EOD may not predict the next token: that token opens a new document.
    ntp_loss_mask = ~is_eod
    ntp_loss_mask[:, -1] = False  # nothing follows the last position

    return DocumentInfo(
        segment_ids=segment_ids.to(torch.int32),
        document_start_mask=document_start_mask,
        ntp_loss_mask=ntp_loss_mask,
        cu_seqlens=cumulative_sequence_lengths(document_start_mask),
    )


def slice_document_info(info: DocumentInfo, start: int, end: int) -> DocumentInfo:
    """Take a window out of an existing `DocumentInfo`.

    Calling `build_document_info` on a slice is wrong for incremental decoding
    and wrong in a way that produces no error: position 0 of any slice is
    unconditionally marked as a document start, so a continuation token looks
    like the beginning of a new document and the KDA recurrent state is reset
    on every single decode step.  The output stays finite and plausible, and
    generation is quietly broken.

    This keeps the mask from the full sequence, where the boundaries are known.
    """
    return DocumentInfo(
        segment_ids=info.segment_ids[:, start:end],
        document_start_mask=info.document_start_mask[:, start:end],
        ntp_loss_mask=info.ntp_loss_mask[:, start:end],
        cu_seqlens=cumulative_sequence_lengths(info.document_start_mask[:, start:end]),
    )


def cumulative_sequence_lengths(document_start_mask: torch.Tensor) -> torch.Tensor:
    """`cu_seqlens` over the batch flattened to one row.

    Boundaries come from every document start *and* every row start.  Using row
    boundaries alone would let a recurrent state run from the end of one
    document into the beginning of the next; using document starts alone would
    let it run across the seam between two batch rows.
    """
    batch, length = document_start_mask.shape
    device = document_start_mask.device

    starts = document_start_mask.clone()
    starts[:, 0] = True  # every row begins a segment regardless of its content
    flat = starts.reshape(-1)
    offsets = torch.nonzero(flat, as_tuple=False).flatten()
    total = torch.tensor([batch * length], device=device, dtype=offsets.dtype)
    return torch.cat([offsets, total]).to(torch.int32)


def build_labels(
    input_ids: torch.Tensor, ntp_loss_mask: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Next-token targets with boundary positions masked out."""
    labels = torch.full_like(input_ids, ignore_index)
    labels[:, :-1] = input_ids[:, 1:]
    return labels.masked_fill(~ntp_loss_mask, ignore_index)


def attention_document_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """[B, 1, L, L] boolean: causal *and* same-document.

    `is_causal=True` alone does not separate packed documents -- it only stops
    a token attending to the future, not to a different document in its past.
    """
    _, length = segment_ids.shape
    device = segment_ids.device
    causal = torch.ones((length, length), dtype=torch.bool, device=device).tril()
    same_document = segment_ids[:, :, None] == segment_ids[:, None, :]
    return (causal[None, :, :] & same_document).unsqueeze(1)


__all__ = [
    "DocumentInfo",
    "attention_document_mask",
    "build_document_info",
    "build_labels",
    "cumulative_sequence_lengths",
    "slice_document_info",
]
