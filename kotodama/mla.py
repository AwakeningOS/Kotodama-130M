"""Gated multi-head latent attention, strict NoPE.

Ported from `KaiNomos-747M/architecture/mla.py` and reduced to the frozen
Kotodama configuration.

Strict NoPE means exactly that: no rotary embedding, no ALiBi, no learned
positional table, and `qk_shared_head_dim = 0` so there is not even a shared
key sub-dimension where a positional channel could hide.  Order information
reaches this block through the residual stream, because the KDA layers ahead of
it are recurrent and their output is position-dependent by construction.  The
attention then reads that as content.

Latent projections keep the parameter count down: queries go 768 -> 128 -> 768
and keys/values 768 -> 128 -> 1536, each through an RMSNorm on the latent.

1,671,552 parameters:
    query path       196,736
    key/value path   295,040
    Q/K head norms       128
    output gate      589,824
    output proj      589,824
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kotodama.config import MLAConfig
from kotodama.layers import RMSNorm, accumulation_dtype
from kotodama.segments import DocumentInfo, attention_document_mask


class MLACache:
    """The 128-wide KV latent and QK-normalization statistics.

    The cache retains the shared 128-wide latent and twelve inverse-RMS scalars:
    140 elements per token. The key projection is absorbed into the query
    during decode; the dynamic RMS term is the per-token statistic that makes
    this exactly compatible with post-projection QK normalization.
    """

    def __init__(self, latent=None, key_inv_rms=None, segment_ids=None):
        self.latent = latent              # [B, past, kv_lora_rank]
        self.key_inv_rms = key_inv_rms    # [B, past, H]
        self.segment_ids = segment_ids  # [B, past]

    def reset(self) -> None:
        self.latent = None
        self.key_inv_rms = None
        self.segment_ids = None

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Invalidate selected batch rows without disturbing other sessions."""
        rows = rows.to(dtype=torch.bool, device=(self.latent.device
                                                 if self.latent is not None
                                                 else rows.device))
        if not bool(rows.any()) or self.latent is None:
            return
        if bool(rows.all()):
            self.reset()
            return
        self.latent[rows] = 0
        self.key_inv_rms[rows] = 0
        # Segment ids are non-negative.  The sentinel can never match a new
        # document and therefore makes every invalidated key unreachable.
        self.segment_ids[rows] = torch.iinfo(self.segment_ids.dtype).min


class GatedNoPEMLA(nn.Module):
    def __init__(self, hidden_size: int, config: MLAConfig):
        super().__init__()
        assert config.use_rope is False
        assert config.qk_shared_head_dim == 0
        self.config = config
        self.hidden_size = hidden_size
        heads = config.num_heads
        self.head_dim = config.qk_nope_head_dim
        self.value_head_dim = config.v_head_dim

        self.q_down = nn.Linear(hidden_size, config.q_lora_rank, bias=False)
        self.q_latent_norm = RMSNorm(config.q_lora_rank)
        self.q_up = nn.Linear(config.q_lora_rank, heads * self.head_dim, bias=False)

        self.kv_down = nn.Linear(hidden_size, config.kv_lora_rank, bias=False)
        self.kv_latent_norm = RMSNorm(config.kv_lora_rank)
        self.kv_up = nn.Linear(
            config.kv_lora_rank, heads * (self.head_dim + self.value_head_dim), bias=False)

        self.q_head_norm = RMSNorm(self.head_dim)
        self.k_head_norm = RMSNorm(self.head_dim)

        # Full rank, and applied to the attention output -- never to the query,
        # the key, or the attention logits.
        self.output_gate = nn.Linear(hidden_size, heads * self.value_head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.value_head_dim, hidden_size, bias=False)

    def project_with_latent(self, x: torch.Tensor):
        batch, length, _ = x.shape
        heads = self.config.num_heads

        query = self.q_up(self.q_latent_norm(self.q_down(x)))
        query = query.view(batch, length, heads, self.head_dim)

        latent = self.kv_latent_norm(self.kv_down(x))
        key_value = self.kv_up(latent)
        key_value = key_value.view(batch, length, heads, self.head_dim + self.value_head_dim)
        raw_key, value = key_value.split([self.head_dim, self.value_head_dim], dim=-1)

        query = self.q_head_norm(query)
        key = self.k_head_norm(raw_key)
        work = accumulation_dtype(raw_key.dtype)
        raw_work = raw_key.to(work)
        key_inv_rms = torch.rsqrt(
            raw_work.pow(2).mean(dim=-1) + self.k_head_norm.eps).to(raw_key.dtype)
        # [B, H, L, D] for attention.
        return (query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                latent, key_inv_rms)

    def project(self, x: torch.Tensor):
        """Expanded training projections, kept as the public audit surface."""
        query, key, value, _, _ = self.project_with_latent(x)
        return query, key, value

    @staticmethod
    def _mask(segment_ids: torch.Tensor, key_segments: torch.Tensor,
              length: int, total: int) -> torch.Tensor:
        offset = total - length
        positions = torch.arange(length, device=segment_ids.device) + offset
        causal = positions[:, None] >= torch.arange(total, device=segment_ids.device)[None, :]
        same_document = segment_ids[:, :, None] == key_segments[:, None, :]
        return (causal[None, :, :] & same_document).unsqueeze(1)

    def _expanded_attention(self, query, key, value, segment_ids, key_segments,
                            document_info, cache_is_empty: bool):
        # The common un-packed path can use the fused causal kernel without a
        # dense [L,L] mask.  Packed documents retain the exact same-document
        # mask; cache continuations need an offset causal mask.
        one_document = (document_info is None or document_info.document_start_mask.shape[1] <= 1
                        or not bool(document_info.document_start_mask[:, 1:].any()))
        if cache_is_empty and one_document:
            return F.scaled_dot_product_attention(
                query, key, value, is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim))
        mask = self._mask(segment_ids, key_segments, query.shape[2], key.shape[2])
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            scale=1.0 / math.sqrt(self.head_dim))

    def _latent_attention(self, query, latent, key_inv_rms, mask):
        """Decode attention with the key/value up-projections absorbed.

        QK RMSNorm is retained exactly: the key norm's static gain is absorbed
        into the query, while its dynamic inverse-RMS scalar travels with the
        cached latent.  Values are expanded only after the weighted latent sum.
        """
        heads = self.config.num_heads
        rank = self.config.kv_lora_rank
        weights = self.kv_up.weight.view(
            heads, self.head_dim + self.value_head_dim, rank).float()
        key_weight = weights[:, :self.head_dim]
        value_weight = weights[:, self.head_dim:]

        weighted_query = query.float() * self.k_head_norm.weight.float().view(
            1, 1, 1, self.head_dim)
        absorbed_query = torch.einsum("bhld,hdr->bhlr", weighted_query, key_weight)
        scores = torch.einsum("bhlr,bsr->bhls", absorbed_query, latent.float())
        scores = scores * key_inv_rms.float().permute(0, 2, 1).unsqueeze(2)
        scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        latent_output = torch.einsum("bhls,bsr->bhlr", probabilities, latent.float())
        output = torch.einsum("bhlr,hvr->bhlv", latent_output, value_weight)
        return output.to(query.dtype)

    def forward(self, x: torch.Tensor, document_info: DocumentInfo | None = None,
                cache: MLACache | None = None):
        batch, length, _ = x.shape
        query, key, value, latent, key_inv_rms = self.project_with_latent(x)
        segment_ids = (document_info.segment_ids if document_info is not None
                       else torch.zeros(batch, length, dtype=torch.int32, device=x.device))

        cache_is_empty = cache is None or cache.latent is None
        if cache is not None and cache.latent is not None:
            all_latent = torch.cat([cache.latent, latent], dim=1)
            all_inv_rms = torch.cat([cache.key_inv_rms, key_inv_rms], dim=1)
            key_segments = torch.cat([cache.segment_ids, segment_ids], dim=1)
        else:
            all_latent = latent
            all_inv_rms = key_inv_rms
            key_segments = segment_ids

        if cache is not None:
            cache.latent = all_latent.detach()
            cache.key_inv_rms = all_inv_rms.detach()
            cache.segment_ids = key_segments.detach()

        if cache is None or cache_is_empty:
            # Training and prefill keep the cheaper 64-wide expanded attention.
            output = self._expanded_attention(
                query, key, value, segment_ids, key_segments,
                document_info, cache_is_empty=True)
        else:
            mask = self._mask(segment_ids, key_segments, length, all_latent.shape[1])
            output = self._latent_attention(query, all_latent, all_inv_rms, mask)

        output = output.transpose(1, 2).reshape(batch, length, -1)
        gate = torch.sigmoid(self.output_gate(x))
        return self.o_proj(output * gate), cache


__all__ = ["GatedNoPEMLA", "MLACache", "attention_document_mask"]
