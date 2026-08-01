"""The one block shape used everywhere.

Pre-RMSNorm, two residuals, identical in the prelude, the recurrent core and
the coda -- only the mixer differs:

    m  = Mixer(RMSNorm_1(x))
    x' = x + m
    f  = SwiGLU(RMSNorm_2(x'))
    y  = x' + f

These residuals live *inside* the block.  The loop's own state update is a
separate thing, `h <- Core(J(h, e))`, with no residual around the Core.  Adding
one there would add the state twice, so the two levels are kept visibly apart:
nothing in this file knows the loop exists.
"""

from __future__ import annotations

import torch
from torch import nn

from kotodama.config import KotodamaConfig
from kotodama.kda import KDACache, KDALayer
from kotodama.layers import RMSNorm, SwiGLUMLP
from kotodama.mla import GatedNoPEMLA, MLACache
from kotodama.segments import DocumentInfo


class MixerCache:
    """Whichever cache the block's mixer needs, and only that one."""

    def __init__(self, kda: KDACache | None = None, mla: MLACache | None = None):
        self.kda = kda
        self.mla = mla

    def reset(self) -> None:
        if self.kda is not None:
            self.kda.reset()
        if self.mla is not None:
            self.mla.reset()

    def reset_rows(self, rows: torch.Tensor) -> None:
        if self.kda is not None:
            self.kda.reset_rows(rows)
        if self.mla is not None:
            self.mla.reset_rows(rows)


class DecoderBlock(nn.Module):
    def __init__(self, config: KotodamaConfig, kind: str, fast_kda: bool = True):
        super().__init__()
        assert kind in ("KDA", "MLA"), kind
        self.kind = kind
        self.mixer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mixer = (
            KDALayer(config.hidden_size, config.kda, fast=fast_kda) if kind == "KDA"
            else GatedNoPEMLA(config.hidden_size, config.mla)
        )
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLUMLP(config.hidden_size, config.ffn_intermediate_size)

    def forward(self, x: torch.Tensor, document_info: DocumentInfo | None = None,
                cache: MixerCache | None = None):
        mixer_cache = None
        if cache is not None:
            mixer_cache = cache.kda if self.kind == "KDA" else cache.mla

        residual = x
        mixed, new_cache = self.mixer(
            self.mixer_norm(x), document_info=document_info, cache=mixer_cache)
        x = residual + mixed

        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        if cache is not None:
            if self.kind == "KDA":
                cache.kda = new_cache
            else:
                cache.mla = new_cache
        return x, cache

    def new_cache(self) -> MixerCache:
        return MixerCache(kda=KDACache()) if self.kind == "KDA" else MixerCache(mla=MLACache())


__all__ = ["DecoderBlock", "MixerCache"]
