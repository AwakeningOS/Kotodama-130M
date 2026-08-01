"""Kotodama Stable Loop 130M v2.

    tokens -> embedding -> PRELUDE (2 blocks, own weights)
                        -> e = RMSNorm(prelude output), fixed for every iteration
                        -> h0 ~ TruncNormal(0, 2/5) during labelled training
                              h0 = 0 for deterministic evaluation/generation
                        -> T times:  h <- Core(J(h, e))     8 shared blocks
                        -> c = W_C h_T                      no residual
                        -> CODA (2 blocks, own weights)
                        -> RMSNorm -> logits = h E^T

129,964,332 parameters, and the same number at every loop depth.

The update rule, exactly
------------------------
    h_{l+1} = Core(J(h_l, e))

There is no residual around the Core.  Each block inside it already has two of
its own; wrapping another one outside would add the state a second time and
move the fixed point of the iteration.  The same applies to the core exit: `c`
is `W_C h_T`, not `h_T + W_C h_T`.

`e` is computed once and reused.  Not recomputed, not detached, not re-projected
per iteration -- gradients from every iteration must reach the prelude.

The core is time-invariant: `F(h, e)`, with no iteration index, no loop count
and no per-iteration parameters.  That is what allows running more iterations at
inference than were ever seen in training.

Loss goes on the final iteration only
-------------------------------------
"Supervising every iteration" means the gradient of the final loss flows
through all of them, which full BPTT already gives.  It does not mean attaching
a loss to each one: tying a readout to every intermediate depth pulls the
trajectory towards answering early at every step, which is the failure that
made a previous project's shared block finish all its work on the first visit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from kotodama.blocks import DecoderBlock
from kotodama.cache import ModelCache
from kotodama.config import EXACT_PARAMETER_COUNT, KotodamaConfig
from kotodama.kda import KDALayer, KDAProjections
from kotodama.layers import RMSNorm
from kotodama.loop_injection import StableDiagonalInjection
from kotodama.segments import DocumentInfo, build_document_info, build_labels


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    final_state: torch.Tensor | None = None
    # Diagnostic only, off by default.  The loop is required telemetry: a
    # stable injection bounds the carry-over operator, not the composed map
    # `Core(J(h, e))`, so divergence has to be watched rather than assumed away.
    loop_states: list[torch.Tensor] | None = None
    # `loss` is the mean over this micro-batch's valid positions; `loss_sum` and
    # `label_count` let the caller form an exact mean over a whole optimizer
    # step instead of averaging per-micro-batch means, which silently
    # over-weights micro-batches that happen to contain more EOD padding.
    loss_sum: torch.Tensor | None = None
    label_count: torch.Tensor | None = None


class KotodamaStableLoop(nn.Module):
    def __init__(self, config: KotodamaConfig, fast_kda: bool = True):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.prelude = nn.ModuleList(
            DecoderBlock(config, kind, fast_kda) for kind in config.prelude_pattern)
        self.prelude_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.loop_injection = StableDiagonalInjection(config.hidden_size)
        self.recurrent_core = nn.ModuleList(
            DecoderBlock(config, kind, fast_kda) for kind in config.recurrent_core_pattern)
        self.core_exit = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.coda = nn.ModuleList(
            DecoderBlock(config, kind, fast_kda) for kind in config.coda_pattern)
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.gradient_checkpointing = config.loop.checkpoint_per_iteration
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Generic first, deliberate last.

        A blanket `normal_` pass after construction would silently overwrite the
        KDA decay schedule, the identity convolutions, the identity core exit
        and the injection parameterisation -- exactly the initialisations that
        make the model start in a known state.
        """
        std = (2.0 / (5.0 * self.config.hidden_size)) ** 0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)

        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(1.0)
        for module in self.modules():
            if isinstance(module, (KDALayer, KDAProjections)):
                module.reset_kda_parameters()
        self.loop_injection.reset_parameters()
        self.core_exit.weight.copy_(torch.eye(self.config.hidden_size))

        # Every residual output starts at exactly zero, so each block is a
        # bitwise identity at step 0 rather than an approximate one.
        for block in [*self.prelude, *self.recurrent_core, *self.coda]:
            block.ffn.down_proj.weight.zero_()
            if block.kind == "KDA":
                block.mixer.projections.o_proj.weight.zero_()
            else:
                block.mixer.o_proj.weight.zero_()

        assert self.lm_head_weight is self.embed_tokens.weight

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Tied, always.  There is no separate LM head to fall out of sync."""
        return self.embed_tokens.weight

    def parameter_count(self) -> int:
        seen, total = set(), 0
        for parameter in self.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
        return total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def recurrent_step(self, h, e, document_info, caches=None):
        """`Core(J(h, e))` -- the injection, then the shared blocks."""
        x = self.loop_injection(h, e)
        for index, block in enumerate(self.recurrent_core):
            cache = caches[index] if caches is not None else None
            x, _ = block(x, document_info=document_info, cache=cache)
        return x

    def resolve_loop_depths(self, batch_size: int, supplied_depths, device):
        if supplied_depths is None:
            depth = self.config.loop.inference_depth
            return torch.full((batch_size,), depth, dtype=torch.int64, device=device)
        return supplied_depths.to(device=device, dtype=torch.int64)

    def initial_loop_state(self, e: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        """Like-init noise for training, a deterministic zero state otherwise.

        Recurrent-depth and Parcae models train from a random state to prevent
        the loop from depending on one privileged path.  Restricting the noise
        to labelled training forwards keeps validation, cache equivalence and
        greedy generation deterministic.  The state is an activation, not a
        parameter, so the model size stays exactly 129,964,332.
        """
        if self.training and labels is not None and self.config.loop.state_init_std > 0.0:
            state = torch.empty_like(e)
            std = self.config.loop.state_init_std
            return nn.init.trunc_normal_(state, std=std, a=-3.0 * std, b=3.0 * std)
        return torch.zeros_like(e)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loop_depths: torch.Tensor | None = None,
        document_info: DocumentInfo | None = None,
        cache: ModelCache | None = None,
        return_loop_states: bool = False,
    ) -> ModelOutput:
        if document_info is None:
            document_info = build_document_info(input_ids)

        if cache is not None and document_info.document_start_mask.shape[1] > 0:
            # A generation session may continue after EOD.  Clear every KDA
            # state, convolution history and MLA latent for only the affected
            # batch rows before the first token of the new document is read.
            cache.reset_rows(document_info.document_start_mask[:, 0])

        x = self.embed_tokens(input_ids)
        for index, block in enumerate(self.prelude):
            block_cache = cache.prelude_layers[index] if cache is not None else None
            x, _ = block(x, document_info=document_info, cache=block_cache)

        # Fixed for every iteration: not recomputed, not detached, not cloned.
        e = self.prelude_norm(x)
        h = self.initial_loop_state(e, labels)

        depths = self.resolve_loop_depths(input_ids.shape[0], loop_depths, input_ids.device)
        max_depth = int(depths.max())
        if cache is not None:
            cache.require_depth(max_depth)

        loop_states: list[torch.Tensor] = []
        for iteration in range(max_depth):
            caches = cache.loop_iterations[iteration].core_layers if cache is not None else None
            if self.gradient_checkpointing and self.training and cache is None:
                candidate = checkpoint(
                    self.recurrent_step, h, e, document_info,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                candidate = self.recurrent_step(h, e, document_info, caches)
            # Sequences that have finished keep their state bitwise unchanged.
            active = (iteration < depths).view(-1, 1, 1)
            h = torch.where(active, candidate, h)
            if return_loop_states:
                loop_states.append(h.detach())

        x = self.core_exit(h)
        for index, block in enumerate(self.coda):
            block_cache = cache.coda_layers[index] if cache is not None else None
            x, _ = block(x, document_info=document_info, cache=block_cache)

        hidden = self.final_norm(x)
        logits = F.linear(hidden, self.lm_head_weight)

        loss = loss_sum = label_count = None
        if labels is not None:
            flat = labels.reshape(-1)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), flat,
                ignore_index=-100, reduction="sum")
            label_count = (flat != -100).sum()
            loss = loss_sum / label_count.clamp_min(1)
        return ModelOutput(logits=logits, loss=loss, final_state=h,
                           loop_states=loop_states if return_loop_states else None,
                           loss_sum=loss_sum, label_count=label_count)

    def loss_from_ids(self, input_ids: torch.Tensor, loop_depths=None) -> ModelOutput:
        """Next-token loss with boundary positions excluded."""
        document_info = build_document_info(input_ids)
        labels = build_labels(input_ids, document_info.ntp_loss_mask)
        return self.forward(input_ids, labels=labels, loop_depths=loop_depths,
                            document_info=document_info)


def build_model(config: KotodamaConfig | None = None, fast_kda: bool = True):
    model = KotodamaStableLoop(config or KotodamaConfig(), fast_kda=fast_kda)
    actual = model.parameter_count()
    assert actual == EXACT_PARAMETER_COUNT, (actual, EXACT_PARAMETER_COUNT)
    return model


__all__ = ["KotodamaStableLoop", "ModelOutput", "build_model"]
