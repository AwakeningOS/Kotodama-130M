# Kotodama-130M Current Research Record — 2026-08-02

[日本語](RESEARCH.md) | **English**

This document covers only `kotodama_stable_loop_130m_v2`. Earlier model
definitions and experiments are not used as evidence for decisions about the
current architecture.

## Summary

The current model combines a 3:1 KDA/MLA hybrid, a stable diagonal recurrent
loop, SwiGLU, and a latent MLA cache with QK normalization. The depth schedule
ramps over one billion training tokens, while a fixed 1/16 tail keeps T=8 in
the training distribution from the beginning. This ensures that the T=8 path
has already received training by the first evaluation at 100 million tokens.

## Questions Answered by Existing Research

### Recurrent Structure and Stability

[Parcae](https://arxiv.org/abs/2604.12946) presents stable state injection
derived by discretizing a continuous system with a negative diagonal,
repeated injection of normalized input, and sequence-level variable-depth
training. Kotodama's `StableDiagonalInjection` and prelude normalization follow
this line of work.

[Scaling by Thinking in Continuous Space](https://arxiv.org/abs/2502.05171)
uses a prelude/core/coda structure, input injection at every iteration, random
initial states, and variable recurrent depth in large-scale training. The
like-init state used in Kotodama's labelled training is based on this
established design.

Parcae also reports that the average recurrent depth used during training
limits the useful range of inference-time scaling. Kotodama therefore exposes
the model to inference depth T=8 with a fixed probability from the beginning
of training.

### KDA/MLA Hybrid

[Kimi Linear](https://arxiv.org/abs/2510.26692) validates a 3:1 hybrid of KDA
and global MLA at scale. Kotodama is a small dense model, but its KDA/MLA ratio
and strict NoPE MLA design follow this line of work.

### Latent KV Cache with QK Normalization

[QK-Normed MLA](https://arxiv.org/abs/2606.16310) shows that latent decoding
can retain post-projection QK RMSNorm by absorbing the static key gain into
the query side and storing a dynamic inverse-RMS scalar with the latent state.
Kotodama's persistent cache stores a 128-element latent vector and 12
inverse-RMS values, for a total of 140 values per token.

### Why Kotodama Does Not Use DeepLoop

[DeepLoop](https://arxiv.org/abs/2607.13491) provides an initialization rule
for shared residual blocks that use Post-LN and DeepNorm. Kotodama instead uses
Pre-RMSNorm blocks with Parcae-style injection. Because the assumptions behind
the formulas differ, Kotodama does not transfer DeepLoop's coefficients
directly.

## Fixed Implementation Decisions

| Item | Current value |
|---|---|
| Architecture ID | `kotodama_stable_loop_130m_v2` |
| Depth ramp | 1B tokens |
| Early T=8 exposure | Always approximately 1/16 or more |
| Loop initial state | Like-init randomness only during labelled training |
| FFN | SwiGLU |
| MLA persistent cache | 128 latent values + 12 inverse-RMS values |
| EOD cache reset | Every layer, independently for each batch row |
| First stop | 100M tokens |
| Production optimizer | AdamW |

## Questions That the Literature Cannot Answer

The following questions are specific to this model and require training and
evaluation:

- Does a 130M model that combines KDA/MLA with a Parcae-style recurrent loop
  achieve better language performance than competing fixed-depth models?
- Does a 1/16 T=8 tail with a 1B-token ramp provide a useful quality-speed
  trade-off under this GPU budget?
- Do validation loss and practical benchmark results improve when inference
  depth increases from T=2 to T=8?
- After like-init training, does performance hold up under deterministic
  zero-init evaluation?

These model-specific diagnostics are evaluated at 100M and 500M tokens. The
formal three-model shared benchmark is run at 1B tokens using the same frozen
question set for Kotodama, Deltaxis, and KaiNomos. Multiple seeds and strict
FLOP matching are not required; the complete models compete on the same
practical benchmark.
