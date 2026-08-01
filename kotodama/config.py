"""Frozen configuration for kotodama_stable_loop_130m_v2.

The invariants in `KotodamaConfig.__post_init__` are not defensive programming.
Every one of them is a contract from the implementation specification, and a
violation means the model being built is not the model that was specified --
so it fails at construction rather than 400,000 steps later.

Parameter budget, verified by arithmetic against the specification:

    tied token embedding              37,748,736
    9 KDA mixers        2,575,948 x 9 23,183,532
    3 Gated MLA mixers  1,671,552 x 3  5,014,656
    12 SwiGLU FFNs      5,234,688 x 12 62,816,256
    24 block RMSNorms                      18,432
    stable loop injection                 591,360
    core exit projection                  589,824
    prelude + final RMSNorm                 1,536
                                     ------------
                                      129,964,332
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ARCHITECTURE_ID = "kotodama_stable_loop_130m_v2"
EXACT_PARAMETER_COUNT = 129_964_332
EOD_TOKEN_ID = 4

PRELUDE_PATTERN = ("KDA", "MLA")
RECURRENT_CORE_PATTERN = ("KDA", "KDA", "KDA", "MLA", "KDA", "KDA", "KDA", "MLA")
CODA_PATTERN = ("KDA", "KDA")


@dataclass(frozen=True)
class KDAConfig:
    num_heads: int = 12
    num_value_heads: int = 12
    head_dim: int = 64
    value_head_dim: int = 64
    short_conv_kernel_size: int = 4
    decay_rank: int = 64
    safe_gate: bool = True
    lower_bound: float = -5.0
    allow_negative_eigenvalues: bool = False
    use_qk_l2norm_in_kernel: bool = True
    use_gate_in_kernel: bool = True
    use_beta_sigmoid_in_kernel: bool = True

    @property
    def scale(self) -> float:
        """1 / sqrt(head_dim).  Specification fixes this at 0.125."""
        return 1.0 / math.sqrt(self.head_dim)


@dataclass(frozen=True)
class MLAConfig:
    num_heads: int = 12
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    # Strict NoPE: there is no shared/RoPE key dimension at all.
    qk_shared_head_dim: int = 0
    v_head_dim: int = 64
    use_rope: bool = False
    qk_rmsnorm: bool = True
    full_rank_output_gate: bool = True


@dataclass(frozen=True)
class LoopConfig:
    train_min_depth: int = 2
    train_max_depth: int = 8
    inference_depth: int = 8
    # Reach the full depth distribution by 1B tokens instead of 4B.  A small
    # deterministic T=8 tail is present from step zero so the recurrent core
    # cannot spend the entire 100M-token gate learning only a shallow solution.
    depth_ramp_tokens: int = 1_000_000_000
    max_depth_probability: float = 1.0 / 16.0
    # Like-init state noise from recurrent-depth models.  It is used only for
    # labelled training forwards; evaluation and generation start from zero so
    # benchmark and greedy decode remain deterministic.
    state_init_std: float = math.sqrt(2.0 / 5.0)
    injection_decay_target: float = math.sqrt(1.0 / 5.0)
    full_bptt: bool = True
    checkpoint_per_iteration: bool = True


@dataclass(frozen=True)
class KotodamaConfig:
    architecture_id: str = ARCHITECTURE_ID
    vocab_size: int = 49_152
    hidden_size: int = 768
    ffn_intermediate_size: int = 2_272
    context_length_train: int = 1_024
    rms_norm_eps: float = 1.0e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    ffn_type: str = "swiglu"

    prelude_pattern: tuple[str, ...] = PRELUDE_PATTERN
    recurrent_core_pattern: tuple[str, ...] = RECURRENT_CORE_PATTERN
    coda_pattern: tuple[str, ...] = CODA_PATTERN

    kda: KDAConfig = field(default_factory=KDAConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def __post_init__(self) -> None:
        assert self.architecture_id == ARCHITECTURE_ID
        assert self.vocab_size == 49_152
        assert self.hidden_size == 768
        assert self.ffn_intermediate_size == 2_272
        assert self.tie_word_embeddings is True
        assert self.dropout == 0.0
        assert self.ffn_type == "swiglu"

        assert self.prelude_pattern == PRELUDE_PATTERN
        assert self.recurrent_core_pattern == RECURRENT_CORE_PATTERN
        assert self.coda_pattern == CODA_PATTERN

        assert self.kda.num_heads == 12
        assert self.kda.head_dim == 64
        assert self.kda.safe_gate is True
        assert self.kda.lower_bound == -5.0

        assert self.mla.num_heads == 12
        # A non-zero shared dimension would reintroduce a positional key path.
        assert self.mla.qk_shared_head_dim == 0
        assert self.mla.use_rope is False

        assert 2 <= self.loop.train_min_depth <= self.loop.train_max_depth <= 8
        assert 0.0 <= self.loop.max_depth_probability < 1.0
        assert self.loop.depth_ramp_tokens > 0
        assert self.loop.state_init_std >= 0.0

    @property
    def kinds(self) -> tuple[str, ...]:
        """Every unique block, in construction order.  Twelve of them."""
        return self.prelude_pattern + self.recurrent_core_pattern + self.coda_pattern

    @property
    def num_unique_blocks(self) -> int:
        return len(self.kinds)


__all__ = [
    "ARCHITECTURE_ID",
    "CODA_PATTERN",
    "EOD_TOKEN_ID",
    "EXACT_PARAMETER_COUNT",
    "PRELUDE_PATTERN",
    "RECURRENT_CORE_PATTERN",
    "KDAConfig",
    "KotodamaConfig",
    "LoopConfig",
    "MLAConfig",
]
