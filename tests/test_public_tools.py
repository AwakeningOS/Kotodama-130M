from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_text import config_from_dict
from scripts.prepare_data import iter_documents


def test_iter_documents_reads_text_and_jsonl(tmp_path: Path):
    text = tmp_path / "docs.txt"
    text.write_text("一つ目\n\n二つ目\n", encoding="utf-8")
    jsonl = tmp_path / "docs.jsonl"
    jsonl.write_text(json.dumps({"text": "三つ目"}, ensure_ascii=False) + "\n")
    assert list(iter_documents([text, jsonl])) == ["一つ目", "二つ目", "三つ目"]


def test_checkpoint_config_round_trip():
    raw = {
        "architecture_id": "kotodama_stable_loop_130m_v2",
        "vocab_size": 49_152,
        "hidden_size": 768,
        "ffn_intermediate_size": 2_272,
        "context_length_train": 1_024,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": True,
        "dropout": 0.0,
        "ffn_type": "swiglu",
        "prelude_pattern": ["KDA", "MLA"],
        "recurrent_core_pattern": ["KDA", "KDA", "KDA", "MLA", "KDA", "KDA", "KDA", "MLA"],
        "coda_pattern": ["KDA", "KDA"],
        "kda": {
            "num_heads": 12, "num_value_heads": 12, "head_dim": 64,
            "value_head_dim": 64, "short_conv_kernel_size": 4,
            "decay_rank": 64, "safe_gate": True, "lower_bound": -5.0,
            "allow_negative_eigenvalues": False,
            "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
        },
        "mla": {
            "num_heads": 12, "q_lora_rank": 128, "kv_lora_rank": 128,
            "qk_nope_head_dim": 64, "qk_shared_head_dim": 0,
            "v_head_dim": 64, "use_rope": False, "qk_rmsnorm": True,
            "full_rank_output_gate": True,
        },
        "loop": {
            "train_min_depth": 2, "train_max_depth": 8,
            "inference_depth": 8, "depth_ramp_tokens": 1_000_000_000,
            "max_depth_probability": 1.0 / 16.0,
            "state_init_std": (2.0 / 5.0) ** 0.5,
            "injection_decay_target": (1.0 / 5.0) ** 0.5,
            "full_bptt": True, "checkpoint_per_iteration": True,
        },
    }
    assert config_from_dict(raw).num_unique_blocks == 12
