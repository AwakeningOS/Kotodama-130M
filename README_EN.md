# Kotodama-130M

[日本語](README.md) | **English**

Kotodama-130M is an experimental language model with roughly 130 million
parameters. Its defining idea is simple: instead of using a different set of
layers at every depth, it runs the same compact reasoning core several times.

This repository does not include pretrained weights, a tokenizer, or training
data. It contains the model, training and exact-resume code, data preparation
tools, and a text generation utility. Bring text that you have the right to
use, train your own tokenizer, and grow a small language model from scratch.

> Kotodama is currently a base model, not a chat-tuned assistant. Text
> completion is a better first experiment than instruction following.

## How it works

A conventional language model usually passes a sequence through a stack of
different layers once. Kotodama places a reusable core between a small input
stage and a small output stage.

```text
input text
  ↓
initial reading stage (KDA → MLA)
  ↓
shared recurrent core (KDA×3 → MLA → KDA×3 → MLA), repeated T times
  ↓
output stage (KDA → KDA)
  ↓
next-token probabilities
```

One way to picture this is to imagine reading a passage once, then revisiting
the same internal notes T times and refining them before writing the next
token. The core weights are shared, so increasing T does not make the model
file larger. At inference time, T acts as a compute knob that trades speed for
additional processing.

### KDA and MLA, without the jargon

- **KDA** gradually writes what it reads into a fixed-size running state. The
  state does not grow without bound as the context gets longer. It is useful
  for local flow, recent changes, and values that must be updated over time.
- **MLA** keeps a compressed record of earlier tokens and can look back across
  the sequence. It complements the information that a fixed-size recurrent
  state may struggle to preserve.

Kotodama uses three KDA blocks for every MLA block. It does not use RoPE;
ordering is learned through causal processing and the model's internal state.

### Keeping repeated computation stable

Repeatedly applying one network can cause activations to grow too large or
make the model forget its original input. Kotodama therefore injects a stable
amount of the initially processed input on every iteration. During training,
each sequence receives a depth from T=2 through T=8. A small number of T=8
sequences are present from the beginning so that the model cannot learn only a
shallow path.

## Requirements

- Linux
- Python 3.12
- A CUDA-capable NVIDIA GPU; a 24 GB card is recommended
- UTF-8 training text that you have permission to use
- A 49,152-piece SentencePiece tokenizer that you train yourself

Model construction, part of the test suite, and slow generation can run on a
CPU. Practical pretraining is intended for CUDA GPUs.

## Installation

```bash
git clone https://github.com/AwakeningOS/Kotodama-130M.git
cd Kotodama-130M
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
python -m pip install -r requirements.txt
```

`flash-linear-attention` must support your CUDA and PyTorch versions. If you
port the code to a different environment, run the CPU suite and a short GPU
acceptance test before committing to a long training run.

## 1. Train a tokenizer

Prepare UTF-8 text with one document per line. You may pass more than one input
file.

```bash
python scripts/train_tokenizer.py \
  --input corpus/train.txt \
  --model-prefix tokenizer/kotodama
```

The architecture fixes the vocabulary at 49,152 pieces. The special token IDs
are `unk=0`, `bos=1`, `eos=2`, `pad=3`, and `<|eod|>=4`. A very small corpus
usually cannot support a 49,152-piece vocabulary, so tokenizer training needs a
reasonably large and varied text collection.

You may use an existing tokenizer only if it has exactly 49,152 pieces and
places `<|eod|>` at ID 4.

## 2. Pack your data

Keep training and validation documents separate. For `.txt` files, every
non-empty line is one document. For `.jsonl` files, the `text` field in each
record is one document.

```bash
python scripts/prepare_data.py \
  --train corpus/train.txt \
  --validation corpus/validation.txt \
  --tokenizer-model tokenizer/kotodama.model \
  --output-dir data/my-corpus
```

Do not place benchmark questions, answer keys, private personal information,
or text without suitable permission in the training corpus.

## 3. Start with one optimizer step

One optimizer step contains 65,536 tokens. The default micro-batch size of 8
was selected for a 24 GB RTX 3090.

```bash
python train.py \
  --data-dir data/my-corpus \
  --run-dir runs/my-first-kotodama \
  --allow-gpu \
  --target-tokens 65536 \
  --max-steps 1
```

If the run exceeds your VRAM budget, try `--micro-batch 4`, `2`, or `1`. The
total number of tokens in an optimizer step stays the same; a smaller
micro-batch simply uses more accumulation passes. The first step also includes
compilation and is slower than steady-state training.

## 4. Train in bounded sessions

```bash
python train.py \
  --data-dir data/my-corpus \
  --run-dir runs/my-first-kotodama \
  --allow-gpu \
  --resume \
  --target-tokens 100000000 \
  --max-minutes 30
```

When the time budget expires, or the process receives Ctrl+C or SIGTERM, it
finishes the current optimizer step and writes a checkpoint. The newest two
regular checkpoints are retained. Running the same command with `--resume`
restores the model, optimizer, random number generators, and exact data
position.

## 5. Generate a continuation

```bash
python scripts/generate_text.py \
  --checkpoint runs/my-first-kotodama/step_0000001.pt \
  --tokenizer-model tokenizer/kotodama.model \
  --prompt "Long ago, beside the sea" \
  --depth 2 \
  --max-new-tokens 128
```

Try `--depth 1`, `2`, `4`, or `8` to change the amount of computation while
keeping the same weights. More depth is not guaranteed to improve the result.
Depth remains fixed during one generation session because changing it midway
would make the per-iteration cache histories inconsistent.

## Performance techniques

- Only the shared recurrent step is compiled, avoiding whole-model
  recompilation when T changes.
- Sequences with the same T are grouped together, reducing work on sequences
  that have already completed their assigned depth.
- Shallow depths skip activation checkpointing, while deeper paths recompute
  activations to stay within the VRAM budget.
- KDA uses a fused kernel, and MLA stores a compressed cache.
- Generation carries KDA state and MLA cache forward instead of recomputing the
  entire prefix for every token.
- The ordinary path avoids constructing a large document mask when the input
  contains no document boundary.

In one internal RTX 3090 measurement, compiling only the recurrent core made a
depth-4 training path about 62.8% faster. With a 1,024-token prompt and 256 new
tokens, a training-in-progress checkpoint generated 34.07 tokens/s at T2 and
10.25 tokens/s at T8. These figures describe one particular CUDA, PyTorch, FLA,
and checkpoint combination; they are not guaranteed performance.

## Approximate training time

Early BF16 training on one RTX 3090 reached roughly 13,000 tokens/s. A simple
division gives about 2.1 hours for 100M tokens, 10.7 hours for 500M, 21 hours
for 1B, and 14 days for 16B. Real training slows as the average T increases and
also includes compilation, validation, checkpoint I/O, and storage overhead.

The following ranges are more practical planning estimates:

| Training tokens | Approximate time on one RTX 3090 |
|---:|---:|
| 1 step (65,536 tokens) | Within a few minutes, including first compilation |
| 100M | About 2–3 hours |
| 500M | About 12–18 hours |
| 1B | About 1–2 days |
| 16B | About 2–4 weeks |

Your data, GPU, cooling, micro-batch size, CUDA stack, and storage can change
these numbers substantially. A sensible progression is one step, then a
30-minute session, then 100M tokens, checking checkpoints and generations at
each stage.

## Repository layout

```text
kotodama/                 model, caches, generation, and corpus loading
train.py                  AdamW pretraining with exact resume
recommended_inference.py  current reference depth preset (T2)
scripts/train_tokenizer.py
scripts/prepare_data.py
scripts/generate_text.py
tests/                    CPU correctness tests
RESEARCH.md               design sources and unresolved questions
```

## Not included

- Pretrained weights or optimizer checkpoints
- A tokenizer
- Training or validation data
- Raw internal benchmark logs
- Private experiments or operational files

A single checkpoint is roughly 780 MB, so do not commit checkpoints to regular
Git history. The repository's `.gitignore` excludes them.

## Disclaimer

Kotodama is an experimental research and learning implementation. Its quality,
safety, and factual reliability are not guaranteed. The 129,964,332-parameter
implementation passed CPU/GPU correctness checks and short training runs before
publication, but this repository does not distribute trained weights. Users
are responsible for checking the rights, privacy implications, and appropriate
use of their data and generated output.

MIT License
