# Tiny LLM

A byte-level Transformer language model trained from scratch, in one file.

This is a teaching implementation. There is no tokenizer, no pretrained weights,
no framework — just a decoder-only Transformer over raw bytes, written so you can
read the whole thing top to bottom and see how the pieces fit.

It is **not** a chatbot. It learns to continue text in the style of whatever you
feed it. Give it a few kilobytes of writing on a subject and it will produce
more writing on that subject, with the grammar and vocabulary it picked up.

## Requirements

Python 3.10+ and PyTorch. That's it.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you do not have an NVIDIA GPU, install the CPU-only build of PyTorch. It is a
much smaller download — roughly 200 MB instead of 3.5 GB of CUDA libraries you
cannot use:

```sh
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Quick start

```sh
# Look at the corpus and the model size before committing to a run
.venv/bin/python tiny-llm.py info --data-dir data

# Train. Takes about 3 minutes on a laptop CPU.
.venv/bin/python tiny-llm.py train --data-dir data

# Sample from the result
.venv/bin/python tiny-llm.py generate --prompt "Astronomy Notes" --tokens 400
```

## Commands

### `info`

Reports corpus statistics and the parameter budget, without training anything.
Run this first — it tells you whether your model is the right size for your data.

### `train`

Trains the model and saves the best checkpoint. It prints a sample at the end so
you can see what you got without a second command.

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--data-dir` | `data` | Recursively reads `.md` and `.txt` files |
| `--steps` | 3000 | More steps is not always better — see below |
| `--block-size` | 128 | Context length in bytes |
| `--n-embd` | 64 | Embedding width |
| `--n-layer` | 2 | Number of Transformer blocks |
| `--learning-rate` | 1e-3 | Peaks here, then cosine-decays to 10% |
| `--patience` | 12 | Stop after this many evaluations with no improvement |
| `--save-optimizer` | off | Store AdamW state too, for resuming training |

### `generate`

Samples from a saved checkpoint.

```sh
.venv/bin/python tiny-llm.py generate \
  --checkpoint checkpoints/tiny-llm.pt \
  --prompt "Cooking Notes" \
  --tokens 500 \
  --temperature 0.8 \
  --top-k 40
```

Lower `--temperature` makes it more repetitive and safe. Higher makes it more
chaotic. `--top-k 0` disables the top-k filter and samples from the full
distribution, which is usually worse.

## The data

`data/` holds five short notes on unrelated subjects — computing, astronomy,
gardening, cooking, and history. Together they are about 17 KB. That is
deliberately small: small enough to train in minutes, and small enough that you
can watch the model's behaviour change as it learns.

Replace them with your own `.md` or `.txt` files. More text is better. A few
hundred kilobytes will teach it far more than 17 KB can.

## What to expect, and why

The interesting part of this project is not the architecture — it is watching a
small model deal with a small corpus.

**Validation loss bottoms out early, then rises.** With ~15 KB of training text
and a model with hundreds of thousands of parameters, there is far more model
than data. The model memorizes the training text and stops generalizing. Training
loss keeps falling toward zero while validation loss climbs. The script tracks
the best validation checkpoint and stops when it stops improving, so the saved
model is the good one — but the gap is the lesson.

Run `info` and look at "parameters per training byte". When that number is above
1, the model has more parameters than the corpus has bytes, and memorization is
the expected outcome. Roughly 20 bytes of training text per parameter is a
comfortable ratio; the sample corpus is a long way from that.

**Splitting matters.** Validation is split *within each document*, not by
cutting the concatenated corpus at one offset. If you cut the concatenation, whole
files land on one side of the cut and validation ends up measuring a different
subject than training. Splitting per document keeps the two comparable.

**Depth is not automatically better.** On this corpus, a two-block model reaches
a lower validation loss than a three-block one with the same width, and trains
faster:

| config | parameters | best val loss |
| --- | --- | --- |
| n_embd=32, n_layer=2 | 45,728 | 2.2966 |
| n_embd=48, n_layer=3 | 115,152 | 2.1194 |
| **n_embd=64, n_layer=2** | **140,608** | **2.0947** |
| n_embd=64, n_layer=3 | 190,400 | 2.1774 |

That is the small-corpus effect again: extra depth adds capacity to memorize
with, and there is nothing here to generalize to. The default is the two-block
model. Add depth when you add data, not before.

**Bytes, not tokens.** The vocabulary is 256 — every possible byte. That means no
tokenizer, no out-of-vocabulary problem, and any input works. It also means the
model spends capacity learning to spell, and its context window of 128 is only
about 128 characters. A real tokenizer would fit far more text into the same
window.

## Files

```
tiny-llm.py       the model, the training loop, and the CLI
data/             training text
checkpoints/      saved checkpoints (best validation loss)
```

Checkpoints store the model weights, the config needed to rebuild the model, and
the step and losses. Attention masks are not stored — they are derived from the
config and rebuilt on load, so they never bloat the file.

## Credits

Architecture follows the decoder-only Transformer from *Attention Is All You
Need* (Vaswani et al., 2017), with the pre-norm residual layout and scaled
initialization popularized by GPT-2 (Radford et al., 2019). Query/key
normalization and RMSNorm come from more recent practice; both help small models
train stably without a long warmup.
