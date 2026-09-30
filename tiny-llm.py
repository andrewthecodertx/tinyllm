#!/usr/bin/env python3
"""Tiny byte-level Transformer language model trained from scratch.

Reads a directory of .md/.txt files and trains a character-level (byte-level)
decoder-only Transformer. The whole thing -- model, training loop, evaluation --
lives in this one file so it can be read end to end.

Examples:
  python tiny-llm.py info --data-dir data
  python tiny-llm.py train --data-dir data
  python tiny-llm.py generate --checkpoint checkpoints/tiny-llm.pt --tokens 400
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256
DEFAULT_CHECKPOINT = Path("checkpoints/tiny-llm.pt")


@dataclass
class Config:
    block_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    # Two blocks beat three on the bundled corpus, which is small enough that
    # extra depth mostly adds capacity to memorize with. Raise this when you
    # train on substantially more text.
    n_layer: int = 2
    dropout: float = 0.10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    steps: int = 3000
    eval_interval: int = 100
    eval_batches: int = 50
    patience: int = 12
    seed: int = 1337


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


class Head(nn.Module):
    """One self-attention head with a causal mask."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Scaling queries and keys before the dot product keeps the attention
        # logits in a sane range, which matters a lot when training without a
        # long warmup.
        self.q_norm = RMSNorm(head_size)
        self.k_norm = RMSNorm(head_size)
        # persistent=False keeps the mask out of the state_dict. It is derived
        # data, not a learned parameter, so saving it just bloats checkpoints.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        key = self.k_norm(self.key(x))
        query = self.q_norm(self.query(x))
        weights = query @ key.transpose(-2, -1)
        weights = weights.masked_fill(~self.mask[:time, :time], float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadAttention(
            config.n_embd, config.n_head, config.block_size, config.dropout
        )
        self.ln2 = RMSNorm(config.n_embd)
        self.ffwd = FeedForward(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class TinyLLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, VOCAB_SIZE, bias=False)
        self.apply(self._init_weights)
        # GPT-2 style scaled init on the residual output projections. Without
        # it the residual stream grows with depth and the first steps are noisy.
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(f"context length {time} exceeds block size {self.config.block_size}")
        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, VOCAB_SIZE), targets.reshape(batch * time)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(seed)
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Read every .md/.txt file as its own document. Returns (name, text)."""
    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not paths:
        raise FileNotFoundError(f"No .txt or .md files found under {data_dir}")
    documents = []
    for path in paths:
        try:
            documents.append((path.name, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not valid UTF-8") from error
    return documents


def split_documents(
    documents: list[tuple[str, str]], val_fraction: float
) -> tuple[list[str], list[str]]:
    """Split *within each document* so every file contributes to both sides.

    Splitting the concatenated corpus at one byte offset would leave whole
    files stranded on one side of the cut, which means validation measures a
    different subject than training. Splitting per document keeps the two
    splits comparable.
    """
    train_parts, val_parts = [], []
    for _, text in documents:
        cut = int(len(text) * (1.0 - val_fraction))
        if cut <= 0 or cut >= len(text):
            train_parts.append(text)
            continue
        train_parts.append(text[:cut])
        val_parts.append(text[cut:])
    return train_parts, val_parts


def encode(parts: list[str]) -> torch.Tensor:
    joined = "\n\n<|document-boundary|>\n\n".join(parts)
    return torch.tensor(list(joined.encode("utf-8")), dtype=torch.long)


def corpus_stats(documents: list[tuple[str, str]]) -> dict:
    text = "\n".join(name for name, _ in documents)
    words = " ".join(body for _, body in documents).split()
    return {
        "files": len(documents),
        "bytes": sum(len(body.encode("utf-8")) for _, body in documents),
        "words": len(words),
        "unique_words": len(set(word.lower() for word in words)),
        "bytes_per_file": {name: len(body.encode("utf-8")) for name, body in documents},
    }


# ---------------------------------------------------------------------------
# Model sizing
# ---------------------------------------------------------------------------

def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def describe_budget(model: nn.Module, train_bytes: int) -> list[str]:
    """Return human-readable lines describing the parameter/data budget."""
    total = parameter_count(model)
    per_byte = total / max(1, train_bytes)
    # Rule of thumb from the scaling literature: roughly 20 tokens of training
    # data per parameter before returns flatten out.
    suggested = max(1, train_bytes // 20)
    lines = [f"parameters: {total:,}"]
    lines.append(f"training bytes: {train_bytes:,}")
    lines.append(f"parameters per training byte: {per_byte:.1f}")
    if per_byte > 1.0:
        lines.append(
            f"note: this is a lot of model for the corpus. A corpus this size "
            f"would suggest roughly {suggested:,} parameters. The model will "
            f"memorize rather than generalize, so watch validation loss and "
            f"expect the best checkpoint early -- not at the final step."
        )
    return lines


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor, config: Config, device: torch.device, name: str = "split"
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= config.block_size + 1:
        raise ValueError(
            f"{name} has {len(data)} bytes but needs more than block_size + 1 "
            f"({config.block_size + 1}). Lower --block-size or raise --val-fraction."
        )
    starts = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyLLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a fixed, evenly spaced set of windows.

    Using the same windows every time makes val loss comparable between steps,
    which is what makes "keep the best checkpoint" meaningful. It also runs in
    one forward pass per split instead of eval_batches separate passes.
    """
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("val", val_data)):
        usable = len(data) - config.block_size - 1
        count = max(1, min(config.eval_batches, usable))
        starts = torch.linspace(0, usable - 1, count).long()
        x = torch.stack([data[i : i + config.block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in starts]).to(device)
        _, loss = model(x, y)
        results[name] = loss.item()
    model.train()
    return results


def learning_rate_at(step: int, config: Config) -> float:
    """Linear warmup, then cosine decay to 10% of the peak rate."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = max(1, config.steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (0.1 + 0.9 * cosine)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return device


def save_checkpoint(
    path: Path,
    model: TinyLLM,
    optimizer: Optional[torch.optim.Optimizer],
    config: Config,
    step: int,
    train_loss: float,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "model_state": model.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    # Optimizer state is only useful for resuming training, and it is typically
    # larger than the model itself. Skip it for inference checkpoints.
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model(checkpoint_path: Path, device: torch.device) -> TinyLLM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint["config"]
    known = {field.name for field in fields(Config)}
    unknown = set(saved) - known
    if unknown:
        raise ValueError(f"checkpoint has unknown config keys: {sorted(unknown)}")
    config = Config(**saved)
    model = TinyLLM(config).to(device)
    state = checkpoint["model_state"]
    # Checkpoints from earlier versions stored attention masks and LayerNorm
    # biases, and had no query/key normalization. Those weights cannot be
    # reused, so say so plainly instead of dumping a few hundred key names.
    if any(key.endswith(".mask") for key in state):
        raise ValueError(
            f"{checkpoint_path} was written by an older version of this file and "
            f"is not compatible with the current model. Attention masks are no "
            f"longer stored, and the attention blocks now normalize queries and "
            f"keys. Retrain to produce a checkpoint in the current format:\n"
            f"  python tiny-llm.py train --data-dir data "
            f"--checkpoint {checkpoint_path}"
        )
    model.load_state_dict(state)
    model.eval()
    return model


def sample_text(
    model: TinyLLM,
    prompt: str,
    tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
    seed: Optional[int] = None,
) -> str:
    raw = prompt.encode("utf-8") or b"\n"
    context = torch.tensor([list(raw)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(
            context,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
    return bytes(output[0].tolist()).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    """Build a Config from parsed args, falling back to Config defaults.

    `info` does not expose the training hyperparameters, so every value is
    read with a default rather than requiring the attribute to exist.
    """
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    defaults = Config()
    values = {}
    for field in fields(Config):
        values[field.name] = getattr(args, field.name, getattr(defaults, field.name))
    return Config(**values)


def command_info(args: argparse.Namespace) -> None:
    documents = load_documents(Path(args.data_dir))
    stats = corpus_stats(documents)
    print(f"data directory: {args.data_dir}")
    print(f"files: {stats['files']}")
    print(f"bytes: {stats['bytes']:,}")
    print(f"words: {stats['words']:,} ({stats['unique_words']:,} unique)")
    if stats["words"]:
        unique_ratio = stats["unique_words"] / stats["words"]
        print(f"type/token ratio: {unique_ratio:.3f}")
    print("per file:")
    for name, size in stats["bytes_per_file"].items():
        print(f"  {name:<28} {size:>8,} bytes")
    config = build_config(args)
    train_parts, val_parts = split_documents(documents, args.val_fraction)
    train_data, val_data = encode(train_parts), encode(val_parts)
    print(f"train bytes: {len(train_data):,}")
    print(f"val bytes:   {len(val_data):,}")
    model = TinyLLM(config)
    for line in describe_budget(model, len(train_data)):
        print(line)
    print(f"context length: {config.block_size} bytes")
    print(f"steps: {config.steps:,} at batch {config.batch_size}")


def command_train(args: argparse.Namespace) -> None:
    config = build_config(args)
    device = choose_device(args.device)
    documents = load_documents(Path(args.data_dir))
    train_parts, val_parts = split_documents(documents, args.val_fraction)
    train_data, val_data = encode(train_parts), encode(val_parts)
    if len(val_data) <= config.block_size + 1:
        raise ValueError(
            f"validation split is only {len(val_data):,} bytes, which is too small "
            f"for --block-size {config.block_size}. Raise --val-fraction or lower "
            f"--block-size."
        )

    # Seed before constructing the model so the run is reproducible end to end.
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = TinyLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    print(f"device: {device}")
    print(f"files: {len(documents)}")
    print(f"corpus: {len(train_data) + len(val_data):,} bytes "
          f"({len(train_data):,} train, {len(val_data):,} validation)")
    for line in describe_budget(model, len(train_data)):
        print(line)

    checkpoint_path = Path(args.checkpoint)
    best_val_loss = float("inf")
    best_step = -1
    evals_without_improvement = 0
    started = time.time()

    for step in range(config.steps + 1):
        if step % config.eval_interval == 0 or step == config.steps:
            losses = estimate_loss(model, train_data, val_data, config, device)
            improved = losses["val"] < best_val_loss
            marker = " *" if improved else ""
            print(
                f"step {step:>6} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}{marker}"
            )
            if improved:
                best_val_loss = losses["val"]
                best_step = step
                evals_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer if args.save_optimizer else None,
                    config,
                    step,
                    losses["train"],
                    losses["val"],
                )
            else:
                evals_without_improvement += 1
                if config.patience and evals_without_improvement >= config.patience:
                    print(
                        f"validation has not improved in {config.patience} evaluations; "
                        f"stopping at step {step}"
                    )
                    break

        if step == config.steps:
            break

        current_lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        x, y = get_batch(train_data, config, device, "train split")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    elapsed = time.time() - started
    print(f"finished in {elapsed / 60:.1f} min; best validation loss "
          f"{best_val_loss:.4f} at step {best_step}")
    if best_step < config.steps // 2:
        print(
            "note: the best checkpoint arrived early and validation then got worse. "
            "That is the model memorizing the training split. The saved checkpoint "
            "is the good one; more steps would not help without more data."
        )
    print(f"checkpoint: {checkpoint_path} "
          f"({checkpoint_path.stat().st_size / 1_000_000:.2f} MB)")

    # Show what the trained model actually produces.
    best_model = load_model(checkpoint_path, device)
    prompt = args.prompt
    if prompt is None:
        prompt = documents[0][1].splitlines()[0][:40] + "\n"
    print(f"\n--- sample (prompt: {prompt!r}) ---")
    print(sample_text(best_model, prompt, args.tokens, args.temperature, args.top_k, device))


def command_generate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    print(
        sample_text(
            model,
            args.prompt,
            args.tokens,
            args.temperature,
            args.top_k,
            device,
            seed=args.seed,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample a tiny byte-level Transformer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--block-size", type=int, default=128)
        target.add_argument("--n-embd", type=int, default=64)
        target.add_argument("--n-head", type=int, default=4)
        target.add_argument("--n-layer", type=int, default=2)
        target.add_argument("--val-fraction", type=float, default=0.10)

    info = subparsers.add_parser("info", help="Report corpus and model size before training.")
    info.add_argument("--data-dir", default="data")
    add_model_args(info)
    info.set_defaults(func=command_info)

    train = subparsers.add_parser("train", help="Train a model from .txt and .md files.")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    train.add_argument("--device", default="auto")
    add_model_args(train)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--steps", type=int, default=3000)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--eval-batches", type=int, default=50,
                       help="Number of evaluation windows per split.")
    train.add_argument("--patience", type=int, default=12,
                       help="Stop after this many evaluations without improvement (0 disables).")
    train.add_argument("--save-optimizer", action="store_true",
                       help="Also store optimizer state, for resuming training.")
    train.add_argument("--prompt", default=None,
                       help="Prompt for the sample printed after training.")
    train.add_argument("--tokens", type=int, default=300)
    train.add_argument("--temperature", type=float, default=0.8)
    train.add_argument("--top-k", type=int, default=40)
    train.add_argument("--seed", type=int, default=1337)
    train.set_defaults(func=command_train)

    generate = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    generate.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    generate.add_argument("--device", default="auto")
    generate.add_argument("--prompt", default="\n")
    generate.add_argument("--tokens", type=int, default=400)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40, help="Set to 0 to disable top-k sampling.")
    generate.add_argument("--seed", type=int, default=None)
    generate.set_defaults(func=command_generate)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "generate" and args.top_k <= 0:
        args.top_k = None
    if args.command == "train" and args.top_k <= 0:
        args.top_k = None
    if args.command in {"train", "info"} and not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be between 0 and 0.5")
    args.func(args)


if __name__ == "__main__":
    main()