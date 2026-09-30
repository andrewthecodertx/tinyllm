"""Tests for tiny-llm.py.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_module():
    """Load tiny-llm.py, whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location("tiny_llm", ROOT / "tiny-llm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tiny_llm"] = module
    spec.loader.exec_module(module)
    return module


tiny = load_module()


# --- config -----------------------------------------------------------------


def test_config_builds_from_partial_args():
    """`info` does not expose training flags; build_config must still work."""

    class Args:
        n_embd = 64
        n_head = 4
        n_layer = 3
        block_size = 128
        val_fraction = 0.1

    config = tiny.build_config(Args())
    assert config.n_embd == 64
    assert config.steps == tiny.Config().steps


def test_n_embd_must_divide_n_head():
    class Args:
        n_embd = 65
        n_head = 4
        n_layer = 3
        block_size = 128
        val_fraction = 0.1

    with pytest.raises(ValueError):
        tiny.build_config(Args())


# --- model ------------------------------------------------------------------


def test_forward_shapes_and_loss():
    config = tiny.Config(block_size=16, n_embd=32, n_layer=2)
    model = tiny.TinyLLM(config)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, torch.randint(0, 256, (2, 16)))
    assert logits.shape == (2, 16, 256)
    assert loss.item() > 0


def test_an_untrained_model_is_not_confident():
    """Loss at init should be near ln(256) ~= 5.545 for a uniform guess."""
    config = tiny.Config(block_size=16, n_embd=32, n_layer=2)
    torch.manual_seed(0)
    model = tiny.TinyLLM(config)
    x = torch.randint(0, 256, (4, 16))
    _, loss = model(x, torch.randint(0, 256, (4, 16)))
    assert 5.0 < loss.item() < 6.0


def test_causal_mask_hides_future_bytes():
    """Changing a later byte must not change an earlier prediction."""
    config = tiny.Config(block_size=8, n_embd=32, n_layer=1)
    torch.manual_seed(0)
    model = tiny.TinyLLM(config).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = torch.tensor([[1, 2, 3, 4, 200, 201, 202, 203]])
    with torch.no_grad():
        logits_first, _ = model(first)
        logits_second, _ = model(second)
    assert torch.allclose(logits_first[:, :4], logits_second[:, :4], atol=1e-5)


def test_context_longer_than_block_size_is_rejected():
    config = tiny.Config(block_size=8, n_embd=32, n_layer=1)
    model = tiny.TinyLLM(config)
    with pytest.raises(ValueError):
        model(torch.randint(0, 256, (1, 9)))


def test_generate_extends_the_sequence():
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    model = tiny.TinyLLM(config)
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=10)
    assert out.shape == (1, 13)


def test_generate_rejects_zero_temperature():
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    model = tiny.TinyLLM(config)
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1]]), max_new_tokens=1, temperature=0)


def test_generate_is_deterministic_for_a_given_seed():
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    torch.manual_seed(1)
    model = tiny.TinyLLM(config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt, 12, seed=7)
    second = model.generate(prompt, 12, seed=7)
    assert torch.equal(first, second)


def test_attention_masks_are_not_saved():
    """Masks are derived data; persisting them just bloats checkpoints."""
    config = tiny.Config(block_size=16, n_embd=32, n_layer=2)
    state = tiny.TinyLLM(config).state_dict()
    assert not [key for key in state if key.endswith(".mask")]


# --- data -------------------------------------------------------------------


def test_split_gives_every_document_to_both_sides():
    documents = [("a.md", "x" * 100), ("b.md", "y" * 100), ("c.md", "z" * 100)]
    train, val = tiny.split_documents(documents, 0.1)
    assert len(train) == 3 and len(val) == 3
    # every document contributes validation bytes, not just the last one
    assert all(len(part) == 10 for part in val)


def test_split_never_produces_an_empty_side():
    """A single-byte document would give an empty validation side; keep it whole."""
    train, val = tiny.split_documents([("a.md", "x")], 0.1)
    assert train == ["x"]
    assert val == []


def test_split_takes_the_cut_from_the_end():
    train, val = tiny.split_documents([("a.md", "abcdefghij")], 0.1)
    assert train == ["abcdefghi"]
    assert val == ["j"]


def test_encode_round_trips_through_bytes():
    parts = ["hello", "world"]
    data = tiny.encode(parts)
    assert data.dtype == torch.long
    assert bytes(data.tolist()).decode() == "hello\n\n<|document-boundary|>\n\nworld"


def test_corpus_stats_counts_files(tmp_path: Path):
    (tmp_path / "one.md").write_text("alpha beta gamma")
    (tmp_path / "two.md").write_text("delta")
    documents = tiny.load_documents(tmp_path)
    stats = tiny.corpus_stats(documents)
    assert stats["files"] == 2
    assert stats["words"] == 4


def test_load_documents_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        tiny.load_documents(tmp_path)


# --- training utilities -----------------------------------------------------


def test_learning_rate_warms_up_then_decays():
    config = tiny.Config(learning_rate=1e-3, warmup_steps=100, steps=1000)
    assert tiny.learning_rate_at(0, config) < config.learning_rate
    assert tiny.learning_rate_at(99, config) == pytest.approx(config.learning_rate)
    assert tiny.learning_rate_at(1000, config) < tiny.learning_rate_at(500, config)
    assert tiny.learning_rate_at(1500, config) >= 0.1 * config.learning_rate * 0.99


def test_get_batch_rejects_short_split():
    config = tiny.Config(block_size=128, batch_size=2)
    with pytest.raises(ValueError):
        tiny.get_batch(torch.zeros(50, dtype=torch.long), config, torch.device("cpu"))


def test_get_batch_targets_are_shifted_by_one():
    config = tiny.Config(block_size=8, batch_size=1)
    data = torch.arange(100, dtype=torch.long)
    x, y = tiny.get_batch(data, config, torch.device("cpu"))
    assert torch.equal(y[0, :-1], x[0, 1:])


def test_estimate_loss_is_stable_across_calls():
    """Fixed evaluation windows make val loss comparable between steps."""
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1, eval_batches=4)
    torch.manual_seed(0)
    model = tiny.TinyLLM(config)
    train_data = torch.randint(0, 256, (500,))
    val_data = torch.randint(0, 256, (500,))
    first = tiny.estimate_loss(model, train_data, val_data, config, torch.device("cpu"))
    second = tiny.estimate_loss(model, train_data, val_data, config, torch.device("cpu"))
    assert first == second


def test_parameter_count_matches_the_budget_report():
    config = tiny.Config(block_size=16, n_embd=32, n_layer=2)
    model = tiny.TinyLLM(config)
    lines = tiny.describe_budget(model, 1000)
    assert f"parameters: {tiny.parameter_count(model):,}" in lines


def test_budget_warns_when_model_outgrows_the_corpus():
    config = tiny.Config(block_size=16, n_embd=64, n_layer=3)
    lines = tiny.describe_budget(tiny.TinyLLM(config), 100)
    assert any("memorize" in line for line in lines)


# --- checkpoints ------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path):
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    torch.manual_seed(0)
    model = tiny.TinyLLM(config)
    path = tmp_path / "ckpt.pt"
    tiny.save_checkpoint(path, model, None, config, 5, 1.0, 1.1)

    loaded = tiny.load_model(path, torch.device("cpu"))
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_without_optimizer_state_is_smaller(tmp_path: Path):
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    model = tiny.TinyLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randint(0, 256, (2, 16)))[0].sum()
    loss.backward()
    optimizer.step()

    plain = tmp_path / "plain.pt"
    full = tmp_path / "full.pt"
    tiny.save_checkpoint(plain, model, None, config, 1, 1.0, 1.0)
    tiny.save_checkpoint(full, model, optimizer, config, 1, 1.0, 1.0)
    assert plain.stat().st_size < full.stat().st_size


def test_load_rejects_unknown_config_keys(tmp_path: Path):
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "config": {**config.__dict__, "not_a_real_option": 1},
            "model_state": tiny.TinyLLM(config).state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError):
        tiny.load_model(path, torch.device("cpu"))


def test_load_rejects_legacy_checkpoints_with_a_useful_message(tmp_path: Path):
    """Old checkpoints stored masks; the error should say so, not list keys."""
    config = tiny.Config(block_size=16, n_embd=32, n_layer=1)
    state = tiny.TinyLLM(config).state_dict()
    state["blocks.0.attn.heads.0.mask"] = torch.ones(16, 16, dtype=torch.bool)
    path = tmp_path / "legacy.pt"
    torch.save({"config": config.__dict__, "model_state": state}, path)

    with pytest.raises(ValueError, match="older version"):
        tiny.load_model(path, torch.device("cpu"))


def test_end_to_end_training_improves_loss(tmp_path: Path):
    """A few steps on repeated text should reduce the loss."""
    corpus = tmp_path / "data"
    corpus.mkdir()
    (corpus / "a.md").write_text("the quick brown fox jumps over the lazy dog. " * 40)

    config = tiny.Config(
        block_size=32, n_embd=32, n_layer=1, batch_size=8, eval_batches=4, warmup_steps=5
    )
    torch.manual_seed(0)
    documents = tiny.load_documents(corpus)
    train_parts, val_parts = tiny.split_documents(documents, 0.1)
    train_data, val_data = tiny.encode(train_parts), tiny.encode(val_parts)

    model = tiny.TinyLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = tiny.estimate_loss(model, train_data, val_data, config, torch.device("cpu"))

    for step in range(60):
        x, y = tiny.get_batch(train_data, config, torch.device("cpu"))
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    after = tiny.estimate_loss(model, train_data, val_data, config, torch.device("cpu"))
    assert after["train"] < before["train"]