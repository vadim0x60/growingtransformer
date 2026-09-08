import importlib
import json
import math

import numpy as np
import pytest
import torch

from growing_transformer.data import SPLITS, batch, load_split
from growing_transformer.model import GrowingTransformer, ModelConfig
from growing_transformer.train import (
    TrainConfig, evaluate, language_loss, restore_checkpoint, save_checkpoint,
)


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def small(**kwargs):
    return ModelConfig(d_model=16, head_dim=4, context=8, ff_mult=2, **kwargs)


def update(model, optimizer):
    x = torch.randint(27, (2, 8))
    y = torch.randint(27, (2, 8))
    optimizer.zero_grad(set_to_none=True)
    loss = language_loss(model(x), y) + 0.01 * model.provisional_penalty()
    loss.backward()
    optimizer.step()
    model.observe_usage()
    return loss.detach()


def activate(gate):
    gate.ema.fill_(0.8)
    gate.age.fill_(10)


def test_initial_frontier_and_penalty_gradient():
    model = GrowingTransformer(small())
    assert model.architecture() == [2, 2]
    assert [g.provisional.item() for g in model.gates()] == [False, False, True, True, False, True]
    penalty = model.provisional_penalty()
    assert penalty.item() == pytest.approx(0.3)
    penalty.backward()
    for gate in model.gates():
        if gate.provisional:
            assert gate.logit.grad > 0  # Gradient descent discourages usage.
        else:
            assert gate.logit.grad is None


def test_causal_and_provisional_parts_learn_from_main_loss():
    model = GrowingTransformer(small())
    x = torch.randint(27, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 27
    torch.testing.assert_close(model(x)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
    language_loss(model(x), (x + 1) % 27).backward()
    for gate in model.gates():
        assert gate.logit.grad is not None
        assert torch.isfinite(gate.logit.grad) and gate.logit.grad != 0
    for layer in model.layers:
        assert layer.heads[-1].qkv.weight.grad.abs().sum() > 0
    assert model.layers[-1].ff[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("which,expected", [("head", [3, 2]), ("layer", [2, 2, 2])])
def test_independent_growth(which, expected):
    model = GrowingTransformer(small())
    optimizer = torch.optim.AdamW(model.parameters())
    gate = model.layers[0].heads[-1].gate if which == "head" else model.layers[-1].gate
    activate(gate)
    events = model.grow(optimizer, threshold=0.5, warmup=10)
    assert len(events) == 1 and events[0]["kind"] == which
    assert model.architecture() == expected
    assert not gate.provisional
    assert model.layers[-1].gate.provisional
    for layer in model.layers:
        assert sum(h.gate.provisional.item() for h in layer.heads) == 1
        assert layer.heads[-1].gate.provisional


def test_growth_preserves_existing_parameters_optimizer_and_outputs_when_new_parts_disabled():
    model = GrowingTransformer(small())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    update(model, optimizer)
    parameters = dict(model.named_parameters())
    values = {name: p.clone() for name, p in parameters.items()}
    moments = {name: optimizer.state[p]["exp_avg"].clone() for name, p in parameters.items()}
    inputs = torch.randint(27, (2, 8))
    before = model(inputs).detach()
    for gate in model.gates():
        if gate.provisional:
            activate(gate)
    assert len(model.grow(optimizer, threshold=0.5, warmup=10)) == 3
    assert model.architecture() == [3, 3, 2]
    for name, parameter in parameters.items():
        assert dict(model.named_parameters())[name] is parameter
        torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], moments[name], rtol=0, atol=0)
    # Only the newly appended parts remain provisional, so this must be exact.
    torch.testing.assert_close(model(inputs, ablate_heads=True, ablate_layers=True), before, rtol=0, atol=0)
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len(set(optimized)) == len(list(model.parameters()))
    assert set(optimized) == {id(p) for p in model.parameters()}
    assert all(group["lr"] == 0.003 for group in optimizer.param_groups)
    new = model.layers[0].heads[-1].out.weight
    old_value = new.clone()
    assert torch.isfinite(update(model, optimizer))
    assert not torch.equal(new, old_value)


def test_threshold_warmup_caps_and_no_immediate_regrowth():
    model = GrowingTransformer(small(max_heads=3, max_layers=3))
    optimizer = torch.optim.AdamW(model.parameters())
    for gate in model.gates():
        if gate.provisional:
            activate(gate)
    assert model.grow(optimizer, threshold=0.9, warmup=10) == []
    assert model.grow(optimizer, threshold=0.5, warmup=11) == []
    assert len(model.grow(optimizer, threshold=0.5, warmup=10)) == 3
    assert model.grow(optimizer, threshold=0.5, warmup=10) == []
    for gate in model.gates():
        if gate.provisional:
            activate(gate)
    assert len(model.grow(optimizer, threshold=0.5, warmup=10)) == 1
    assert model.architecture() == [3, 3, 3]
    assert model.grow(optimizer, threshold=0.5, warmup=10) == []
    assert sum(g.provisional.item() for g in model.gates()) == 4


def test_ema_tracks_training_not_checks():
    model = GrowingTransformer(small())
    gate = model.layers[-1].gate
    with torch.no_grad():
        gate.logit.fill_(0)  # gate = 0.5
    model.observe_usage(decay=0.5)
    assert gate.ema.item() == pytest.approx(0.3)
    assert gate.age.item() == 1
    assert model.layers[0].gate.age.item() == 0


@pytest.mark.parametrize("adaptive", [True, False])
def test_checkpoint_exact_continuation_including_future_growth(tmp_path, adaptive):
    config = TrainConfig(model=small(adaptive=adaptive))
    model = GrowingTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters())
    generator = torch.Generator().manual_seed(18)
    update(model, optimizer)
    model.grow(optimizer, threshold=0, warmup=1)
    update(model, optimizer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, generator, config, 2, 1.2)

    expected_batch = torch.randint(1000, (10,), generator=generator)
    expected_loss = update(model, optimizer)
    model.grow(optimizer, threshold=0, warmup=1)
    restored, restored_optimizer, restored_generator, state = restore_checkpoint(path)
    assert state["step"] == 2 and state["training_seconds"] == 1.2
    assert torch.equal(expected_batch, torch.randint(1000, (10,), generator=restored_generator))
    torch.testing.assert_close(update(restored, restored_optimizer), expected_loss, rtol=0, atol=0)
    restored.grow(restored_optimizer, threshold=0, warmup=1)
    assert model.architecture() == restored.architecture()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_fixed_baseline():
    model = GrowingTransformer(small(adaptive=False))
    optimizer = torch.optim.AdamW(model.parameters())
    assert model.provisional_penalty().item() == 0
    assert all(g().item() == 1 and not g.logit.requires_grad for g in model.gates())
    assert torch.isfinite(update(model, optimizer))
    assert model.grow(optimizer, threshold=0, warmup=1) == []
    x = torch.randint(27, (2, 8))
    torch.testing.assert_close(model(x), model(x, ablate_heads=True, ablate_layers=True), rtol=0, atol=0)


def test_batch_next_character_and_split_bounds(tmp_path):
    generator = torch.Generator().manual_seed(0)
    x, y = batch(np.arange(20, dtype=np.uint8), 100, 8, generator, "cpu")
    assert torch.equal(y, x + 1)
    assert x.min() >= 0 and y.max() <= 19
    assert y.max() == 19  # Final legal starting position is included.
    assert SPLITS == {"train": (0, 90_000_000), "valid": (90_000_000, 95_000_000),
                      "test": (95_000_000, 100_000_000)}
    with pytest.raises(ValueError):
        batch(np.arange(8), 1, 8, generator, "cpu")
    with pytest.raises(ValueError):
        load_split(tmp_path, "train")


@pytest.mark.parametrize("length", [2, 8, 9, 17, 20, 25])
def test_full_evaluation_counts_tail_and_reports_bpc(length):
    config = TrainConfig(model=small(), batch_size=2)
    model = GrowingTransformer(config.model)
    with torch.no_grad():
        model.readout.weight.zero_()  # Uniform distribution: log2(27) bits/char.
    result = evaluate(model, np.arange(length, dtype=np.uint8) % 27, config, full=True)
    assert result["tokens"] == length - 1
    assert result["bpc"] == pytest.approx(math.log2(27), rel=1e-6)
    assert all(delta == 0 for delta in result["ablation_delta_bpc"].values())
    assert model.training


def test_sampled_evaluation_repeatable_and_rng_isolated():
    config = TrainConfig(model=small(), batch_size=2, eval_batches=3)
    model = GrowingTransformer(config.model).eval()
    state = torch.get_rng_state().clone()
    data = np.arange(100, dtype=np.uint8) % 27
    first = evaluate(model, data, config)
    assert first == evaluate(model, data, config)
    assert first["tokens"] == 48
    assert torch.equal(state, torch.get_rng_state())
    assert not model.training


def test_trainer_checks_every_n_steps_and_avoids_test_split(tmp_path, monkeypatch):
    trainer = importlib.import_module("growing_transformer.train")
    splits = []

    def data(directory, split):
        splits.append(split)
        return np.arange(100, dtype=np.uint8) % 27

    monkeypatch.setattr(trainer, "load_split", data)
    config = TrainConfig(model=small(max_heads=4, max_layers=4), output=str(tmp_path),
                         steps=4, batch_size=2, growth_interval=2, growth_warmup=2,
                         growth_threshold=0.09, eval_batches=1)
    model = trainer.train(config)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["kind"] == "growth_check"] == [2, 4]
    assert model.architecture() == [4, 4, 3, 2]
    assert splits == ["train", "valid"]
    restored, _, _, saved = restore_checkpoint(tmp_path / "latest.pt")
    assert saved["step"] == 4 and restored.architecture() == model.architecture()
    with pytest.raises(ValueError, match="Run already exists"):
        trainer.train(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in CPU orb")
def test_cuda_expansion():
    model = GrowingTransformer(small()).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    for gate in model.gates():
        if gate.provisional:
            activate(gate)
    model.grow(optimizer, threshold=0.5, warmup=10)
    assert all(p.is_cuda for p in model.parameters())
    x = torch.randint(27, (2, 8), device="cuda")
    language_loss(model(x), x).backward()
    optimizer.step()
