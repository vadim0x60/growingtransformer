import importlib
import json
import math
from pathlib import Path
import time

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
                         growth_threshold=0.09, eval_batches=1, log_interval=3)
    model = trainer.train(config)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["kind"] == "growth_check"] == [2, 4]
    steps = [r for r in records if r["kind"] == "train"]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    assert [r["architecture"] for r in steps] == [[2, 2], [2, 2], [3, 3, 2], [3, 3, 2]]
    assert [r["tokens_seen"] for r in steps] == [16, 32, 48, 64]
    for record in steps:
        assert record["bpc"] == pytest.approx(record["ce"] / math.log(2))
        assert record["loss"] == pytest.approx(record["ce"] + 0.01 * record["penalty_unscaled"])
        assert record["gradient_norm_before_clip"] > 0
        assert 0 <= record["accuracy"] <= 1
    diagnostics = [r for r in records if r["kind"] == "diagnostics"]
    assert [r["step"] for r in diagnostics] == [3, 4]
    assert "layers.2.heads.1.out.weight" in diagnostics[0]["parameter_norms_before_update"]
    assert "layers.3.heads.0.out.weight" not in diagnostics[1]["parameter_norms_before_update"]
    checks = [r for r in records if r["kind"] == "growth_check"]
    assert checks[0]["architecture_before"] == [2, 2]
    assert checks[0]["architecture"] == [3, 3, 2]
    assert checks[0]["probe_ce_before"] != checks[0]["probe_ce_after"]
    assert len({r["session_id"] for r in records}) == 1
    assert all(a["wall_seconds"] <= b["wall_seconds"] for a, b in zip(records, records[1:]))
    assert records[-1]["kind"] == "complete"
    # Retain event checkpoints even when checkpoint_interval has not elapsed.
    paths = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert [p.name for p in paths] == ["step-000000000.pt", "step-000000002.pt", "step-000000004.pt"]
    assert restore_checkpoint(paths[1])[0].architecture() == [3, 3, 2]
    assert model.architecture() == [4, 4, 3, 2]
    assert splits == ["train", "valid"]
    restored, _, _, saved = restore_checkpoint(tmp_path / "latest.pt")
    assert saved["step"] == 4 and restored.architecture() == model.architecture()
    with pytest.raises(ValueError, match="Run already exists"):
        trainer.train(config)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable in CPU orb"))])
def test_instrumented_trainer_resume_through_growth(tmp_path, monkeypatch, device):
    trainer = importlib.import_module("growing_transformer.train")
    monkeypatch.setattr(trainer, "load_split", lambda *_: np.arange(100, dtype=np.uint8) % 27)
    config = TrainConfig(model=small(max_heads=4, max_layers=4), device=device,
                         output=str(tmp_path / "continuous"), steps=5, batch_size=2,
                         growth_interval=2, growth_warmup=2, growth_threshold=0.09,
                         eval_batches=1, log_interval=2, checkpoint_interval=3)
    expected = trainer.train(config)
    source = Path(config.output) / "checkpoints" / "step-000000002.pt"
    source_bytes = source.read_bytes()
    config.output = str(tmp_path / "resumed")
    actual = trainer.train(config, resume=source)
    assert actual.architecture() == expected.architecture() == [4, 4, 3, 2]
    tolerance = 0 if device == "cpu" else 1e-6
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=tolerance, atol=tolerance)
    assert source.read_bytes() == source_bytes
    old_records = [json.loads(s) for s in (tmp_path / "continuous" / "metrics.jsonl").read_text().splitlines()]
    new_records = [json.loads(s) for s in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert new_records[0]["session_id"] != old_records[0]["session_id"]
    assert new_records[0]["resume_from"] == str(source)
    assert [r["step"] for r in new_records if r["kind"] == "train"] == [3, 4, 5]
    assert [r["ce"] for r in new_records if r["kind"] == "train"] == pytest.approx([
        r["ce"] for r in old_records if r["kind"] == "train" and r["step"] > 2],
        rel=tolerance, abs=tolerance)
    saved = torch.load(source, weights_only=True)
    assert new_records[0]["wall_seconds"] >= saved["wall_seconds"] > 0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in CPU orb")
def test_cuda_gpu_config_at_capacity():
    config = TrainConfig(**json.loads((Path(__file__).parents[1] / "configs/gpu.json").read_text()))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = GrowingTransformer(config.model, [config.model.max_heads] * config.model.max_layers).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    x = torch.randint(27, (config.batch_size, config.model.context), device="cuda")
    y = (x + 1) % 27
    started = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = language_loss(model(x), y) + config.penalty * model.provisional_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(loss)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(json.dumps({"capacity_probe": model.architecture(), "peak_allocated_bytes": peak,
                      "seconds_for_three_steps": time.perf_counter() - started}))
    assert peak < 0.8 * torch.cuda.get_device_properties(0).total_memory
