import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import batch, load_split
from .model import GrowingTransformer, ModelConfig


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: str = "data/text8"
    output: str = "runs/adaptive"
    device: str = "cpu"
    seed: int = 42
    threads: int = 2
    steps: int = 100_000
    batch_size: int = 32
    lr: float = 0.0003
    weight_decay: float = 0.01
    penalty: float = 0.01
    growth_interval: int = 1000
    growth_warmup: int = 1000
    growth_threshold: float = 0.5
    ema_decay: float = 0.95
    log_interval: int = 100
    eval_interval: int = 1000
    eval_batches: int = 20
    checkpoint_interval: int = 1000

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        for name in ("threads", "steps", "batch_size", "growth_interval", "growth_warmup",
                     "log_interval", "eval_interval", "eval_batches", "checkpoint_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.growth_threshold <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid growth threshold or EMA decay")
        if self.lr <= 0 or self.penalty < 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or loss/decay coefficient")
        if self.model.vocab_size != 27:
            raise ValueError("text8 requires vocabulary size 27")


def language_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model, data, config, *, full=False):
    """Paired ablations on identical tokens, isolated from the training RNG.

    Full evaluation scores every target except the split's first character once,
    resetting context at non-overlapping block boundaries (including a short tail).
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1)
    modes = {"base": {}, "without_heads": {"ablate_heads": True},
             "without_layer": {"ablate_layers": True},
             "without_both": {"ablate_heads": True, "ablate_layers": True}}
    totals = dict.fromkeys(modes, 0.0)
    count = 0

    def batches():
        if not full:
            for _ in range(config.eval_batches):
                yield batch(data, config.batch_size, model.config.context, generator, config.device)
            return
        context = model.config.context
        full_blocks = (len(data) - 1) // context
        for index in range(0, full_blocks, config.batch_size):
            starts = range(index * context, min(index + config.batch_size, full_blocks) * context, context)
            seq = np.stack([data[start:start + context + 1] for start in starts])
            tokens = torch.from_numpy(seq.astype(np.int64)).to(config.device)
            yield tokens[:, :-1], tokens[:, 1:]
        tail = full_blocks * context
        if tail < len(data) - 1:
            tokens = torch.tensor(np.array(data[tail:], dtype=np.int64), device=config.device)[None]
            yield tokens[:, :-1], tokens[:, 1:]

    try:
        for x, y in batches():
            for name, kwargs in modes.items():
                totals[name] += language_loss(model(x, **kwargs), y).item() * y.numel()
            count += y.numel()
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Evaluation needs at least two characters")
    bpc = {name: total / count / math.log(2) for name, total in totals.items()}
    return {"bpc": bpc["base"], "tokens": count, "full": full,
            "ablation_delta_bpc": {name: value - bpc["base"] for name, value in bpc.items()
                                   if name != "base"}}


def save_checkpoint(path, model, optimizer, generator, config, step, training_seconds):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {
        "version": 1, "spec": model.spec(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "parameter_groups": [[names[id(p)] for p in group["params"]]
                             for group in optimizer.param_groups],
        "config": asdict(config), "step": step, "training_seconds": training_seconds,
        "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    model = GrowingTransformer(ModelConfig(**state["spec"]["config"]),
                               state["spec"]["head_counts"]).to(device)
    model.load_state_dict(state["model"])
    parameters = dict(model.named_parameters())
    groups = [{"params": [parameters[name] for name in names]}
              for names in state["parameter_groups"]]
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(state["optimizer"])
    generator = torch.Generator()
    generator.set_state(state["batch_rng"])
    torch.set_rng_state(state["torch_rng"])
    if str(device).startswith("cuda") and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model, optimizer, generator, state


def train(config, resume=None):
    torch.set_num_threads(config.threads)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    if not resume and (log_path.exists() or (output / "latest.pt").exists()):
        raise ValueError("Run already exists; choose another output or resume its checkpoint")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    training = load_split(Path(config.data), "train")
    validation = load_split(Path(config.data), "valid")
    if resume:
        model, optimizer, generator, state = restore_checkpoint(resume, config.device)
        start, training_seconds = state["step"], state["training_seconds"]
    else:
        torch.manual_seed(config.seed)
        model = GrowingTransformer(config.model).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        generator = torch.Generator().manual_seed(config.seed)
        start, training_seconds = 0, 0.0

    def log(record):
        with log_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def validation_record(step):
        return {"kind": "validation", "step": step, "architecture": model.architecture(),
                **evaluate(model, validation, config)}

    log({"kind": "start", "resumed": bool(resume), "step": start,
         "device": config.device, "torch": str(torch.__version__), "config": asdict(config)})
    log(validation_record(start))
    model.train()
    for step in range(start + 1, config.steps + 1):
        if str(config.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        x, y = batch(training, config.batch_size, config.model.context, generator, config.device)
        optimizer.zero_grad(set_to_none=True)
        ce = language_loss(model(x), y)
        penalty = model.provisional_penalty()
        loss = ce + config.penalty * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.observe_usage(config.ema_decay)
        if str(config.device).startswith("cuda"):
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started

        if step % config.growth_interval == 0:
            before_usage = model.usage()
            events = model.grow(optimizer, threshold=config.growth_threshold, warmup=config.growth_warmup)
            log({"kind": "growth_check", "step": step, "events": events,
                 "usage_before": before_usage, "architecture": model.architecture()})
        if step % config.log_interval == 0 or step == config.steps:
            log({"kind": "train", "step": step, "ce": ce.item(), "loss": loss.item(),
                 "penalty_unscaled": penalty.item(), "architecture": model.architecture(),
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "tokens_seen": step * config.batch_size * config.model.context,
                 "training_seconds": training_seconds, "usage": model.usage()})
        if step % config.eval_interval == 0 or step == config.steps:
            log(validation_record(step))
        if step % config.checkpoint_interval == 0 or step == config.steps:
            save_checkpoint(output / "latest.pt", model, optimizer, generator, config, step, training_seconds)
    return model


def main():
    parser = argparse.ArgumentParser(description="Grow a character-level transformer on text8")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including steps before resume")
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--fixed", action="store_true", help="Fixed 2-layer/2-head ungated baseline")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--full-eval", action="store_true")
    args = parser.parse_args()
    if args.resume and (args.config or args.fixed):
        parser.error("Resume uses the saved configuration; do not pass --config or --fixed")
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    if not args.eval_only and (args.full_eval or args.eval_split != "valid"):
        parser.error("--full-eval and --eval-split require --eval-only")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        values = saved["config"]
    else:
        values = json.loads(args.config.read_text()) if args.config else {}
    for key in ("steps", "device", "output"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.fixed:
        values.setdefault("model", {})["adaptive"] = False
    config = TrainConfig(**values)
    if args.eval_only:
        torch.set_num_threads(config.threads)
        model, _, _, state = restore_checkpoint(args.resume, config.device)
        result = evaluate(model, load_split(Path(config.data), args.eval_split), config, full=args.full_eval)
        print(json.dumps({"step": state["step"], "split": args.eval_split,
                          "architecture": model.architecture(), **result}, allow_nan=False))
    else:
        if args.resume and config.steps <= saved["step"]:
            parser.error("--steps must exceed the saved step to resume training")
        train(config, args.resume)


if __name__ == "__main__":
    main()
