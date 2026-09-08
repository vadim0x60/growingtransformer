from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 27
    d_model: int = 128
    head_dim: int = 32
    ff_mult: int = 4
    context: int = 256
    gate_init: float = 0.1
    max_heads: int = 8
    max_layers: int = 8
    adaptive: bool = True

    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.head_dim, self.ff_mult, self.context) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 < self.gate_init < 1:
            raise ValueError("gate_init must be between zero and one")
        if self.max_heads < 2 or self.max_layers < 2:
            raise ValueError("Caps must allow the initial two layers/two heads")


class Gate(nn.Module):
    def __init__(self, initial: float, provisional: bool):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))
        self.register_buffer("provisional", torch.tensor(provisional))
        self.register_buffer("ema", torch.tensor(initial))
        self.register_buffer("age", torch.tensor(0, dtype=torch.long))
        self.fixed = False

    def forward(self):
        return torch.ones_like(self.logit) if self.fixed else self.logit.sigmoid()

    @torch.no_grad()
    def observe(self, decay: float):
        self.ema.lerp_(self(), 1 - decay)
        self.age.add_(1)


class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig, provisional: bool):
        super().__init__()
        self.qkv = nn.Linear(config.d_model, 3 * config.head_dim, bias=False)
        self.out = nn.Linear(config.head_dim, config.d_model, bias=False)
        self.gate = Gate(config.gate_init if provisional else 0.9, provisional)

    def forward(self, x, ablate: bool = False):
        if ablate and self.gate.provisional:
            return torch.zeros_like(x)
        q, k, v = self.qkv(x).unsqueeze(1).chunk(3, dim=-1)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.gate() * self.out(attention.squeeze(1))


class Layer(nn.Module):
    def __init__(self, config: ModelConfig, heads: int, provisional: bool):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(config, config.adaptive and i == heads - 1)
            for i in range(heads)
        ])
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ff_mult * config.d_model),
            nn.GELU(),
            nn.Linear(config.ff_mult * config.d_model, config.d_model),
        )
        self.gate = Gate(config.gate_init if provisional else 0.9, provisional)

    def forward(self, x, ablate_heads=False, ablate_layers=False):
        if ablate_layers and self.gate.provisional:
            return x
        normalized = self.attention_norm(x)
        # Fixed scaling: adding a head must not rescale existing contributions.
        attention = sum(h(normalized, ablate_heads) for h in self.heads) / math.sqrt(2)
        x = x + self.gate() * attention
        return x + self.gate() * self.ff(self.ff_norm(x))


class GrowingTransformer(nn.Module):
    def __init__(self, config: ModelConfig, head_counts: list[int] | None = None):
        super().__init__()
        self.config = config
        counts = [2, 2] if head_counts is None else head_counts
        if not 2 <= len(counts) <= config.max_layers or any(
            not 2 <= count <= config.max_heads for count in counts
        ):
            raise ValueError("Architecture exceeds configured bounds")
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.context, config.d_model)
        self.layers = nn.ModuleList([
            Layer(config, count, config.adaptive and i == len(counts) - 1)
            for i, count in enumerate(counts)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if not config.adaptive:
            # A conventional fixed baseline: no attenuation or gate learning.
            for gate in self.gates():
                gate.fixed = True
                gate.logit.requires_grad_(False)

    def gates(self):
        for layer in self.layers:
            yield layer.gate
            for head in layer.heads:
                yield head.gate

    def forward(self, tokens, *, ablate_heads=False, ablate_layers=False):
        if tokens.shape[1] > self.config.context:
            raise ValueError("Sequence exceeds configured context")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        for layer in self.layers:
            x = layer(x, ablate_heads, ablate_layers)
        return self.readout(self.norm(x))

    def provisional_penalty(self):
        # Sum (not mean) so adding capacity does not dilute the penalty.
        result = self.embedding.weight.new_zeros(())
        for gate in self.gates():
            if gate.provisional:
                result = result + gate()
        return result

    @torch.no_grad()
    def observe_usage(self, decay=0.95):
        for gate in self.gates():
            if gate.provisional:
                gate.observe(decay)

    def architecture(self):
        return [len(layer.heads) for layer in self.layers]

    def usage(self):
        def record(gate):
            return {"gate": gate().item(), "ema": gate.ema.item(),
                    "age": gate.age.item(), "provisional": gate.provisional.item()}
        return [{"layer": record(layer.gate),
                 "heads": [record(head.gate) for head in layer.heads]}
                for layer in self.layers]

    @torch.no_grad()
    def grow(self, optimizer, *, threshold=0.5, warmup=1000):
        """Check the pre-expansion frontier once; preserve all existing Parameters."""
        if not 0 <= threshold <= 1 or warmup < 1:
            raise ValueError("Invalid growth threshold or warmup")
        if not self.config.adaptive:
            return []
        events = []
        reference = self.embedding.weight

        def ready(gate):
            return bool(gate.provisional and gate.age >= warmup and gate.ema >= threshold)

        def register(module):
            module.to(device=reference.device, dtype=reference.dtype)
            module.train(self.training)
            # Match current LR and all optimizer hyperparameters (including schedulers).
            group = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
            optimizer.add_param_group({**group, "params": list(module.parameters())})

        for index, layer in enumerate(self.layers):
            last = layer.heads[-1]
            if len(layer.heads) < self.config.max_heads and ready(last.gate):
                new_head = AttentionHead(self.config, provisional=True)
                register(new_head)
                last.gate.provisional.fill_(False)
                layer.heads.append(new_head)
                events.append({"kind": "head", "layer": index, "heads": len(layer.heads)})

        last_layer = self.layers[-1]
        if len(self.layers) < self.config.max_layers and ready(last_layer.gate):
            new_layer = Layer(self.config, heads=2, provisional=True)
            register(new_layer)
            last_layer.gate.provisional.fill_(False)
            self.layers.append(new_layer)
            events.append({"kind": "layer", "layers": len(self.layers)})
        return events

    def spec(self):
        return {"config": asdict(self.config), "head_counts": self.architecture()}
