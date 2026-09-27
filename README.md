# Growing Transformer

An experimental character-level causal transformer whose attention-head counts and
depth grow independently during training. It starts with **two layers and two heads
per layer**. The last layer and the last head of every layer are provisional.

**Status:** implemented and CPU-smoke-tested on real text8. This is not a completed
training experiment or evidence that the method discovers a better architecture.
GPU execution is not yet verified.

## Mechanism

Each head has its own Q/K/V projection, fixed head width, output projection into a
fixed residual width, and learnable sigmoid gate. Head outputs are summed with a
constant 1/√2 factor; adding a head never changes existing head widths or rescales
their contributions. This is equivalent to concatenated heads followed by an output
projection split into per-head blocks, but allows arbitrary head counts.

Each pre-norm layer has attention and feed-forward residual updates, both scaled by
its layer gate. Disabling a layer therefore leaves an identity path. Provisional
gates start at 0.1; initially non-provisional gates start at 0.9. All adaptive-model
gates remain learnable.

The objective is mean next-character cross-entropy plus `penalty` times the sum of
provisional head and layer gates. The sum is not normalized by component count.
Heads inside the provisional layer receive their own penalty in addition to the
layer penalty.

After each optimizer step, provisional gate values update an exponential moving
average. Every `growth_interval` steps, a component qualifies if its EMA reaches
`growth_threshold` and its age reaches `growth_warmup`. Each layer can independently
promote its last head and append one new provisional head. The model can independently
promote its last layer and append a two-head layer, whose last head is provisional.
Newly added parts must complete their own warm-up; they cannot trigger further
growth in the same check.

Promotion removes the provisional penalty without resetting the gate, weights, or
Adam moments. New parts have small but nonzero initial contributions, so expansion
can perturb predictions. Existing parameters are never replaced; new parameters
get new optimizer groups with matching hyperparameters. At a head/layer cap, that
component stays provisional and penalized, with no further expansion in that dimension.

**Gate magnitude is only a usage proxy:** projection weights can compensate for
small gates. Validation measures paired ablations of all provisional heads, the
provisional layer, and both together. Positive ablation delta in bits/character
means disabling those parts hurts prediction. These are diagnostics, not an
additional growth criterion or per-component attribution. The implementation does
not claim to find an optimal architecture, prune unused components, or remove
remaining provisional parts automatically.

## Install and test

Run from the repository root with Python 3.11 or later:

```bash
uv venv
# CPU installation used in this orb:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m growing_transformer.data
```

For a GPU machine, install a CUDA-enabled PyTorch build appropriate to that node
instead of the CPU wheel. Data preparation downloads [text8 from Matt Mahoney's
site](https://mattmahoney.net/dc/textdata.html), checks a pinned archive checksum,
validates its length and alphabet, and writes memory-mapped token files. Text8 is
not bundled; consult the source for dataset terms.

The alphabet is space plus a–z. The contiguous split is **90 million training
characters, 5 million validation characters, and 5 million test characters**.
Training samples only from the training split. Targets are shifted by one
character; attention is causal.

## Run

```bash
# Ordinary smoke test; no growth is guaranteed.
.venv/bin/python -m growing_transformer.train --config configs/smoke.json

# Same small training budget, fixed two-layer/two-head ungated baseline.
.venv/bin/python -m growing_transformer.train --config configs/smoke.json --fixed --output runs/fixed-smoke

# Deliberately permissive threshold: tests machinery, NOT learned architecture quality.
.venv/bin/python -m growing_transformer.train --config configs/growth-smoke.json

# Continue to a total of 16 steps using the checkpoint's saved configuration.
.venv/bin/python -m growing_transformer.train --resume runs/growth-smoke/latest.pt --steps 16

# Future single-GPU experiment; this configuration is a starting point, not tuned.
.venv/bin/python -m growing_transformer.train --config configs/gpu.json
.venv/bin/python -m growing_transformer.train --config configs/gpu.json --fixed --output runs/gpu-fixed
```

Completed smoke runs already exist in the original Amp orb workspace. Choose a
fresh `--output` directory to repeat them; the trainer refuses to overwrite an
existing run unless resuming. Run directories contain the resolved configuration,
JSONL metrics, an atomic `latest.pt` checkpoint, and retained snapshots under
`checkpoints/` at initialization, checkpoint intervals, expansion, and completion.
Checkpoints store architecture, provisional flags, usage EMAs/ages, parameters,
optimizer groups/moments, step count,
and PyTorch/batch RNG states. CPU continuation, including subsequent expansion,
is tested for exact equality. Cross-device or cross-version bitwise equality is
not promised.

Training uses AdamW, a constant learning rate, gradient clipping, and FP32 on one
device. There is no distributed-training or mixed-precision implementation. Logs
record every step's loss/BPC, accuracy, pre-clipping gradient norm, learning rates,
architecture, parameter counts, tokens, and training time. `log_interval` controls
console progress and detailed gate, parameter-norm, and CUDA-memory diagnostics;
it does not downsample step records in `metrics.jsonl`. Growth checks include a
paired before/after-expansion loss probe. Run metadata includes the code revision,
software versions, hardware, UTC timestamps, and a unique ID for each session.
Training-step time excludes evaluation, checkpointing, expansion, and JSON writing;
separate wall-time and operation-duration fields capture overhead.

For cluster execution, see the [tue-hpc runbook](reports/tue-hpc-runbook.md) and
`scripts/tue-hpc.sbatch`. Use `--data` for prepared text8 on scratch and a fresh
`--output` for every resumed allocation so interrupted attempts remain available.
The first GPU run is exploratory: retain observations and checkpoints, defer
interpretation, and do not alter growth settings based on whether growth occurs.

## Evaluation

```bash
# Full validation, not a sampled estimate:
.venv/bin/python -m growing_transformer.train --resume runs/smoke/latest.pt --eval-only --full-eval

# Reserve test evaluation for the final selected experiment:
.venv/bin/python -m growing_transformer.train --resume runs/gpu-adaptive/latest.pt --eval-only --eval-split test --full-eval
```

Training-time validation uses fixed-seed sampled windows with a separate RNG.
Every ablation uses the same windows. Full evaluation scores every character after
the initial token exactly once, resetting context at non-overlapping block
boundaries and including any short final block. Thus it evaluates 4,999,999 targets
per held-out split. This context convention should be matched when comparing with
other implementations. Bits per character is cross-entropy in nats divided by ln(2);
lower is better. Evaluation never includes the gate penalty.

## CPU smoke results — 2026-09-08

Python 3.11.6, PyTorch 2.14.0+cpu, NumPy 2.4.6, two CPU threads, seed 42. Both
ordinary runs used residual width 32, head width 8, context 64, batch size 8, and
100 optimizer steps (**only 51,200 training tokens**, sampled with replacement).

| Run | Sampled validation BPC, before → after | Full validation BPC after | Final heads per layer |
| --- | --- | --- | --- |
| Adaptive | 4.8469 → 3.8342 | 3.8113 | [2, 2] |
| Fixed baseline | 4.8573 → 3.8014 | 3.7703 | [2, 2] |

The adaptive model did not cross the normal 0.5 gate threshold. The permissive
0.09-threshold run expanded from [2, 2] to [3, 3, 2] at step 4, [4, 4, 3, 2] at
step 8, [4, 4, 4, 3] at step 12, then [4, 4, 4, 4] after resuming through step 16.
Its threshold starts below the initial gate value: growth is intentionally induced,
not evidence of learned demand.

For the ordinary adaptive run, full-validation ablation deltas were +0.000616 BPC
for provisional heads, +0.018146 for the provisional layer, and +0.020048 for both.
These results establish that the pipeline executes and learns something; **the
fixed baseline is better at this tiny budget**. No test-set evaluation or substantial
training was performed. Raw metrics and full-validation outputs are preserved in
[`reports/`](reports/).

Verification: `python -m pytest -q` produced **19 passed, 1 skipped** (CUDA unavailable).
Tests cover causality, penalty gradients, independent/simultaneous growth, warm-up
and caps, existing-weight/optimizer preservation, learning after expansion, exact
checkpoint continuation, trainer cadence, data alignment, and evaluation accounting.

The first GPU run records growth behavior without tuning toward a preferred result.
Follow-up studies can vary penalty strength and growth threshold, run multiple
seeds, and compare fixed/adaptive models at matched token and compute budgets.
Inspect ablation deltas before concluding that growth reflects useful capacity.
