# Adaptive seed-42 V100 run

This directory summarizes the first exploratory GPU experiment. The complete raw
payload is stored in `artifacts.tar.zst`; see the parent
[results documentation](../../README.md) for extraction and verification commands.

## Provenance

- Code revision: `00e934e7bff4d0356693eb7456afdf35c4765309`
- Research allocation: Slurm job `314435`, Tesla V100-PCIE-16GB, FP32
- Setup allocation: job `314428` stopped after two seconds at the revision guard;
  no CUDA test or research training ran in that allocation
- Total allocated GPU time: 1,294 seconds (0.359444 GPU-hours)
- Environment: Python 3.11.3, PyTorch 2.6.0+cu118, NumPy 2.2.6
- Preflight: 23 passed, no skips; CUDA growth/resume and maximum-capacity checks
  passed before research training

## Approved experiment

One adaptive seed-42 run trained on text8 for 100,000 optimizer steps with batch
32 and context 256 (819,200,000 training targets). It started with two layers and
two heads per layer. Width and depth were capped at eight, with growth checks every
1,000 steps. No sweep, fixed control, tuning, forced growth, mixed precision, or
test-set evaluation was performed.

## Observations

- Final architecture: `[2, 2, 2]`, with one layer added at step 19,000
- Final parameter count: 534,921
- Final sampled validation: 1.726546567 BPC on 163,840 targets
- Full validation: 1.746946335 BPC on exactly 4,999,999 targets
- Full-validation ablation deltas: +0.259501770 BPC without provisional heads,
  +0.900198698 without the provisional layer, and +1.182816270 without both
- Final checkpoint: step 100,000; SHA-256
  `0c2b62fc17730f826f9de0d0b01f4a8a9892d6804587b8a51f8a37763999423c`
- Retained research snapshots: 101 (initial, every 1,000 steps, and final)

These are observations from one exploratory run, not evidence that the growth
rule is better or worse than a fixed architecture.

## Archive integrity

- Archived source files: 125
- Uncompressed payload: 686,370,617 bytes
- Compressed archive: 581,122,158 bytes
- Archive SHA-256:
  `baa6af02401916781172f8dfd043ddaecb683c71dc699491279b2b49a83873d5`
- `SHA256SUMS` SHA-256:
  `9281655d8725780f1126440a77dec7df42c1cd583e8e139c15fa0eca80e68612`

The payload includes all retained checkpoints, resolved configurations, metrics,
environment/GPU records, preflight evidence, validation output, and scheduler and
console logs. It excludes text8 data, the Python environment, caches, and temporary
files.
