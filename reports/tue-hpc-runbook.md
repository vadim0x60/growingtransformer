# First exploratory text8 GPU run

This is an observation run, not a benchmark win/loss test. Train **one adaptive
model, seed 42**, with `configs/gpu.json`: 100,000 optimizer steps, batch 32,
context 256 (819,200,000 target characters), FP32 AdamW, constant LR 0.0003,
weight decay 0.01, penalty 0.01, threshold 0.5, growth checks/warm-up every
1,000 steps. Start at two layers/two heads; cap at eight layers/eight heads.
Do not tune settings, force growth, add seeds, or run controls during this run.
No growth, early growth, reaching caps, and worse validation are all observations.
Reserve the test split. Use the final checkpoint, not best-validation selection.

## Authorization and budget

Use only the **full commit ID named in the designer's handoff**. This document
alone does not authorize submission. Wait for confirmed home cleanup and the
explicit instruction that the approved run may start.

Request `mcs.gpu.q`, `gpu:tesla_v100-pcie-16gb:1`, one task/node, four CPUs,
16 GiB host RAM, at most 24 hours. The total budget is **24 allocated GPU-hours**,
including preflight, evaluation, and any recovery jobs; pending time does not
count. Permit at most two recovery submissions for transient infrastructure
failures, resolved storage interruptions, or interrupted final evaluation, each
with its walltime reduced to the remaining budget. A failed preflight, numerical
failure, or unexplained stall requires the designer's clearance before retrying;
do not independently change the code or environment to bypass a failed check.
No concurrent attempts, automatic requeue, distributed training, or CPU fallback.

## Prepare the isolated environment and data

Run from the clean approved checkout on tue-hpc. Do not use the broken user-site
torch or the older PyTorch module. Do not install packages into home. Confirm at
least 20 GiB of scratch quota remains before beginning. Run `myquota`
to inspect both space and file quotas; require at least 50,000 free scratch file
entries for the environment. Keep generated run data and caches out of home.
Do not infer personal quota headroom from `df` filesystem totals.

```bash
module load Umbrella/2024
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONNOUSERSITE=1
unset PYTHONPATH
BASE=/scratch-shared/20194474/growingtransformer
mkdir -p "$BASE/tmp" "$BASE/pip-cache" "$BASE/data"
export TMPDIR="$BASE/tmp" PIP_CACHE_DIR="$BASE/pip-cache"
python -m venv "$BASE/venv-torch260-cu118"
PYTHON="$BASE/venv-torch260-cu118/bin/python"
"$PYTHON" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install 'numpy==2.2.6' 'pytest==8.3.5'
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" -m pip check
"$PYTHON" -m growing_transformer.data --directory "$BASE/data/text8"
# Do not export the login-node setup temp path into a batch allocation.
unset TMPDIR
```

The CUDA 11.8 wheel supports V100; actual driver/device operation must pass the
job's preflight. Record the resolved environment with `pip freeze` (the job does
this). Avoid package upgrades between attempts. The code has been CPU-tested
with PyTorch 2.6.0, NumPy 2.2.6, and pytest 8.3.5; that is not GPU verification.
The batch script routes model/CUDA caches to scratch and leaves temporary files
to the cluster's job-local `$TMPDIR` (or node-local `/tmp` if unset).

## Submit once, after the handoff

Set `EXPECTED_REVISION` to the exact full ID in the handoff, not whatever happens
to be the remote tip. Submit from that checkout; Slurm preserves the working
directory. Use `sbatch --test-only` first, then make one real submission.

```bash
export EXPECTED_REVISION="FULL_COMMIT_ID_FROM_HANDOFF"
test "$(git rev-parse HEAD)" = "$EXPECTED_REVISION"
BASE=/scratch-shared/20194474/growingtransformer
RUN="$BASE/$EXPECTED_REVISION/adaptive-seed42"
mkdir -p "$RUN"
sbatch --test-only --output="$RUN/slurm-%j.out" --error="$RUN/slurm-%j.err" \
    scripts/tue-hpc.sbatch
# Run the following only if the test-only request succeeds:
sbatch --parsable --output="$RUN/slurm-%j.out" --error="$RUN/slurm-%j.err" \
    scripts/tue-hpc.sbatch
```

The script first verifies the CUDA build and V100, then runs the complete test
suite, including CUDA expansion/resume and the full batch/context at the maximum
architecture. The capacity test requires peak allocated memory below 80% of GPU
VRAM. Expect **23 passed, no skips**. Tests have a 30-minute timeout. It then runs
the separate 12-step permissive growth smoke test and resumes to step 16 (each
has a 10-minute timeout). These smoke outputs are not research results.

Only after those checks pass does the script execute the research command:

```bash
"$PYTHON" -m growing_transformer.train --config configs/gpu.json \
    --data "$BASE/data/text8" --output "$RUN/job-$SLURM_JOB_ID/training"
```

After step 100,000, the script runs full validation with paired ablations, never
test evaluation. Full validation must report 4,999,999 targets. If only this last
evaluation is interrupted, use the evaluation-only recovery command below,
rather than restarting training.

## Outputs and measurement semantics

Each allocation gets a fresh `job-<id>` directory containing environment and GPU
details, console/preflight logs, smoke outputs (first attempt only), `training/`,
and, after successful final evaluation, `full-validation.json`. `training/` has:

- `config.json`: fully resolved configuration, including data/output paths.
- `metrics.jsonl`: UTC timestamp, session ID, and active wall time on every record.
  Every step records CE/BPC, total loss, penalty, accuracy, pre-clipping gradient
  norm, LR per optimizer group, architecture, parameters, tokens, and step time.
- Every 100 steps: all gate values/EMAs/ages/provisional flags, weight and gradient
  L2 norms for every named parameter, and CUDA allocated/reserved/peak memory.
  Module names remain stable because growth only appends components.
- Every 1,000 steps: growth checks with before/after architecture and gate state,
  plus a paired same-training-batch CE probe immediately before/after expansion;
  sampled validation BPC and joint provisional-component ablations on fixed
  windows. These ablations are not individual-component attribution.
- `latest.pt` and `checkpoints/step-NNNNNNNNN.pt`: initial, every 1,000 steps,
  after every actual expansion, and final snapshots. All include parameters,
  architecture, optimizer state, gate state, RNG states, and completed step count.
  Retain them all for later analyses; do not keep only the final model.

Training losses describe the pre-update model; the accompanying architecture is
the one used for that step, before any expansion. Diagnostic gradients/weights
are measured before clipping/update; gate diagnostics are after the update.
Growth probes are after the optimizer update, and differ only by expansion.
They do not measure whether growth helps later learning.
An architecture list contains the head count of each layer, from input to output.
Head and layer additions in the same growth check are simultaneous decisions;
their order in the event list is implementation order, not evidence that width
growth caused or preceded depth growth during learning.

`step_seconds` includes periodic norm collection but excludes JSON writing,
evaluation, expansion, and checkpointing. `wall_seconds` includes in-process
overhead; on resume it continues from the saved checkpoint's wall-time sample
(taken before that checkpoint's I/O). It excludes queue time, work lost after the
checkpoint, and prior checkpoint-save overhead at the resume boundary. For total
compute cost, retain Slurm `sacct` allocation elapsed times across **all attempts**.
Session IDs and `resume_from` identify branches; never blindly concatenate
overlapping step ranges as additional training progress.

## Health checks and intervention policy

Check after startup, at the first validation/checkpoint, and at least every
15 minutes while running. Record interventions with UTC time, job ID, and reason.

| Observation | Action |
| --- | --- |
| Failed/skipped GPU test, wrong device/build, non-finite loss/gradient, OOM, corrupted data/checkpoint, write failure | Stop/hold. Preserve logs. Report to the designer; do not change batch size, precision, LR, gates, or caps to make it continue. |
| No metrics progress for 15 minutes while Slurm says RUNNING | Inspect Slurm state, process/GPU activity, and whether evaluation/checkpointing is active. If still stalled at a second check five minutes later, cancel and report; do not launch a duplicate. |
| Peak GPU allocated memory above 90%, sustained step time above 5× the preceding 1,000-step median, or validation BPC >1 bit above the prior evaluation | Report and investigate, but do not stop or tune solely for these observations. Growth can change timing and quality. |
| Scratch quota headroom below 10 GiB | Cancel before storage exhaustion, preserve existing artifacts, and resolve storage before resuming. Do not delete retained checkpoints. |
| No growth, saturated caps, tiny gates, or worse validation | Record; continue within budget. These are not operational failures. |
| Step 100,000 or total allocated budget exhausted | End training. If budget expires first, label it a budget-limited partial run, not a completed 100,000-step run. |

There is no signal-triggered emergency checkpoint. Cancel with Slurm `scancel`
only for the approved reasons; use the latest **completed, loadable** checkpoint.
Atomic replacement protects `latest.pt`, but up to 999 steps can be lost with
this configuration between save boundaries, or more if interrupted during
evaluation/checkpoint I/O. Only the saved step is a durable recovery point.
Ignore incomplete `.tmp` files; never resume from them.

For a transient node/preemption failure, confirm the old job has stopped, verify
the checkpoint loads, record `sacct` elapsed time, and submit the same script with
the absolute checkpoint path as its first argument. Override `--time` with the
remaining budget. The script uses `--resume ... --steps 100000` and a new output
directory, preserving the old attempt. Do not pass a new config when resuming.
Numerical failures and unexplained stalls require diagnosis before any retry.

Recovery commands (same approved checkout and exported `EXPECTED_REVISION`,
`BASE`, `RUN`, and `PYTHON` as above):

```bash
# List every allocation for this run, including failed preflights; no .batch/.extern rows.
JOB_IDS="COMMA_SEPARATED_PRIOR_JOB_IDS"
sacct -X -n -P -j "$JOB_IDS" --format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode
# First confirm all listed jobs are terminal, each allocated one GPU, and none are missing.
USED_SECONDS=$(sacct -X -n -P -j "$JOB_IDS" --format=ElapsedRaw | \
    awk -F'|' '{sum += $1} END {printf "%.0f", sum}')
REMAINING_SECONDS=$((86400 - USED_SECONDS))
test "$REMAINING_SECONDS" -gt 0
REMAINING=$(printf '%02d:%02d:%02d' "$((REMAINING_SECONDS / 3600))" \
    "$((REMAINING_SECONDS / 60 % 60))" "$((REMAINING_SECONDS % 60))")
CHECKPOINT="ABSOLUTE_PATH_TO_PRIOR_ATTEMPT/training/latest.pt"
"$PYTHON" -c 'import sys; from growing_transformer.train import restore_checkpoint; print(restore_checkpoint(sys.argv[1])[3]["step"])' "$CHECKPOINT"
# Only for an authorized recovery reason, with fewer than two recoveries so far:
sbatch --parsable --time="$REMAINING" --output="$RUN/slurm-%j.out" \
    --error="$RUN/slurm-%j.err" scripts/tue-hpc.sbatch "$CHECKPOINT"
```

For interrupted **final evaluation only**, require the checkpoint step to be
100,000 and replace the last command (do not execute both) with:

```bash
sbatch --parsable --time="$REMAINING" --output="$RUN/slurm-%j.out" \
    --error="$RUN/slurm-%j.err" scripts/tue-hpc.sbatch --eval-only "$CHECKPOINT"
```

This still runs the GPU preflight, writes to a fresh job directory, and counts
toward the same recovery-count and GPU-hour limits. Checkpoint loading and all
preflight gates must pass before continuation. The script rejects a checkpoint
whose saved training configuration differs from `configs/gpu.json` (except paths).

## Archive before scratch expiry

Scratch expires after 14 days. Cleanup guidance requires keeping large/generated
outputs out of home, so do not automatically copy results there. At run completion,
report the run-directory size and earliest scratch expiry, identify available
durable project storage, and request approval of the archive destination. If no
durable destination is available, report that as an outstanding retention blocker;
do not describe scratch-only results as durably archived.

Preserve the entire run directory (metrics, all retained checkpoints, logs,
resolved environment and validation) until transfer. Exclude incomplete `*.tmp`
files from the archive and verify it with checksums before deleting any source.
Do not archive the venv, pip cache, or dataset. Report final architecture,
achieved step/token count, validation BPC, allocated GPU-hours, job IDs, artifact
path and retention status, without declaring scientific success or failure.
