#!/bin/bash
#SBATCH --job-name=growing-text8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue

# Submit from the exact approved checkout, with --output and --error on scratch.
# Optional first argument: absolute checkpoint path for continuation in a fresh directory.
# Or: --eval-only <final-checkpoint>, for interrupted final validation.
set -euo pipefail
case "$#" in
    0) ;;
    1) test -f "$1" ;;
    2) test "$1" = --eval-only; test -f "$2" ;;
    *) echo "Usage: sbatch script [checkpoint | --eval-only checkpoint]" >&2; exit 2 ;;
esac
module load Umbrella/2024
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset PYTHONPATH

BASE=/scratch-shared/20194474/growingtransformer
PYTHON="$BASE/venv-torch260-cu118/bin/python"
REVISION=$(git rev-parse HEAD)
: "${EXPECTED_REVISION:?Set EXPECTED_REVISION to the approved full commit ID}"
test "$REVISION" = "$EXPECTED_REVISION"
test -z "$(git status --porcelain --untracked-files=no)"
RUN="$BASE/$REVISION/adaptive-seed42"
ATTEMPT="$RUN/job-$SLURM_JOB_ID"
mkdir -p "$RUN"
mkdir "$ATTEMPT"
exec > >(tee -a "$ATTEMPT/console.log") 2>&1

"$PYTHON" -m pip freeze > "$ATTEMPT/environment.txt"
nvidia-smi > "$ATTEMPT/nvidia-smi.txt"
"$PYTHON" - <<'PY'
import torch
assert torch.__version__ == "2.6.0+cu118", torch.__version__
assert torch.cuda.is_available(), "CUDA is required; CPU fallback is forbidden"
assert torch.cuda.device_count() == 1, "Request exactly one visible GPU"
assert "V100" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print("CUDA environment:", torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY

# Includes CUDA expansion/resume and three full-size steps at the 8-layer/8-head cap.
timeout 30m "$PYTHON" -m pytest -q -s --junitxml="$ATTEMPT/preflight.xml" \
    > "$ATTEMPT/preflight.log" 2>&1
tail -n 3 "$ATTEMPT/preflight.log"
"$PYTHON" - "$ATTEMPT/preflight.xml" <<'PY'
import sys
from xml.etree import ElementTree
suites = ElementTree.parse(sys.argv[1]).getroot().findall(".//testsuite")
assert suites and all(int(s.attrib["skipped"]) == 0 for s in suites), "Skipped preflight tests"
PY

if [ "$#" -gt 0 ]; then
    SOURCE="${!#}"
    "$PYTHON" - "$SOURCE" <<'PY'
from dataclasses import asdict
import json
import sys
import torch
from growing_transformer.train import TrainConfig
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
with open("configs/gpu.json") as stream:
    expected = asdict(TrainConfig(**json.load(stream)))
actual = asdict(TrainConfig(**state["config"]))
for key in ("data", "output"):
    del expected[key], actual[key]
assert actual == expected, "Checkpoint does not belong to the approved configuration"
assert 0 <= state["step"] <= 100000
PY
fi

if [ "${1:-}" = --eval-only ]; then
    CHECKPOINT="$2"
elif [ "$#" -eq 0 ]; then
    timeout 10m "$PYTHON" -m growing_transformer.train \
        --config configs/growth-smoke.json --device cuda \
        --data "$BASE/data/text8" --output "$ATTEMPT/growth-smoke"
    timeout 10m "$PYTHON" -m growing_transformer.train \
        --resume "$ATTEMPT/growth-smoke/latest.pt" --steps 16 \
        --output "$ATTEMPT/growth-smoke-resumed"
    "$PYTHON" -m growing_transformer.train --config configs/gpu.json \
        --data "$BASE/data/text8" --output "$ATTEMPT/training"
    CHECKPOINT="$ATTEMPT/training/latest.pt"
else
    "$PYTHON" -m growing_transformer.train --resume "$1" --steps 100000 \
        --data "$BASE/data/text8" --output "$ATTEMPT/training"
    CHECKPOINT="$ATTEMPT/training/latest.pt"
fi

"$PYTHON" -m growing_transformer.train --resume "$CHECKPOINT" \
    --eval-only --full-eval > "$ATTEMPT/full-validation.json"
echo "Full validation complete for $CHECKPOINT: $ATTEMPT"
