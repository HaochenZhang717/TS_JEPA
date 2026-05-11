#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Checkpoint discovery settings
CKPT_ROOT="/mnt/unites8/playpen/haochenz/TS_JEPA/logs/output_model_dtsjepd/weather"
CKPT_GLOB="*_epoch_5000.pt"

# Shared eval settings (debug defaults)
DATA="weather"
INPUT_COLS="OT"
DATA_PATH="./data/${DATA}/${DATA}.csv"

BATCH_SIZE="64"
NUM_EPOCHS="5"
EVAL_LR="1e-3"
WEIGHT_DECAY="1e-4"
STRIDE="4"
NUM_WORKERS="0"
DECODER="mlp"
DECODER_HIDDEN_DIM="256"
METRIC_ORIG_SCALE="0"   # 0: normalized scale, 1: original scale
SEED="42"
MAX_JOBS="0"             # 0 means run all discovered checkpoints

source ~/.zshrc >/dev/null 2>&1 || true
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type f -name "${CKPT_GLOB}" | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints found under ${CKPT_ROOT} with glob ${CKPT_GLOB}" >&2
  exit 1
fi

ran=0
for CHECKPOINT in "${CKPTS[@]}"; do
  if [[ "${MAX_JOBS}" != "0" && ${ran} -ge ${MAX_JOBS} ]]; then
    break
  fi

  echo "============================================================"
  echo "Local debug eval"
  echo "checkpoint=${CHECKPOINT}"
  echo "decoder=${DECODER}, hidden=${DECODER_HIDDEN_DIM}"
  echo "batch_size=${BATCH_SIZE}, num_epochs=${NUM_EPOCHS}"
  echo "metric_orig_scale=${METRIC_ORIG_SCALE}"
  echo "============================================================"

  CMD=(
    python -m dtsjepd.prediction_eval
    --checkpoint "${CHECKPOINT}"
    --data "${DATA}"
    --data_path "${DATA_PATH}"
    --input_cols "${INPUT_COLS}"
    --batch_size "${BATCH_SIZE}"
    --num_epochs "${NUM_EPOCHS}"
    --lr "${EVAL_LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --stride "${STRIDE}"
    --num_workers "${NUM_WORKERS}"
    --decoder "${DECODER}"
    --decoder_hidden_dim "${DECODER_HIDDEN_DIM}"
    --seed "${SEED}"
  )

  if [[ "${METRIC_ORIG_SCALE}" == "1" ]]; then
    CMD+=(--metric_on_original_scale)
  fi

  "${CMD[@]}"
  ran=$((ran + 1))
done

echo "Finished local eval runs: ${ran}"
