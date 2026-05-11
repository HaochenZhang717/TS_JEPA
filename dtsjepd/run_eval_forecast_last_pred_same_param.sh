#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Checkpoints produced by dtsjepd/pretrain_same_param.sh.
CKPT_ROOT="${PROJECT_ROOT}/logs/output_model_dtsjepd_same_param/weather"
CKPT_GLOB="dtsjepd_lr_*_epoch_5000.pt"

# Match the original downstream protocol as closely as possible, while using
# dtsjepd's train-only normalization through dtsjepd.eval_forecast_last_pred.
DATA="weather"
INPUT_COLS="OT"
DATA_PATH="./data/${DATA}/${DATA}.csv"

BATCH_SIZE="32"
NUM_EPOCHS="500"
LR="1e-4"
WEIGHT_DECAY="0"
NUM_WORKERS="0"
SEED="42"
MAX_JOBS="0"      # 0 means run all discovered checkpoints
SAVE_JSON_DIR="./logs/eval_forecast_last_pred/same_param"

source ~/.zshrc >/dev/null 2>&1 || true
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
if [[ -x "${CONDA_BIN}" ]]; then
  eval "$("${CONDA_BIN}" shell.bash hook)"
  conda activate vlm
fi

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
  echo "Same-param eval_forecast_last_pred"
  echo "checkpoint=${CHECKPOINT}"
  echo "batch_size=${BATCH_SIZE}, num_epochs=${NUM_EPOCHS}, lr=${LR}"
  echo "save_json_dir=${SAVE_JSON_DIR}"
  echo "============================================================"

  python -m dtsjepd.eval_forecast_last_pred \
    --checkpoint "${CHECKPOINT}" \
    --data "${DATA}" \
    --data_path "${DATA_PATH}" \
    --input_cols "${INPUT_COLS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save_json_dir "${SAVE_JSON_DIR}"

  ran=$((ran + 1))
done

echo "Finished same-param eval_forecast_last_pred runs: ${ran}"
