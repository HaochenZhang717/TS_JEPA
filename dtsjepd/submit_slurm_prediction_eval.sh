#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_prediction_eval.sh"

# Checkpoint discovery settings
CKPT_ROOT="/mnt/unites8/playpen/haochenz/TS_JEPA/logs/output_model_dtsjepd/weather"
CKPT_GLOB="*_epoch_5000.pt"

# Shared eval settings
DATA="weather"
INPUT_COLS="OT"
DATA_PATH="./data/${DATA}/${DATA}.csv"

BATCH_SIZE="256"
NUM_EPOCHS="100"
EVAL_LR="1e-3"
WEIGHT_DECAY="1e-4"
STRIDE="4"
NUM_WORKERS="0"
DECODER="cnn_mlp"
DECODER_HIDDEN_DIM="64"
METRIC_ORIG_SCALE="0"
SEED="42"
SAVE_JSON_DIR="./logs/eval_prediction/slurm"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Cannot find Slurm script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_ROOT}" >&2
  exit 1
fi

mkdir -p /playpen-shared/haochenz/logs/slurm

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type f -name "${CKPT_GLOB}" | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints found under ${CKPT_ROOT} with glob ${CKPT_GLOB}" >&2
  exit 1
fi

submitted=0
for CHECKPOINT in "${CKPTS[@]}"; do
  base="$(basename "${CHECKPOINT}" .pt)"
  short="${base#dtsjepd_}"
  JOB_NAME="dtsjepd_eval"
  RUN_TAG="pred_eval_${short}_$(date +%Y%m%d_%H%M%S)"

  echo "Submitting ${JOB_NAME}"
  echo "  checkpoint=${CHECKPOINT}"

  sbatch \
    --job-name="${JOB_NAME}" \
    --export=ALL,CHECKPOINT="${CHECKPOINT}",DATA="${DATA}",DATA_PATH="${DATA_PATH}",INPUT_COLS="${INPUT_COLS}",BATCH_SIZE="${BATCH_SIZE}",NUM_EPOCHS="${NUM_EPOCHS}",EVAL_LR="${EVAL_LR}",WEIGHT_DECAY="${WEIGHT_DECAY}",STRIDE="${STRIDE}",NUM_WORKERS="${NUM_WORKERS}",DECODER="${DECODER}",DECODER_HIDDEN_DIM="${DECODER_HIDDEN_DIM}",METRIC_ORIG_SCALE="${METRIC_ORIG_SCALE}",SEED="${SEED}",RUN_TAG="${RUN_TAG}",SAVE_JSON_DIR="${SAVE_JSON_DIR}" \
    "${SLURM_SCRIPT}"

  submitted=$((submitted + 1))
  sleep 1
done

echo "Submitted ${submitted} eval Slurm jobs."
