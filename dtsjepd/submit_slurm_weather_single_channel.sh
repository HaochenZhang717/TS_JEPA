#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_weather_single_channel.sh"

# Sweep settings
LRS=("1e-4" "1e-5" "1e-6")
EMA_DECAYS=("0.998" "0.999" "0.9995")
LAMBDA_DENOISES=("0.01" "0.1" "0")


LRS=("1e-5")
EMA_DECAYS=("0.999")
LAMBDA_DENOISES=("0.01")

# Shared training settings
DATA="weather"
INPUT_COLS="OT"
BATCH_SIZE="128"
NUM_EPOCHS="5001"
WARMUP_EPOCHS="50"
RATIO_PATCHES="10"
PATCH_SIZE="32"
STRIDE="4"
MASK_RATIO="0.3"
EVAL_EVERY="1"
CHECKPOINT_SAVE="5000"
CLIP_GRAD="1.0"

ENC_EMBED_DIM="128"
ENC_NHEAD="4"
ENC_NUM_LAYERS="2"
PRED_EMBED_DIM="128"
PRED_NHEAD="4"
PRED_NUM_LAYERS="2"

DENOISE_HIDDEN_DIM="64"
TIME_FREQUENCY_DIM="64"
P_MEAN="0.0"
P_STD="1.0"
T_EPS="1e-5"
NOISE_SCALE="1.0"

WANDB_PROJECT="TS_D_JEPA"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Cannot find Slurm script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p /playpen-shared/haochenz/logs/slurm

submitted=0
for LR in "${LRS[@]}"; do
  for EMA in "${EMA_DECAYS[@]}"; do
    for LAMBDA_DENOISE in "${LAMBDA_DENOISES[@]}"; do
      RUN_TAG="weather_ot_lr${LR}_ema${EMA}_lambda${LAMBDA_DENOISE}_$(date +%Y%m%d_%H%M%S)"
      JOB_NAME="dtsjepd_w_lr${LR}_e${EMA}_l${LAMBDA_DENOISE}"

      echo "Submitting ${JOB_NAME} (${RUN_TAG})"
      sbatch \
        --job-name="${JOB_NAME}" \
        --export=ALL,DATA="${DATA}",INPUT_COLS="${INPUT_COLS}",BATCH_SIZE="${BATCH_SIZE}",NUM_EPOCHS="${NUM_EPOCHS}",WARMUP_EPOCHS="${WARMUP_EPOCHS}",LR="${LR}",EMA="${EMA}",RATIO_PATCHES="${RATIO_PATCHES}",PATCH_SIZE="${PATCH_SIZE}",STRIDE="${STRIDE}",MASK_RATIO="${MASK_RATIO}",EVAL_EVERY="${EVAL_EVERY}",CHECKPOINT_SAVE="${CHECKPOINT_SAVE}",CLIP_GRAD="${CLIP_GRAD}",ENC_EMBED_DIM="${ENC_EMBED_DIM}",ENC_NHEAD="${ENC_NHEAD}",ENC_NUM_LAYERS="${ENC_NUM_LAYERS}",PRED_EMBED_DIM="${PRED_EMBED_DIM}",PRED_NHEAD="${PRED_NHEAD}",PRED_NUM_LAYERS="${PRED_NUM_LAYERS}",LAMBDA_DENOISE="${LAMBDA_DENOISE}",DENOISE_HIDDEN_DIM="${DENOISE_HIDDEN_DIM}",TIME_FREQUENCY_DIM="${TIME_FREQUENCY_DIM}",P_MEAN="${P_MEAN}",P_STD="${P_STD}",T_EPS="${T_EPS}",NOISE_SCALE="${NOISE_SCALE}",WANDB_PROJECT="${WANDB_PROJECT}",RUN_TAG="${RUN_TAG}" \
        "${SLURM_SCRIPT}"

      submitted=$((submitted + 1))
      sleep 1
    done
  done
done

echo "Submitted ${submitted} Slurm jobs."
