#!/usr/bin/env bash
set -euo pipefail

# DTS-JEPD lambda=0 baseline with parameters matched to the original pretrain.sh.
# This is meant to test whether the DTS-JEPD training path reproduces the
# original TS-JEPA setting when the denoising objective is disabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

#LRS=("5e-5" "1e-5" "1e-6")
#EMA_DECAYS=("0.995" "0.998" "0.999")

LRS=("1e-5")
EMA_DECAYS=("0.995")

DATA="weather"
INPUT_COLS="OT"
OUTPUT_DIR="./logs/output_model_dtsjepd_same_param"
BATCH_SIZE="32"
NUM_EPOCHS="5001"
WARMUP_EPOCHS="0"
RATIO_PATCHES="10"
PATCH_SIZE="32"
STRIDE="0"
MASK_RATIO="0.7"
EVAL_EVERY="10"
CHECKPOINT_SAVE="5000"
CLIP_GRAD="10"

ENC_EMBED_DIM="128"
ENC_NHEAD="2"
ENC_NUM_LAYERS="1"
PRED_EMBED_DIM="128"
PRED_NHEAD="2"
PRED_NUM_LAYERS="1"

LAMBDA_DENOISE="0"
DENOISE_HIDDEN_DIM="128"
TIME_FREQUENCY_DIM="128"
P_MEAN="0.0"
P_STD="1.0"
T_EPS="1e-5"
NOISE_SCALE="1.0"


WANDB_PROJECT="TS_D_JEPA"

for LR in "${LRS[@]}"; do
  for EMA in "${EMA_DECAYS[@]}"; do
    RUN_TAG="same_param_lambda0_lr${LR}_ema${EMA}_$(date +%Y%m%d_%H%M%S)"
    echo "============================================================"
    echo "Starting DTS-JEPD same-param lambda=0 run"
    echo "lr=${LR}, ema_momentum=${EMA}, tag=${RUN_TAG}"
    echo "============================================================"

    python -m dtsjepd.train \
      --data "${DATA}" \
      --input_cols "${INPUT_COLS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_epochs "${NUM_EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --lr "${LR}" \
      --ema_momentum "${EMA}" \
      --ratio_patches "${RATIO_PATCHES}" \
      --patch_size "${PATCH_SIZE}" \
      --stride "${STRIDE}" \
      --mask_ratio "${MASK_RATIO}" \
      --eval_every "${EVAL_EVERY}" \
      --checkpoint_save "${CHECKPOINT_SAVE}" \
      --output_dir "${OUTPUT_DIR}" \
      --clip_grad "${CLIP_GRAD}" \
      --encoder_embed_dim "${ENC_EMBED_DIM}" \
      --encoder_nhead "${ENC_NHEAD}" \
      --encoder_num_layers "${ENC_NUM_LAYERS}" \
      --predictor_embed "${PRED_EMBED_DIM}" \
      --predictor_nhead "${PRED_NHEAD}" \
      --predictor_num_layers "${PRED_NUM_LAYERS}" \
      --lambda_denoise "${LAMBDA_DENOISE}" \
      --denoise_hidden_dim "${DENOISE_HIDDEN_DIM}" \
      --time_frequency_dim "${TIME_FREQUENCY_DIM}" \
      --P_mean "${P_MEAN}" \
      --P_std "${P_STD}" \
      --t_eps "${T_EPS}" \
      --noise_scale "${NOISE_SCALE}" \
      --save_suffix "${RUN_TAG}" \
      --log_wandb \
      --wandb_project_name "${WANDB_PROJECT}"
  done
done
