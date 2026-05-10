#!/usr/bin/env bash
set -euo pipefail

# Sweep settings
LRS=("5e-5" "1e-5" "1e-6")
EMA_DECAYS=("0.995" "0.998" "0.999")
LAMBDA_DENOISES=("0.01" "0.1")

# Shared training settings
DATA="weather"
INPUT_COLS="OT"
BATCH_SIZE="32"
NUM_EPOCHS="5001"
RATIO_PATCHES="10"
PATCH_SIZE="32"
MASK_RATIO="0.7"
EVAL_EVERY="10"
CHECKPOINT_SAVE="5000"

ENC_EMBED_DIM="128"
ENC_NHEAD="2"
ENC_NUM_LAYERS="1"
PRED_EMBED_DIM="128"
PRED_NHEAD="2"
PRED_NUM_LAYERS="1"

DENOISE_HIDDEN_DIM="128"
TIME_FREQUENCY_DIM="128"
P_MEAN="0.0"
P_STD="1.0"
T_EPS="1e-5"
NOISE_SCALE="1.0"

WANDB_PROJECT="TS_D_JEPA"

for LR in "${LRS[@]}"; do
  for EMA in "${EMA_DECAYS[@]}"; do
    for LAMBDA_DENOISE in "${LAMBDA_DENOISES[@]}"; do
      RUN_TAG="weather_ot_lr${LR}_ema${EMA}_lambda${LAMBDA_DENOISE}_$(date +%Y%m%d_%H%M%S)"
      echo "============================================================"
      echo "Starting DTS-JEPD run: data=${DATA}, input_cols=${INPUT_COLS}, lr=${LR}, ema=${EMA}, lambda=${LAMBDA_DENOISE}, tag=${RUN_TAG}"
      echo "============================================================"

      python -m dtsjepd.train \
        --data "${DATA}" \
        --input_cols "${INPUT_COLS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --lr "${LR}" \
        --ema_momentum "${EMA}" \
        --ratio_patches "${RATIO_PATCHES}" \
        --patch_size "${PATCH_SIZE}" \
        --mask_ratio "${MASK_RATIO}" \
        --eval_every "${EVAL_EVERY}" \
        --checkpoint_save "${CHECKPOINT_SAVE}" \
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
done
