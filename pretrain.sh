#!/usr/bin/env bash
set -euo pipefail

# Sweep settings
LRS=("5e-5" "1e-5" "1e-6")
#EMA_DECAYS=("0.995" "0.998" "0.999")
#LRS=("1e-7")
EMA_DECAYS=("0.998")


# Shared training settings
DATA="weather"
BATCH_SIZE="32"
RATIO_PATCHES="10"
MASK_RATIO="0.7"
ENC_EMBED_DIM="128"
ENC_NHEAD="2"
ENC_NUM_LAYERS="1"
PRED_EMBED_DIM="128"
PRED_NHEAD="2"
PRED_NUM_LAYERS="1"
WANDB_PROJECT="TS_JEPA"

for LR in "${LRS[@]}"; do
  for EMA in "${EMA_DECAYS[@]}"; do
    RUN_TAG="lr${LR}_ema${EMA}_$(date +%Y%m%d_%H%M%S)"
    echo "============================================================"
    echo "Starting run with lr=${LR}, ema_momentum=${EMA}, tag=${RUN_TAG}"
    echo "============================================================"

    python pretrain.py \
      --data "${DATA}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --ema_momentum "${EMA}" \
      --ratio_patches "${RATIO_PATCHES}" \
      --mask_ratio "${MASK_RATIO}" \
      --encoder_embed_dim "${ENC_EMBED_DIM}" \
      --encoder_nhead "${ENC_NHEAD}" \
      --encoder_num_layers "${ENC_NUM_LAYERS}" \
      --predictor_embed "${PRED_EMBED_DIM}" \
      --predictor_nhead "${PRED_NHEAD}" \
      --predictor_num_layers "${PRED_NUM_LAYERS}" \
      --save_suffix "${RUN_TAG}" \
      --log_wandb \
      --wandb_project_name "${WANDB_PROJECT}"
  done
done
