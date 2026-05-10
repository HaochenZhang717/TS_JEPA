#!/usr/bin/env bash
#SBATCH --job-name=dtsjepd_weather
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

mkdir -p /playpen-shared/haochenz/logs/slurm

source ~/.zshrc >/dev/null 2>&1 || true
CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate vlm

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

: "${DATA:=weather}"
: "${INPUT_COLS:=OT}"
: "${BATCH_SIZE:=128}"
: "${NUM_EPOCHS:=5001}"
: "${WARMUP_EPOCHS:=50}"
: "${LR:=1e-5}"
: "${EMA:=0.998}"
: "${RATIO_PATCHES:=10}"
: "${PATCH_SIZE:=32}"
: "${STRIDE:=4}"
: "${MASK_RATIO:=0.3}"
: "${EVAL_EVERY:=1}"
: "${CHECKPOINT_SAVE:=5000}"
: "${CLIP_GRAD:=1.0}"
: "${ENC_EMBED_DIM:=128}"
: "${ENC_NHEAD:=4}"
: "${ENC_NUM_LAYERS:=2}"
: "${PRED_EMBED_DIM:=128}"
: "${PRED_NHEAD:=4}"
: "${PRED_NUM_LAYERS:=2}"
: "${LAMBDA_DENOISE:=0.01}"
: "${DENOISE_HIDDEN_DIM:=64}"
: "${TIME_FREQUENCY_DIM:=64}"
: "${P_MEAN:=0.0}"
: "${P_STD:=1.0}"
: "${T_EPS:=1e-5}"
: "${NOISE_SCALE:=1.0}"
: "${WANDB_PROJECT:=TS_D_JEPA}"
: "${RUN_TAG:=weather_ot_lr${LR}_ema${EMA}_lambda${LAMBDA_DENOISE}_${SLURM_JOB_ID:-manual}}"

echo "============================================================"
echo "DTS-JEPD single-channel Weather Slurm job"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "data=${DATA}, input_cols=${INPUT_COLS}"
echo "lr=${LR}, ema=${EMA}, lambda_denoise=${LAMBDA_DENOISE}"
echo "batch_size=${BATCH_SIZE}, epochs=${NUM_EPOCHS}, warmup=${WARMUP_EPOCHS}"
echo "mask_ratio=${MASK_RATIO}, stride=${STRIDE}, run_tag=${RUN_TAG}"
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
