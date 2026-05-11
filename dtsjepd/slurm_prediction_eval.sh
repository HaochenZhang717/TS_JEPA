#!/usr/bin/env bash
#SBATCH --job-name=dtsjepd_eval
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

: "${CHECKPOINT:?CHECKPOINT must be provided by sbatch --export}"
: "${DATA:=weather}"
: "${DATA_PATH:=./data/${DATA}/${DATA}.csv}"
: "${INPUT_COLS:=OT}"
: "${BATCH_SIZE:=256}"
: "${NUM_EPOCHS:=50}"
: "${EVAL_LR:=1e-3}"
: "${WEIGHT_DECAY:=1e-4}"
: "${STRIDE:=4}"
: "${NUM_WORKERS:=0}"
: "${DECODER:=mlp}"
: "${DECODER_HIDDEN_DIM:=256}"
: "${METRIC_ORIG_SCALE:=1}"
: "${SEED:=42}"
: "${RUN_TAG:=pred_eval_${SLURM_JOB_ID:-manual}}"
: "${SAVE_JSON_DIR:=./logs/eval_prediction/slurm}"

echo "============================================================"
echo "DTS-JEPD prediction eval Slurm job"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "checkpoint=${CHECKPOINT}"
echo "data=${DATA}, input_cols=${INPUT_COLS}"
echo "decoder=${DECODER}, hidden=${DECODER_HIDDEN_DIM}"
echo "eval_lr=${EVAL_LR}, epochs=${NUM_EPOCHS}, batch_size=${BATCH_SIZE}"
echo "run_tag=${RUN_TAG}"
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

if [[ -n "${SAVE_JSON_DIR}" ]]; then
  CMD+=(--save_json_dir "${SAVE_JSON_DIR}")
fi

"${CMD[@]}"
