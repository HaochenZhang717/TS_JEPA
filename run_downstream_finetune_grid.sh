#!/usr/bin/env bash
set -euo pipefail

# Downstream finetune/eval script aligned with pretrain.sh.
# For each pretrained parameter setting in pretrain.sh (LR x EMA), run exactly one
# downstream finetune/eval with fixed downstream hyperparameters.
#
# Usage:
#   bash run_downstream_finetune_grid.sh
# Optional overrides:
#   DATA=weather DOWNSTREAM_LR=1e-04 CHECKPOINT_TO_USE=5000 \
#   LRS="5e-5 1e-5 1e-6" EMA_DECAYS="0.995 0.998 0.999" \
#   bash run_downstream_finetune_grid.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/logs/downstream_from_pretrain_${TS}"
mkdir -p "${OUT_DIR}"

# Keep these defaults consistent with pretrain.sh.
DATA="${DATA:-weather}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RATIO_PATCHES="${RATIO_PATCHES:-10}"
MASK_RATIO="${MASK_RATIO:-0.7}"
ENC_DIM="${ENC_DIM:-128}"
ENC_HEAD="${ENC_HEAD:-2}"
ENC_LAYER="${ENC_LAYER:-1}"
DEC_DIM="${DEC_DIM:-128}"
DEC_HEAD="${DEC_HEAD:-2}"
DEC_LAYER="${DEC_LAYER:-1}"
PRETRAIN_ENCODER_KERNEL_SIZE="${PRETRAIN_ENCODER_KERNEL_SIZE:-3}"

# Fixed downstream finetune params (no tuning).
DOWNSTREAM_LR="${DOWNSTREAM_LR:-1e-04}"
CHECKPOINT_TO_USE="${CHECKPOINT_TO_USE:-5000}"

# Pretrain sweep settings from pretrain.sh.
LRS_STR="${LRS:-5e-5 1e-5 1e-6}"
EMA_DECAYS_STR="${EMA_DECAYS:-0.995 0.998 0.999}"
read -r -a LRS <<< "${LRS_STR}"
read -r -a EMA_DECAYS <<< "${EMA_DECAYS_STR}"

SUMMARY_CSV="${OUT_DIR}/summary.csv"
printf "run_id,status,data,batch_size,lr,lr_pretrain,mask_ratio,ema_pretrain,ratio_patches,checkpoint_to_use,enc_dim,enc_head,enc_layer,enc_kernel,dec_dim,dec_head,dec_layer,mse,mae,log_file\n" > "${SUMMARY_CSV}"

run_id=0
for lr_pretrain in "${LRS[@]}"; do
  for ema_pretrain in "${EMA_DECAYS[@]}"; do
    run_id=$((run_id + 1))
    run_tag="run$(printf '%04d' "${run_id}")_prelr_${lr_pretrain}_ema_${ema_pretrain}"
    log_file="${OUT_DIR}/${run_tag}.log"

    echo "======================================================"
    echo "[${run_tag}] Running downstream finetune/eval"
    echo "pretrain lr=${lr_pretrain}, pretrain ema=${ema_pretrain}"
    echo "downstream lr=${DOWNSTREAM_LR}, checkpoint=${CHECKPOINT_TO_USE}"
    echo "======================================================"

    set +e
    python "${ROOT_DIR}/eval_forecast_last_pred.py" \
      --data "${DATA}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${DOWNSTREAM_LR}" \
      --lr_pretrain "${lr_pretrain}" \
      --mask_ratio "${MASK_RATIO}" \
      --ema_pretrain "${ema_pretrain}" \
      --ratio_patches "${RATIO_PATCHES}" \
      --checkpoint_to_use "${CHECKPOINT_TO_USE}" \
      --pretrain_encoder_embed_dim "${ENC_DIM}" \
      --pretrain_encoder_nhead "${ENC_HEAD}" \
      --pretrain_encoder_num_layers "${ENC_LAYER}" \
      --pretrain_encoder_kernel_size "${PRETRAIN_ENCODER_KERNEL_SIZE}" \
      --pretrain_decoder_embed_dim "${DEC_DIM}" \
      --pretrain_decoder_nhead "${DEC_HEAD}" \
      --pretrain_decoder_num_layers "${DEC_LAYER}" \
      > "${log_file}" 2>&1
    status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
      echo "[WARN] Failed run: ${run_tag}, check ${log_file}"
      printf "%s,FAILED,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,%s\n" \
        "${run_tag}" "${DATA}" "${BATCH_SIZE}" "${DOWNSTREAM_LR}" "${lr_pretrain}" "${MASK_RATIO}" "${ema_pretrain}" \
        "${RATIO_PATCHES}" "${CHECKPOINT_TO_USE}" "${ENC_DIM}" "${ENC_HEAD}" "${ENC_LAYER}" \
        "${PRETRAIN_ENCODER_KERNEL_SIZE}" "${DEC_DIM}" "${DEC_HEAD}" "${DEC_LAYER}" "${log_file}" >> "${SUMMARY_CSV}"
      continue
    fi

    mse="$(grep -E 'MSE Loss is:' "${log_file}" | tail -n1 | awk -F': ' '{print $2}' | tr -d '[:space:]')"
    mae="$(grep -E 'MAE Loss is:' "${log_file}" | tail -n1 | awk -F': ' '{print $2}' | tr -d '[:space:]')"

    mse="${mse:-NA}"
    mae="${mae:-NA}"

    printf "%s,OK,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${run_tag}" "${DATA}" "${BATCH_SIZE}" "${DOWNSTREAM_LR}" "${lr_pretrain}" "${MASK_RATIO}" "${ema_pretrain}" \
      "${RATIO_PATCHES}" "${CHECKPOINT_TO_USE}" "${ENC_DIM}" "${ENC_HEAD}" "${ENC_LAYER}" \
      "${PRETRAIN_ENCODER_KERNEL_SIZE}" "${DEC_DIM}" "${DEC_HEAD}" "${DEC_LAYER}" "${mse}" "${mae}" "${log_file}" >> "${SUMMARY_CSV}"

    echo "[DONE] ${run_tag} mse=${mse}, mae=${mae}"
  done
done

# Create sorted view for quick comparison (best MAE first among successful runs)
SORTED_CSV="${OUT_DIR}/summary_sorted_by_mae.csv"
{
  head -n1 "${SUMMARY_CSV}"
  tail -n +2 "${SUMMARY_CSV}" | awk -F, '$2=="OK"' | sort -t, -k19,19g
  tail -n +2 "${SUMMARY_CSV}" | awk -F, '$2=="FAILED"'
} > "${SORTED_CSV}"

echo
echo "All runs finished."
echo "Summary: ${SUMMARY_CSV}"
echo "Sorted : ${SORTED_CSV}"
