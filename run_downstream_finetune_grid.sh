#!/usr/bin/env bash
set -euo pipefail

# Downstream finetune/eval script based on existing pretrained checkpoints.
# It scans logs/output_model/${DATA} and runs exactly one downstream eval for each
# checkpoint that matches epoch_${CHECKPOINT_TO_USE}.pt.
#
# Usage:
#   bash run_downstream_finetune_grid.sh
# Optional overrides:
#   DATA=weather DOWNSTREAM_LR=1e-04 CHECKPOINT_TO_USE=5000 bash run_downstream_finetune_grid.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/logs/downstream_from_pretrain_${TS}"
PLOT_DIR="${OUT_DIR}/plots"
mkdir -p "${OUT_DIR}"
mkdir -p "${PLOT_DIR}"

DATA="${DATA:-weather}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_ENCODER_KERNEL_SIZE="${PRETRAIN_ENCODER_KERNEL_SIZE:-3}"

# Fixed downstream finetune params (no tuning).
DOWNSTREAM_LR="${DOWNSTREAM_LR:-1e-04}"
CHECKPOINT_TO_USE="${CHECKPOINT_TO_USE:-5000}"
PLOT_NUM_STEPS="${PLOT_NUM_STEPS:-20}"

CKPT_DIR="${ROOT_DIR}/logs/output_model/${DATA}"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

SUMMARY_CSV="${OUT_DIR}/summary.csv"
printf "run_id,status,data,batch_size,lr,lr_pretrain,mask_ratio,ema_pretrain,ratio_patches,checkpoint_to_use,enc_dim,enc_head,enc_layer,enc_kernel,dec_dim,dec_head,dec_layer,checkpoint_file,mse,mae,log_file,plot_file\n" > "${SUMMARY_CSV}"

shopt -s nullglob
CKPT_FILES=("${CKPT_DIR}"/*"_epoch_${CHECKPOINT_TO_USE}.pt")
shopt -u nullglob

if [[ ${#CKPT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoint files found for epoch_${CHECKPOINT_TO_USE} in ${CKPT_DIR}" >&2
  exit 1
fi

run_id=0
for ckpt_path in "${CKPT_FILES[@]}"; do
  ckpt_file="$(basename "${ckpt_path}")"
  ckpt_stem="${ckpt_file%.pt}"

  # Expected format (with optional save_suffix section before _epoch):
  # lr_<lr>_ema_momentum_<ema>_mask_ratio_<mask>_ratio_patches_<rp>_encoder_<e1>_<e2>_<e3>_predictor_<p1>_<p2>_<p3>[_anything]_epoch_<ep>
  if [[ "${ckpt_stem}" =~ ^lr_(.+)_ema_momentum_(.+)_mask_ratio_(.+)_ratio_patches_([0-9]+)_encoder_([0-9]+)_([0-9]+)_([0-9]+)_predictor_([0-9]+)_([0-9]+)_([0-9]+)(_.+)?_epoch_([0-9]+)$ ]]; then
    lr_pretrain="${BASH_REMATCH[1]}"
    ema_pretrain="${BASH_REMATCH[2]}"
    mask_ratio="${BASH_REMATCH[3]}"
    ratio_patches="${BASH_REMATCH[4]}"
    enc_dim="${BASH_REMATCH[5]}"
    enc_head="${BASH_REMATCH[6]}"
    enc_layer="${BASH_REMATCH[7]}"
    dec_dim="${BASH_REMATCH[8]}"
    dec_head="${BASH_REMATCH[9]}"
    dec_layer="${BASH_REMATCH[10]}"
    checkpoint_to_use="${BASH_REMATCH[12]}"
  else
    echo "[WARN] Skip unrecognized checkpoint name: ${ckpt_file}"
    continue
  fi

    run_id=$((run_id + 1))
    run_tag="run$(printf '%04d' "${run_id}")_prelr_${lr_pretrain}_ema_${ema_pretrain}_ep_${checkpoint_to_use}"
    log_file="${OUT_DIR}/${run_tag}.log"
    plot_file="${PLOT_DIR}/${run_tag}.png"

    echo "======================================================"
    echo "[${run_tag}] Running downstream finetune/eval"
    echo "pretrain lr=${lr_pretrain}, pretrain ema=${ema_pretrain}"
    echo "downstream lr=${DOWNSTREAM_LR}, checkpoint=${checkpoint_to_use}"
    echo "source checkpoint: ${ckpt_file}"
    echo "======================================================"

    set +e
    python "${ROOT_DIR}/eval_forecast_last_pred.py" \
      --data "${DATA}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${DOWNSTREAM_LR}" \
      --checkpoint_path "${ckpt_path}" \
      --plot_path "${plot_file}" \
      --plot_num_steps "${PLOT_NUM_STEPS}" \
      --lr_pretrain "${lr_pretrain}" \
      --mask_ratio "${mask_ratio}" \
      --ema_pretrain "${ema_pretrain}" \
      --ratio_patches "${ratio_patches}" \
      --checkpoint_to_use "${checkpoint_to_use}" \
      --pretrain_encoder_embed_dim "${enc_dim}" \
      --pretrain_encoder_nhead "${enc_head}" \
      --pretrain_encoder_num_layers "${enc_layer}" \
      --pretrain_encoder_kernel_size "${PRETRAIN_ENCODER_KERNEL_SIZE}" \
      --pretrain_decoder_embed_dim "${dec_dim}" \
      --pretrain_decoder_nhead "${dec_head}" \
      --pretrain_decoder_num_layers "${dec_layer}" \
      > "${log_file}" 2>&1
    status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
      echo "[WARN] Failed run: ${run_tag}, check ${log_file}"
      printf "%s,FAILED,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NA,NA,%s,%s\n" \
        "${run_tag}" "${DATA}" "${BATCH_SIZE}" "${DOWNSTREAM_LR}" "${lr_pretrain}" "${mask_ratio}" "${ema_pretrain}" \
        "${ratio_patches}" "${checkpoint_to_use}" "${enc_dim}" "${enc_head}" "${enc_layer}" \
        "${PRETRAIN_ENCODER_KERNEL_SIZE}" "${dec_dim}" "${dec_head}" "${dec_layer}" "${ckpt_file}" "${log_file}" "${plot_file}" >> "${SUMMARY_CSV}"
      continue
    fi

    mse="$(grep -E 'MSE Loss is:' "${log_file}" | tail -n1 | awk -F': ' '{print $2}' | tr -d '[:space:]')"
    mae="$(grep -E 'MAE Loss is:' "${log_file}" | tail -n1 | awk -F': ' '{print $2}' | tr -d '[:space:]')"

    mse="${mse:-NA}"
    mae="${mae:-NA}"

    printf "%s,OK,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${run_tag}" "${DATA}" "${BATCH_SIZE}" "${DOWNSTREAM_LR}" "${lr_pretrain}" "${mask_ratio}" "${ema_pretrain}" \
      "${ratio_patches}" "${checkpoint_to_use}" "${enc_dim}" "${enc_head}" "${enc_layer}" \
      "${PRETRAIN_ENCODER_KERNEL_SIZE}" "${dec_dim}" "${dec_head}" "${dec_layer}" "${ckpt_file}" "${mse}" "${mae}" "${log_file}" "${plot_file}" >> "${SUMMARY_CSV}"

    echo "[DONE] ${run_tag} mse=${mse}, mae=${mae}"
done

# Create sorted view for quick comparison (best MAE first among successful runs)
SORTED_CSV="${OUT_DIR}/summary_sorted_by_mae.csv"
{
  head -n1 "${SUMMARY_CSV}"
  tail -n +2 "${SUMMARY_CSV}" | awk -F, '$2=="OK"' | sort -t, -k20,20g
  tail -n +2 "${SUMMARY_CSV}" | awk -F, '$2=="FAILED"'
} > "${SORTED_CSV}"

echo
echo "All runs finished."
echo "Summary: ${SUMMARY_CSV}"
echo "Sorted : ${SORTED_CSV}"
