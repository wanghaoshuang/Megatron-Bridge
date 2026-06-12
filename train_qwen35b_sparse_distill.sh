#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Training script for Qwen3-30B-A3B SparseAttention distillation.
# Student = Qwen3-30B-A3B with FlashMaskAttention, Teacher = original Qwen3-30B-A3B.
workspace=/root/whs
export PYTHONPATH=${workspace}/Megatron-Bridge/src:${workspace}/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH

MODEL_PATH="${workspace}/Qwen/Qwen3-30B-A3B"
CONFIG_FILE="${workspace}/Megatron-Bridge/examples/distillation/qwen3/conf/sparse_distill_qwen3_30b.yaml"

# Use a local lock dir to avoid ~/.cache/huggingface contention across 8 processes
export MEGATRON_CONFIG_LOCK_DIR="${workspace}/Megatron-Bridge/logs"
DATA_PATH="/root/paddlejob/amfp-inference-public/wanghaoshuang/data/megatron_sft_fixed"

LOG_DIR="${workspace}/Megatron-Bridge/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_qwen35b_sparse_distill.log"
rm -f "${LOG_FILE}"

TB_BASE="${workspace}/Megatron-Bridge/nemo_experiments/default/tb_logs/swa_sl8k_sw1k_gs4"
mkdir -p ${TB_BASE}
id=0
while [ -d "${TB_BASE}/exp_${id}" ]; do id=$((id+1)); done
TB_LOG_DIR="${TB_BASE}/exp_${id}"
mkdir -p ${TB_LOG_DIR}

python -m torch.distributed.run --nproc_per_node=8 \
  examples/distillation/qwen3/sparse_distill_qwen3_30b.py \
  --hf_path ${MODEL_PATH} \
  --config_file ${CONFIG_FILE} \
  --data_path ${DATA_PATH} \
  --alpha 1.0 \
  --temperature 1.0 \
  --group_size 4 \
  --segment_size 1024 \
  --pad_token_id 0 \
  logger.tensorboard_dir=${TB_LOG_DIR} \
  2>&1 | tee ${LOG_FILE}
