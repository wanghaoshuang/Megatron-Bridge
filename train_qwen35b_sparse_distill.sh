#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Training script for Qwen3-30B-A3B SparseAttention distillation.
# Student = Qwen3-30B-A3B with FlashMaskAttention, Teacher = original Qwen3-30B-A3B.

source /root/paddlejob/workspace/env_run/output/whs/miniconda3/etc/profile.d/conda.sh
conda activate megtron

export PYTHONPATH=/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/src:/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH

MODEL_PATH="/root/paddlejob/workspace/env_run/output/whs/Qwen/Qwen3-30B-A3B"

LOG_DIR="/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_qwen35b_sparse_distill.log"
rm -f "${LOG_FILE}"

python -m torch.distributed.run --nproc_per_node=8 \
  examples/distillation/qwen3/sparse_distill_qwen3_30b.py \
  --hf_path ${MODEL_PATH} \
  --dataset llm-finetune \
  --alpha 1.0 \
  --temperature 1.0 \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.expert_model_parallel_size=4 \
  model.sequence_parallel=True \
  model.seq_length=2048 \
  train.train_iters=10 \
  validation.eval_iters=2 \
  checkpoint.save=null \
  2>&1 | tee ${LOG_FILE}
