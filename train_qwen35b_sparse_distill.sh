#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Training script for Qwen3-30B-A3B SparseAttention distillation.
# Student = Qwen3-30B-A3B with FlashMaskAttention, Teacher = original Qwen3-30B-A3B.
workspace=/root/whs
export PYTHONPATH=${workspace}/Megatron-Bridge/src:${workspace}/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH

MODEL_PATH="${workspace}/Qwen/Qwen3-30B-A3B"

LOG_DIR="${workspace}/Megatron-Bridge/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_qwen35b_sparse_distill.log"
rm -f "${LOG_FILE}"

python -m torch.distributed.run --nproc_per_node=8 \
  examples/distillation/qwen3/sparse_distill_qwen3_30b.py \
  --hf_path ${MODEL_PATH} \
  --dataset llm-finetune \
  dataset.dataset_name=longalpaca \
  --alpha 1.0 \
  --temperature 1.0 \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.expert_model_parallel_size=4 \
  model.sequence_parallel=True \
  model.seq_length=2048 \
  train.train_iters=100 \
  validation.eval_iters=2 \
  checkpoint.save=null \
  2>&1 | tee ${LOG_FILE}
