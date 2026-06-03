#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Training script for Qwen3-30B-A3B (text LLM) SFT fine-tuning

# Activate conda environment
source /root/paddlejob/workspace/env_run/output/whs/miniconda3/etc/profile.d/conda.sh
conda activate megtron

# Set PYTHONPATH
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/src:/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH

# Model path
MODEL_PATH="/root/paddlejob/workspace/env_run/output/whs/Qwen/Qwen3-30B-A3B"

# Log path
LOG_DIR="/root/paddlejob/workspace/env_run/output/whs/Megatron-Bridge/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_qwen35b.log"

# Run training
python -m torch.distributed.run --nproc_per_node=8 \
  scripts/training/run_recipe.py \
  --recipe qwen3_30b_a3b_sft_config \
  --dataset llm-finetune \
  --hf_path ${MODEL_PATH} \
  model.tensor_model_parallel_size=2 \
  model.pipeline_model_parallel_size=1 \
  model.expert_model_parallel_size=4 \
  model.sequence_parallel=True \
  train.train_iters=10 \
  validation.eval_iters=2 \
  2>&1 | tee ${LOG_FILE}
