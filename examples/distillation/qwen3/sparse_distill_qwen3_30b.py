#!/usr/bin/env python3
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

"""SparseAttention distillation training entrypoint for Qwen3-30B-A3B.

Student (B): Qwen3-30B-A3B with FlashMaskAttention swapped in for every
decoder layer.
Teacher (A): original Qwen3-30B-A3B with standard attention, eval-mode + no_grad.
KL loss is computed between every layer's core-attention output (pre o_proj)
and added to the standard SFT cross-entropy loss.

Usage::

    python -m torch.distributed.run --nproc_per_node=8 \\
        examples/distillation/qwen3/sparse_distill_qwen3_30b.py \\
        --hf_path /path/to/Qwen3-30B-A3B \\
        train.train_iters=10
"""

import argparse
import logging

import torch.nn as nn
from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen.qwen3_swap_attention import (
    AttnOutputCollector,
    swap_to_flashmask,
)
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_sft_config
from megatron.bridge.training.callbacks import Callback, CallbackContext, CallbackManager
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.sparse_distill import SparseDistillForwardStep
from megatron.bridge.recipes.utils.dataset_utils import DATASET_TYPES, apply_dataset_override
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

logger = logging.getLogger(__name__)


def _refresh_ddp_pre_hooks(model: nn.Module) -> None:
    """Re-register DDP forward pre-hooks after submodule replacement.

    swap_to_flashmask replaces core_attention submodules in place on a DDP-wrapped
    model. DDP's remove_forward_pre_hook_handles still holds handles for the old
    modules, and the new FlashMaskAttention instances are unknown to it. This causes
    KeyError in disable_forward_pre_hook when it iterates self.module.modules().

    Fix: remove stale handles, clear the registry, then re-register for all current
    submodules so DDP's hook state is consistent.
    """
    try:
        from megatron.core.distributed import DistributedDataParallel as CoreDDP
    except ImportError:
        return
    if not isinstance(model, CoreDDP) or not model.use_forward_hook:
        return
    for handle in model.remove_forward_pre_hook_handles.values():
        handle.remove()
    model.remove_forward_pre_hook_handles.clear()
    model.enable_forward_pre_hook()


class _SparseDistillSetup(Callback):
    """Hooks called at training start to: (1) swap student attention to
    FlashMaskAttention, (2) build the teacher model, (3) attach hooks for both."""

    def __init__(self, hf_path: str, forward_step: SparseDistillForwardStep) -> None:
        self.hf_path = hf_path
        self.forward_step = forward_step

    def on_train_start(self, context: CallbackContext) -> None:
        # Student: swap attention in place. Weights of linear_qkv / linear_proj /
        # qk_layernorm are preserved.
        for chunk in context.model:
            swap_to_flashmask(chunk)
            # swap_to_flashmask replaces core_attention submodules after DDP has
            # already registered forward pre-hooks for the original modules.  The
            # new FlashMaskAttention instances are absent from DDP's
            # remove_forward_pre_hook_handles, which causes KeyError when
            # disable_forward_pre_hook iterates self.module.modules().
            # Refresh DDP hook registration to include the swapped-in modules.
            _refresh_ddp_pre_hooks(chunk)
            self.forward_step.student_collector.attach(chunk)

        # Teacher: a fresh GPTModel with the same provider but standard attention.
        # Built directly from HF weights, frozen, eval-mode, on the same device.
        provider = AutoBridge.from_hf_pretrained(self.hf_path).to_megatron_provider(load_weights=True)
        # Match parallelism with student
        cfg = context.state.cfg.model
        provider.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        provider.expert_model_parallel_size = cfg.expert_model_parallel_size
        provider.sequence_parallel = cfg.sequence_parallel
        provider.seq_length = cfg.seq_length
        provider.finalize()

        teacher = provider.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)
        for tm in teacher:
            tm.eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            self.forward_step.teacher_collector.attach(tm)
        self.forward_step.teacher_models = teacher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_path", type=str, default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dataset", type=str, default=None, choices=DATASET_TYPES)
    args, cli_overrides = p.parse_known_args()
    return args, cli_overrides


def main():
    args, cli_overrides = parse_args()

    cfg: ConfigContainer = qwen3_30b_a3b_sft_config()
    # Use the local (non-TE) attention path so hooks see Python core_attention.
    cfg.model = AutoBridge.from_hf_pretrained(args.hf_path).to_megatron_provider(load_weights=True)
    cfg.model.transformer_impl = "local"
    # For PP=1 the hook collection logic in this scaffold is straightforward;
    # PP>1 needs cross-stage aggregation — see sparse_dist.md.
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 4
    cfg.model.sequence_parallel = True

    cfg = process_config_with_overrides(config=cfg, cli_overrides=cli_overrides or None)

    if args.dataset is not None:
        cfg = apply_dataset_override(cfg, dataset_type=args.dataset, cli_overrides=cli_overrides)

    student_collector = AttnOutputCollector()
    teacher_collector = AttnOutputCollector()
    forward_step = SparseDistillForwardStep(
        teacher_models=[],
        student_collector=student_collector,
        teacher_collector=teacher_collector,
        alpha=args.alpha,
        temperature=args.temperature,
    )

    cb = _SparseDistillSetup(hf_path=args.hf_path, forward_step=forward_step)
    cm = CallbackManager()
    cm.add(cb)

    pretrain(config=cfg, forward_step_func=forward_step, callbacks=cm)


if __name__ == "__main__":
    main()
