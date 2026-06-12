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

from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen.qwen3_swap_attention import (
    AttnOutputCollector,
    swap_to_flashmask,
    swap_to_memory_qkv,
)
from megatron.bridge.models.qwen.memory_token import (
    MemoryQkvProjection,
    PrependMemoryTokenInjector,
    register_prepend_memory_token_injector,
)
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_sft_config
from megatron.bridge.training.callbacks import Callback, CallbackContext, CallbackManager
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.sparse_distill import SparseDistillForwardStep
from megatron.bridge.recipes.utils.dataset_utils import DATASET_TYPES, apply_dataset_override
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

logger = logging.getLogger(__name__)


class _SparseDistillSetup(Callback):
    """Hooks called at training start to attach output collectors for both
    student and teacher models.

    Note: structural model modifications (swap_to_flashmask, swap_to_memory_qkv,
    freeze_student) are applied via a ``pre_wrap_hook`` so they run *before*
    DDP wrapping and optimizer creation.  This callback only attaches the
    ``AttnOutputCollector`` hooks and builds the teacher model.
    """

    def __init__(
        self,
        hf_path: str,
        forward_step: SparseDistillForwardStep,
    ) -> None:
        self.hf_path = hf_path
        self.forward_step = forward_step

    def on_train_start(self, context: CallbackContext) -> None:
        for chunk in context.model:
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
    p.add_argument("--config_file", type=str, default=None, help="Path to YAML config file.")
    p.add_argument(
        "--data_path",
        "--data-path",
        type=str,
        default=None,
        help="Path to a JSONL pretraining-style dataset file or a directory containing training.jsonl.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dataset", type=str, default=None, choices=DATASET_TYPES)
    p.add_argument(
        "--group_size",
        type=int,
        default=0,
        help="Memory-token group size g. 0 disables memory-token augmentation.",
    )
    p.add_argument(
        "--segment_size",
        type=int,
        default=0,
        help="Sparse-mask segment size s (in original tokens). Must be a multiple "
             "of --group_size. 0 disables the interleaved sparse mask (causal only).",
    )
    p.add_argument(
        "--swa_only",
        action="store_true",
        default=False,
        help="When set, skip rule ② (t -> m) in the sparse attention mask. "
             "SWA = Sliding Window Attention: regular tokens will only attend "
             "within their sliding window and not to memory tokens from "
             "previous segments.",
    )
    p.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="Token id used as a placeholder for memory slots in student input.",
    )
    args, cli_overrides = p.parse_known_args()
    if args.segment_size > 0:
        if args.group_size <= 0:
            raise ValueError("--segment_size requires --group_size > 0")
        if args.segment_size % args.group_size != 0:
            raise ValueError(
                f"--segment_size ({args.segment_size}) must be divisible by "
                f"--group_size ({args.group_size})"
            )
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

    # Ensure eos_token is set for HuggingFaceTokenizer so tokenizer.eos_id is non-None.
    cfg.tokenizer.eos_token = "<|im_end|>"

    # Replace default SQuAD dataset with FinetuningDatasetConfig so YAML can
    # set dataset_root and all other pretrain-style dataset fields.
    cfg.dataset = FinetuningDatasetConfig(seq_length=cfg.model.seq_length)

    cfg = process_config_with_overrides(config=cfg, config_filepath=args.config_file, cli_overrides=cli_overrides or None)

    if args.dataset is not None:
        cfg = apply_dataset_override(cfg, dataset_type=args.dataset, cli_overrides=cli_overrides)

    if args.data_path is not None:
        from pathlib import Path
        path = Path(args.data_path)
        cfg.dataset.dataset_root = path.parent if path.suffix == ".jsonl" else path

    student_collector = AttnOutputCollector()
    teacher_collector = AttnOutputCollector()
    forward_step = SparseDistillForwardStep(
        teacher_models=[],
        student_collector=student_collector,
        teacher_collector=teacher_collector,
        alpha=args.alpha,
        temperature=args.temperature,
        group_size=args.group_size,
        pad_token_id=args.pad_token_id,
    )

    # Structural modifications must happen BEFORE DDP wrapping and optimizer
    # creation so that (a) new parameters (e.g. MemoryQkvProjection.memory_proj)
    # get main_grad buffers allocated by DDP, and (b) frozen parameters are
    # excluded from the optimizer's param groups.
    # We use a pre_wrap_hook which fires inside _build_distributed_model,
    # after the GPTModel is constructed but before DDP wraps it.
    freeze_student = cfg.train.freeze_student
    swa_only = getattr(cfg.train, "swa_only", args.swa_only)

    def _modify_student_before_ddp(models):
        sparse_kwargs = {}
        if args.group_size > 0 and args.segment_size > 0:
            sparse_kwargs = {
                "group_size": args.group_size,
                "segment_size": args.segment_size,
                "swa_only": swa_only,
            }
        for m in models:
            swap_to_flashmask(m, **sparse_kwargs)
            if args.group_size > 0:
                swap_to_memory_qkv(m, group_size=args.group_size)

            # Attach memory-token injector on the first PP stage.
            if args.group_size > 0 and getattr(m, "embedding", None) is not None:
                register_prepend_memory_token_injector(m, group_size=args.group_size)

            # Freeze all parameters except MemoryTokenInjector and
            # MemoryQkvProjection so only they are trained.
            if freeze_student:
                for p in m.parameters():
                    p.requires_grad_(False)
                for module in m.modules():
                    if isinstance(module, PrependMemoryTokenInjector):
                        for p in module.parameters():
                            p.requires_grad_(True)
                    elif isinstance(module, MemoryQkvProjection):
                        for p in module.memory_proj.parameters():
                            p.requires_grad_(True)
        return models

    cfg.model.register_pre_wrap_hook(_modify_student_before_ddp)

    cb = _SparseDistillSetup(
        hf_path=args.hf_path,
        forward_step=forward_step,
    )
    cm = CallbackManager()
    cm.add(cb)

    pretrain(config=cfg, forward_step_func=forward_step, callbacks=cm)


if __name__ == "__main__":
    main()
