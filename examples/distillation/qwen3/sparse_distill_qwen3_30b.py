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

Student (B): Qwen3-30B-A3B with attention configured via ``--attention_type``:
  - ``full``: standard attention (no swaps).
  - ``swa``: Sliding Window Attention only (FlashMaskAttention with swa_only=True).
  - ``msa``: Memory-augmented Sparse Attention (FlashMaskAttention, MemoryQkvProjection,
    and PrependMemoryTokenInjector).
Teacher (A): original Qwen3-30B-A3B with standard attention, eval-mode + no_grad.
KL loss is computed between every layer's core-attention output (pre o_proj)
and added to the standard SFT cross-entropy loss.

Usage::

    python -m torch.distributed.run --nproc_per_node=8 \\
        examples/distillation/qwen3/sparse_distill_qwen3_30b.py \\
        --hf_path /path/to/Qwen3-30B-A3B \\
        --attention_type msa \\
        train.train_iters=10
"""

import argparse
import logging

from megatron.bridge import AutoBridge
from megatron_bridge_patch.peft import apply_all_peft_patches
from megatron.bridge.models.qwen.qwen3_swap_attention import (
    AttnOutputCollector,
    install_seg0_lora_bypass,
    remove_seg0_lora_bypass,
    swap_to_memory_qkv,
    swap_to_swa,
    swap_to_msa,
)
from megatron.core.transformer.attention import SelfAttention
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
    freeze-all-then-selectively-unfreeze) are applied via a ``pre_wrap_hook`` so they run *before*
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
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dataset", type=str, default=None, choices=DATASET_TYPES)
    p.add_argument(
        "--attention_type",
        type=str,
        default="full",
        choices=["full", "swa", "msa"],
        help="Attention type: 'full' (standard attention), 'swa' (sliding window "
             "attention only), 'msa' (memory-augmented sparse attention with "
             "memory token injection).",
    )
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
        "--pad_token_id",
        type=int,
        default=0,
        help="Token id used as a placeholder for memory slots in student input.",
    )
    args, cli_overrides = p.parse_known_args()
    if args.attention_type != "full" and args.group_size <= 0:
        raise ValueError(
            f"--attention_type={args.attention_type} requires --group_size > 0"
        )
    if args.segment_size > 0 and args.segment_size % args.group_size != 0:
        raise ValueError(
            f"--segment_size ({args.segment_size}) must be divisible by "
            f"--group_size ({args.group_size})"
        )
    return args, cli_overrides


def main():
    apply_all_peft_patches()
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

    attention_type = cfg.model.attention_type
    group_size = cfg.model.group_size
    segment_size = cfg.model.segment_size
    m2t_mode = cfg.model.m2t_mode
    m2m_mode = cfg.model.m2m_mode
    seg0_lora_bypass = getattr(cfg.model, 'seg0_lora_bypass', True)
    pad_token_id = cfg.tokenizer.pad_token_id
    train_memory_compression_projection = cfg.train.train_memory_compression_projection
    train_memory_qkv_projection = cfg.train.train_memory_qkv_projection
    train_common_qkv_projection = cfg.train.train_common_qkv_projection
    alpha = cfg.train.distill_alpha
    beta = cfg.train.distill_beta
    temperature = cfg.train.distill_temperature
    use_jsd = cfg.train.use_jsd

    student_collector = AttnOutputCollector()
    teacher_collector = AttnOutputCollector()
    forward_step = SparseDistillForwardStep(
        teacher_models=[],
        student_collector=student_collector,
        teacher_collector=teacher_collector,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        use_jsd=use_jsd,
        group_size=group_size,
        pad_token_id=pad_token_id,
    )

    # Structural modifications must happen BEFORE DDP wrapping and optimizer
    # creation so that (a) new parameters (e.g. MemoryQkvProjection.memory_proj)
    # get main_grad buffers allocated by DDP, and (b) frozen parameters are
    # excluded from the optimizer's param groups.
    # We use a pre_wrap_hook which fires inside _build_distributed_model,
    # after the GPTModel is constructed but before DDP wraps it.

    def _modify_student_before_ddp(models):
        for m in models:
            if attention_type == "swa":
                swap_to_swa(m, window_size=segment_size)
            elif attention_type == "msa":
                swap_to_msa(m, group_size=group_size, segment_size=segment_size, m2t_mode=m2t_mode, m2m_mode=m2m_mode)
                swap_to_memory_qkv(m, group_size=group_size)
                # Attach memory-token injector on the first PP stage.
                if getattr(m, "embedding", None) is not None:
                    register_prepend_memory_token_injector(m, group_size=group_size)

                # Install seg0 LoRA bypass hooks so that the first segment_size
                # tokens receive no LoRA contribution, matching the base model.
                if seg0_lora_bypass and cfg.peft is not None:
                    # seg0_tokens = segment_size (the first segment constitutes seg0)
                    seg0_tokens = segment_size
                    handles = install_seg0_lora_bypass(m, seg0_tokens)
                    # Store handles on the model so they can be cleaned up later if needed
                    m._seg0_bypass_handles = handles
                    logger.info(
                        "[seg0_lora_bypass] Installed on MSA layers, seg0_tokens=%d",
                        seg0_tokens,
                    )

            # When LoRA (MemorySparseAttentionLoRA) is configured via cfg.peft,
            # freeze/unfreeze is handled by the PEFT __call__ flow that runs
            # after this hook. Only apply manual freeze/unfreeze when no PEFT
            # is configured.
            if cfg.peft is None:
                print(f"train_memory_compression_projection: {train_memory_compression_projection}; train_memory_qkv_projection: {train_memory_qkv_projection}; train_common_qkv_projection: {train_common_qkv_projection}")
                # Freeze all parameters except selected modules based on config.
                for p in m.parameters():
                    p.requires_grad_(False)
                for module in m.modules():
                    if isinstance(module, PrependMemoryTokenInjector):
                        if train_memory_compression_projection:
                            for p in module.parameters():
                                p.requires_grad_(True)
                    elif isinstance(module, MemoryQkvProjection):
                        if train_memory_qkv_projection:
                            for p in module.memory_proj.parameters():
                                p.requires_grad_(True)
                        if train_common_qkv_projection:
                            for p in module.linear_qkv.parameters():
                                p.requires_grad_(True)

                # Log all trainable parameters.
                total_trainable_params = 0
                for n, p in m.named_parameters():
                    if p.requires_grad:
                        num_params = p.numel()
                        total_trainable_params += num_params
                        logger.info("[Trainable] %s, shape=%s, numel=%d", n, list(p.shape), num_params)
                logger.info("[Trainable] total trainable parameters: %d", total_trainable_params)

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
