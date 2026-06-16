# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.models.qwen.memory_token import MemoryQkvProjection, PrependMemoryTokenInjector
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    TELinearAdapter,
    patch_linear_module,
)
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear, is_expert_linear


logger = logging.getLogger(__name__)

try:
    import bitsandbytes

    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter)
        adapter_types = adapter_types + (TELinearAdapter,)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(module, nn.Linear) or (module.__class__ == te.Linear):
                # Will use the `patch_linear_module` function if:
                # - is FSDP v1
                # - is DTensor (has _local_tensor attribute)
                # - has quant_state attribute
                if hasattr(module.weight.data, "_local_tensor") or (
                    HAVE_BNB
                    and getattr(module, "quant_state", None) is not None
                    and module.quant_state.__class__ == bitsandbytes.functional.QuantState
                ):
                    lora_cls = patch_linear_module
                elif module.__class__ == te.Linear:
                    lora_cls = TELinearAdapter
                else:
                    lora_cls = LinearAdapter

                return lora_cls(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)

            enable_op_fuser = (
                hasattr(module, "config")
                and getattr(module.config, "use_transformer_engine_op_fuser", False)
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                attrs.in_features,
                attrs.out_features,
                self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(module, "config", None),
                alpha=self.alpha,
                is_expert=is_expert,
                a2a_experimental=self.a2a_experimental,
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


@dataclass
class MemorySparseAttentionLoRA(LoRA):
    """LoRA for SparseAttention distillation with configurable memory-parameter training.

    Extends standard LoRA to optionally unfreeze:
      - ``MemoryQkvProjection.memory_proj`` — per-memory-token QKV projection
      - ``PrependMemoryTokenInjector`` — the SwiGLU FFN that produces memory tokens

    This replaces the manual freeze/unfreeze logic previously in the
    ``_modify_student_before_ddp`` hook of the distillation script.
    """

    train_memory_proj: bool = False
    """Whether to train MemoryQkvProjection.memory_proj parameters."""

    train_memory_injector: bool = False
    """Whether to train PrependMemoryTokenInjector (SwiGLU FFN) parameters."""

    def __call__(self, model, training=True):
        model = super().__call__(model, training=training)

        if not training:
            return model

        models = model if isinstance(model, list) else [model]
        for m in models:
            for module in m.modules():
                if isinstance(module, MemoryQkvProjection) and self.train_memory_proj:
                    for p in module.memory_proj.parameters():
                        p.requires_grad = True
                if isinstance(module, PrependMemoryTokenInjector) and self.train_memory_injector:
                    for p in module.parameters():
                        p.requires_grad = True

            # Log trainable memory parameters.
            for n, p in m.named_parameters():
                if p.requires_grad and ("memory_proj" in n or "prepend_memory_token_injector" in n):
                    logger.info(f"[MemorySparseAttentionLoRA] Trainable: {n}, shape={list(p.shape)}")

        # Freeze MoE router expert_bias: prevent in-place updates during training.
        # expert_bias is a buffer (not a parameter) updated heuristically via
        # _apply_expert_bias(). During LoRA finetuning, LoRA changes seg1 routing
        # patterns which shifts expert_bias, causing seg0 tokens to be routed to
        # different experts than the base model — breaking the seg0 bypass guarantee.
        self._freeze_expert_bias(models)

        return model

    @staticmethod
    def _freeze_expert_bias(models):
        """Disable expert_bias update by setting moe_router_bias_update_rate to 0."""
        from megatron.core.utils import get_model_config
        try:
            config = get_model_config(models[0])
        except Exception:
            # Fallback: find config from router module
            config = None
            for m in models:
                for module in m.modules():
                    if hasattr(module, 'expert_bias') and module.expert_bias is not None:
                        config = module.config
                        break
                if config:
                    break
        if config is not None:
            config.moe_router_bias_update_rate = 0
            logger.info(
                "[MemorySparseAttentionLoRA] Set config.moe_router_bias_update_rate=0 "
                "to preserve seg0 routing consistency."
            )


@dataclass
class VLMLoRA(LoRA):
    """
    Implements the LoRA for Vision-Language Models.
    VLMLoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    """

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        modules_to_freeze = []

        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)


class LoRAMerge(PEFT):
    """
    Implements the LoRA weight merge for parameter-efficient fine-tuning.
    """

    def merge(
        self,
        base_weight: torch.Tensor,
        linear_out: torch.Tensor,
        linear_in: torch.Tensor,
        alpha: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Merges the LoRA adapter weights with the base model weights.
        Handles tensor parallelism by gathering sharded dimensions.

        For ColumnParallelLinear (e.g., linear_qkv, linear_fc1):
            - base_weight: (out_features/TP, in_features)
            - linear_in: (dim/TP, in_features) ← Need to gather this
            - linear_out: (out_features/TP, dim)
            - Target: (out_features/TP, dim) @ (dim, in_features) = (out_features/TP, in_features)

        For RowParallelLinear (e.g., linear_proj, linear_fc2):
            - base_weight: (out_features, in_features/TP)
            - linear_in: (dim, in_features/TP)
            - linear_out: (out_features/TP, dim) ← Need to gather this
            - Target: (out_features, dim) @ (dim, in_features/TP) = (out_features, in_features/TP)

        Args:
            base_weight (torch.Tensor): The base model weights.
            linear_out (torch.Tensor): LoRA's B matrix.
            linear_in (torch.Tensor): LoRA's A matrix.
            alpha (int): Weighting factor for the low-rank projection.
            dim (int): Dimension of the low-rank projection space.

        Returns:
            torch.Tensor: The merged weights.
        """

        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if tp_size == 1:
            # No tensor parallelism, simple multiplication
            lora_weight = alpha / dim * (linear_out @ linear_in)
            return base_weight + lora_weight

        tp_group = parallel_state.get_tensor_model_parallel_group()

        # Case 1: ColumnParallelLinear - linear_in is sharded on dim 0
        # linear_in: (dim/TP, in_features), linear_out: (out_features/TP, dim)
        if linear_in.shape[0] * tp_size == dim and linear_out.shape[1] == dim:
            # Gather linear_in along dimension 0 to get full dim
            linear_in_list = [torch.empty_like(linear_in) for _ in range(tp_size)]
            dist.all_gather(linear_in_list, linear_in, group=tp_group)
            linear_in_full = torch.cat(linear_in_list, dim=0)

            # Multiply: (out_features/TP, dim) @ (dim, in_features)
            lora_weight = alpha / dim * (linear_out @ linear_in_full)

        # Case 2: RowParallelLinear - linear_out is sharded on dim 0
        # linear_in: (dim, in_features/TP), linear_out: (out_features/TP, dim)
        elif linear_out.shape[0] * tp_size == base_weight.shape[0]:
            # Gather linear_out along dimension 0 to get full out_features
            linear_out_list = [torch.empty_like(linear_out) for _ in range(tp_size)]
            dist.all_gather(linear_out_list, linear_out, group=tp_group)
            linear_out_full = torch.cat(linear_out_list, dim=0)

            # Multiply: (out_features, dim) @ (dim, in_features/TP)
            lora_weight = alpha / dim * (linear_out_full @ linear_in)

        else:
            # Fallback: no gathering needed or already full-size
            lora_weight = alpha / dim * (linear_out @ linear_in)

        return base_weight + lora_weight

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Merges the LoRA adapter with the base model weights.

        Args:
            m (nn.Module): The module to apply LoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the LoRA adapter merged into the base model weights.
        """

        if not isinstance(module, LoRALinear):
            return module
        logging.info(f"merging {(prefix if prefix else '') + '.' + (name if name else '')}")

        if hasattr(module.to_wrap, "weight"):
            base_device = module.to_wrap.weight.device
            merged_weight = self.merge(
                module.to_wrap.weight,
                module.adapter.linear_out.weight.to(base_device),
                module.adapter.linear_in.weight.to(base_device),
                module.adapter.alpha,
                module.adapter.dim,
            )
            module.to_wrap.weight.data = merged_weight
        else:  # TE Grouped Linear
            for i in range(module.to_wrap.num_gemms):
                base_device = getattr(module.to_wrap, f"weight{i}").device
                merged_weight = self.merge(
                    getattr(module.to_wrap, f"weight{i}"),
                    module.adapter.linear_out.weight.to(base_device),
                    module.adapter.linear_in.weight.to(base_device),
                    module.adapter.alpha,
                    module.adapter.dim,
                )
                getattr(module.to_wrap, f"weight{i}").data = merged_weight
        return module
