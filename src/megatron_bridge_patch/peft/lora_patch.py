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

"""
LoRA patch to add Router LoRA and per-expert GroupedLinear LoRA support.

This patch extends the LoRA.transform() method to support:
1. Applying LoRA to MoE Router modules
2. Per-expert LoRA for TE GroupedLinear (moe_grouped_gemm=True)

Usage:
    from megatron_bridge_patch.peft.lora_patch import apply_grouped_mlp_lora_patch
    apply_grouped_mlp_lora_patch()
"""

import dataclasses
import functools
import logging
import torch.nn as nn

_LORA_TRANSFORM_PATCH_APPLIED = False
_RECOMPUTE_PATCH_APPLIED = False

logger = logging.getLogger(__name__)


def apply_lora_router_patch():
    """Patch LoRA.transform() to add support for Router LoRA.

    保留此函数以保持向后兼容性，实际逻辑已合并到 apply_grouped_mlp_lora_patch 中统一处理。
    """
    apply_grouped_mlp_lora_patch()


def apply_grouped_mlp_lora_patch():
    """对 LoRA.transform() 进行补丁修改，统一支持 Router LoRA 和按专家（per-expert）的 GroupedLinear LoRA。

    1. Router LoRA: 当模块为 MoE Router 且匹配 target_modules 时，应用 RouterLoRAAdapter。
    2. GroupedLinear LoRA: 当 moe_grouped_gemm=True 时，为 TE GroupedLinear
       （及 QuantizedGroupedLinear）的每个专家创建独立的 LoRA adapter，而非共享一个 adapter。
    """
    global _LORA_TRANSFORM_PATCH_APPLIED
    if _LORA_TRANSFORM_PATCH_APPLIED:
        return
    
    from megatron.bridge.peft.lora import LoRA

    # Patch LoRA dataclass to accept moe_lora_backend from YAML config.
    # We must inject it into __dataclass_fields__ AND __init__ so that
    # Hydra's instantiate() recognizes it as a valid key (not just a class attr).
    if 'moe_lora_backend' not in LoRA.__dataclass_fields__:
        _new_field = dataclasses.field(default="te")
        _new_field.name = 'moe_lora_backend'
        _new_field._field_type = dataclasses._FIELD
        LoRA.__dataclass_fields__['moe_lora_backend'] = _new_field

        _orig_init = LoRA.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self, *args, moe_lora_backend="te", **kwargs):
            _orig_init(self, *args, **kwargs)
            self.moe_lora_backend = moe_lora_backend

        del _patched_init.__wrapped__
        LoRA.__init__ = _patched_init

    from megatron.bridge.peft.lora_layers import (
        LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter,
    )
    from megatron.bridge.peft.utils import (
        ParallelLinearAdapter,
        get_adapter_attributes_from_linear,
        is_expert_linear,
    )
    from megatron_bridge_patch.peft.router_adapter import RouterLoRAAdapter, is_router_module
    from megatron_bridge_patch.peft.grouped_mlp_lora import (
        GroupedLoRALinear,
        is_te_grouped_linear,
        is_quantized_grouped_linear,
        get_num_experts_from_module,
    )
    from megatron_bridge_patch.peft.utils_patch import create_grouped_lora_gemm

    # 已经经过 transform 的模块类型，跳过避免重复包装
    _already_transformed_types = (
        LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter, GroupedLoRALinear,
    )

    _original_transform = LoRA.transform
    
    def _patched_transform(self, module, name=None, prefix=None):
        """Unified transform: Router LoRA + per-expert GroupedLinear LoRA."""

        # 1. 跳过已经经过 transform 的模块
        if isinstance(module, _already_transformed_types):
            logger.info(f"[Transform] Skipped already-transformed module: "
                        f"name={name}, prefix={prefix}, type={type(module).__name__}")
            return module

        # 2. Skip MemoryQkvProjection — do NOT apply LoRA to it or its inner linear_qkv.
        from megatron.bridge.models.qwen.memory_token import MemoryQkvProjection
        if isinstance(module, MemoryQkvProjection):
            return module

        # 3. Router LoRA
        if is_router_module(module):
            # Build full name for pattern matching
            full_name = f"{prefix}.{name}" if prefix else name
            should_apply = False
            
            # Check against target_modules patterns
            for pattern in self.target_modules:
                if pattern == "router" or pattern == name or (full_name and pattern in full_name):
                    should_apply = True
                    break
            
            # Also apply if 'router' is explicitly in target_modules
            if "router" in self.target_modules:
                should_apply = True
            
            if should_apply:
                logger.info(f"[Transform] {type(module).__name__} -> RouterLoRAAdapter: "
                            f"full_name={full_name}, dim={self.dim}, alpha={self.alpha}")
                RouterLoRAAdapter(
                    router=module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    dropout_position=self.dropout_position,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_B_init_method=self.lora_B_init_method,
                    lora_dtype=self.lora_dtype,
                )
                # Return the same module (it's been modified in-place)
                return module
            else:
                logger.info(f"[Transform] Router module not matched by target_modules: "
                            f"full_name={full_name}, type={type(module).__name__}")

        # 3. Per-expert LoRA for TE GroupedLinear
        if (ans := self.match(module, name, prefix)) is not None:
            match, full_name = ans

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)

            if (is_te_grouped_linear(module) or is_quantized_grouped_linear(module)) and is_expert:
                num_experts = get_num_experts_from_module(module)
                backend = getattr(self, 'moe_lora_backend', 'te')

                # IMPORTANT: For GroupedLoRALinear, the LoRA gemm_A/gemm_B use
                # te.GroupedLinear(parallel_mode=None), which does NOT perform TP
                # splitting internally.  The base TE GroupedLinear, however, has
                # already been TP-split (output_size /= tp_size for ColumnParallel,
                # input_size /= tp_size for RowParallel) when explicit_expert_comm
                # is True.  Therefore we must use the TP-split dimensions directly
                # from the module (m.in_features / m.out_features) rather than the
                # full dimensions from attrs (which multiply back by tp_size for
                # ParallelLinearAdapter which handles TP internally).
                actual_in = module.in_features
                actual_out = module.out_features

                logger.info(f"[Transform] {type(module).__name__} -> GroupedLoRALinear: "
                            f"full_name={full_name}, num_experts={num_experts}, "
                            f"in_features={actual_in}, out_features={actual_out}, "
                            f"dim={self.dim}, alpha={self.alpha}, backend={backend}")

                gemm_A, gemm_B = create_grouped_lora_gemm(
                    num_experts, actual_in, actual_out, self.dim,
                    self.lora_A_init_method, self.lora_B_init_method,
                    getattr(module, "config", None),
                    backend=backend,
                )

                return GroupedLoRALinear(
                    module, gemm_A, gemm_B, num_experts,
                    alpha=self.alpha, dim=self.dim, linear_name=full_name,
                    input_is_parallel=attrs.input_is_parallel,
                    backend=backend,
                )
            else:
                logger.info(f"[Transform] Matched but falling back to original transform: "
                            f"full_name={full_name}, type={type(module).__name__}, "
                            f"is_expert={is_expert}, "
                            f"is_te_grouped={is_te_grouped_linear(module)}, "
                            f"is_quantized_grouped={is_quantized_grouped_linear(module)}")

        # 4. 其他模块：回退到原始 transform
        logger.info(f"[Transform] Fallback to original transform: "
                    f"name={name}, prefix={prefix}, type={type(module).__name__}")
        return _original_transform(self, module, name, prefix)
    
    # Apply the patch
    LoRA.transform = _patched_transform

    _LORA_TRANSFORM_PATCH_APPLIED = True
    logger.info("[Patch] Applied unified LoRA transform patch (Router + per-expert GroupedLinear)")


def apply_recompute_patch():
    """Patch maybe_enable_recompute_inputs_grad to recognise GroupedLoRALinear params.

    Root cause:
      GroupedLoRALinear stores LoRA A/B matrices as self.gemm_A / self.gemm_B.
      - backend="te": these are te.GroupedLinear submodules, param names like ".gemm_A.weight0"
      - backend="torch": these are nn.Parameter, param names like ".gemm_A"
      In both cases, they don't match ".adapter." marker.

      The original trainable_adapter / trainable_base checks only look for ".adapter.",
      causing trainable_adapter=False and the whole patch to be skipped — leaving
      hidden_states.requires_grad=False when entering torch.utils.checkpoint,
      which silently skips backward and produces grad_norm=0 for all LoRA params.
    """
    global _RECOMPUTE_PATCH_APPLIED
    if _RECOMPUTE_PATCH_APPLIED:
        return

    import torch
    from functools import wraps
    from typing import Set, Optional

    import megatron.bridge.peft.recompute as _recompute_mod
    from megatron.bridge.peft.recompute import PEFT_RECOMPUTE_PATCHED, _iter_unwrapped_models
    from megatron.bridge.utils.common_utils import print_rank_0

    # ".adapter." covers standard AdapterWrapper LoRA params.
    # ".gemm_a" / ".gemm_b" cover GroupedLoRALinear (both backends):
    #   - TE backend: gemm_A is te.GroupedLinear submodule → param name ".gemm_A.weight0"
    #   - torch backend: gemm_A is nn.Parameter → param name ".gemm_A"
    # All matched via n.lower() containing ".gemm_a" / ".gemm_b".
    _ADAPTER_MARKERS = (".adapter.", ".gemm_a", ".gemm_b")

    def _patched_maybe_enable_recompute_inputs_grad(
        model, peft_recompute_patched: Optional[Set[int]] = None
    ) -> Set[int]:
        from megatron.core.transformer.transformer_block import TransformerBlock

        patched_registry = peft_recompute_patched or PEFT_RECOMPUTE_PATCHED

        try:
            for unwrapped_model in _iter_unwrapped_models(model):
                cfg = getattr(unwrapped_model, "config", None)
                if cfg is None or getattr(cfg, "recompute_method", None) is None:
                    continue

                if id(unwrapped_model) in patched_registry:
                    continue

                params = list(unwrapped_model.named_parameters())
                trainable_adapter = any(
                    p.requires_grad and any(m in n.lower() for m in _ADAPTER_MARKERS)
                    for n, p in params
                )
                trainable_base = any(
                    p.requires_grad
                    and ".to_wrap." not in n.lower()
                    and not any(m in n.lower() for m in _ADAPTER_MARKERS)
                    for n, p in params
                )

                if not (trainable_adapter and not trainable_base):
                    continue

                def _patch_transformer_block(module: torch.nn.Module) -> bool:
                    if isinstance(module, TransformerBlock):
                        original_forward = module.forward

                        @wraps(original_forward)
                        def patched_forward(
                            hidden_states, *args, _original_forward=original_forward, **kwargs
                        ):
                            if (
                                torch.is_tensor(hidden_states)
                                and not hidden_states.requires_grad
                                and hidden_states.is_floating_point()
                            ):
                                hidden_states = hidden_states.detach().requires_grad_(True)
                            return _original_forward(hidden_states, *args, **kwargs)

                        module.forward = patched_forward
                        return True
                    return False

                patched = False
                for module in unwrapped_model.modules():
                    if _patch_transformer_block(module):
                        patched = True
                if patched:
                    patched_registry.add(id(unwrapped_model))
                    print_rank_0(
                        "[PEFT+Recompute] Patched TransformerBlock.forward to enable grad on "
                        "hidden_states input. This ensures checkpoint backward is called when "
                        "only adapters are trainable (PP=1 with frozen base model).",
                    )
        except Exception as exc:
            print_rank_0(f"[PEFT+Recompute] Warning: Failed to patch TransformerBlock: {exc}")

        return patched_registry

    # Replace in recompute module
    _recompute_mod.maybe_enable_recompute_inputs_grad = _patched_maybe_enable_recompute_inputs_grad

    # Also replace the reference already imported by name in base.py
    try:
        import megatron.bridge.peft.base as _base_mod
        _base_mod.maybe_enable_recompute_inputs_grad = _patched_maybe_enable_recompute_inputs_grad
    except Exception:
        pass

    _RECOMPUTE_PATCH_APPLIED = True
    logger.info(
        "[Patch] Applied recompute patch: GroupedLoRALinear gemm_A/gemm_B params "
        "now recognised as adapter params in maybe_enable_recompute_inputs_grad"
    )