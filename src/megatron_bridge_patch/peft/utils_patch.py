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
PEFT utils patch for enhanced initialization functions.

This patch adds:
1. New initialization functions for expert-aware Kaiming uniform
2. get_init_fn() and get_init_fn_expert() utility functions
3. Updated ParallelLinearAdapter to use expert-aware initialization

Usage:
    from megatron_bridge_patch.peft.utils_patch import apply_peft_utils_patch
    apply_peft_utils_patch()
"""

import math
import os
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

_PEFT_UTILS_PATCH_APPLIED = False


# ============================================================================
# New initialization functions
# ============================================================================

def _calculate_fan_in_and_fan_out(tensor):
    """Calculate fan_in and fan_out for a tensor."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    """Calculate the correct fan value based on mode."""
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function."""
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: Optional[torch.Generator] = None,
):
    """
    Fill the input Tensor with values using a Kaiming uniform distribution.

    Modified version for MoE expert initialization with fixed fan calculation.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    gain = calculate_gain(nonlinearity, a)
    # Use fixed fan value for consistent expert initialization
    std = gain / math.sqrt(64.0)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def _use_expert_fixed_fan_init() -> bool:
    """Whether expert LoRA layers should use custom fixed fan_in=64 kaiming init."""
    return os.getenv("LORA_EXPERT_FIXED_FAN_INIT", "").lower() in ("1", "true", "yes", "on")


def init_method_kaiming_uniform_expert(val: float, is_expert: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an initialization method based on Kaiming uniform distribution with expert-awareness.

    Args:
        val: The 'a' parameter for Kaiming uniform initialization.
        is_expert: Whether this is an expert layer.

    Returns:
        Initialization function that applies Kaiming uniform distribution to a tensor.
    """
    if is_expert and _use_expert_fixed_fan_init():
        def init_(tensor: torch.Tensor) -> torch.Tensor:
            return kaiming_uniform_(tensor, a=val)
    else:
        def init_(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.kaiming_uniform_(tensor, a=val)
    return init_


def get_init_fn(init_method: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get initialization function by method name.

    Args:
        init_method: Name of the initialization method ('xavier', 'normal', 'kaiming', or 'zero').

    Returns:
        Initialization function that can be applied to a tensor.

    Raises:
        NotImplementedError: If init_method is not supported.
    """
    from megatron.bridge.peft.utils import init_method_normal, init_method_kaiming_uniform, init_method_const

    if init_method == "xavier":
        init_fn = nn.init.xavier_normal_
    elif init_method == "normal":
        init_fn = init_method_normal(0.2)
    elif init_method == "kaiming":
        init_fn = init_method_kaiming_uniform(math.sqrt(5))
    elif init_method == "zero":
        init_fn = init_method_const(0.0)
    elif init_method == "constant":
        init_fn = init_method_const(val=0.5)
    else:
        raise NotImplementedError(f"init_method should be zero, normal, kaiming or xavier, got {init_method}")
    return init_fn


def create_grouped_lora_gemm(num_experts, in_features, out_features, dim,
                              lora_A_init_method, lora_B_init_method, config,
                              backend="te"):
    """创建用于 LoRA 计算的 gemm_A + gemm_B。

    根据 backend 参数选择实现：
      - "te"（默认）: 返回 TE GroupedLinear 实例对，bit-exact，兼容所有 GPU
      - "torch": 返回 stacked weight tensor 对 [E, in, rank]，使用 torch.nn.functional.grouped_mm

    Args:
        num_experts: 专家数量（local，即本 EP rank 的专家数）
        in_features: 输入维度
        out_features: 输出维度
        dim: LoRA rank
        lora_A_init_method: A 矩阵初始化方法名称
        lora_B_init_method: B 矩阵初始化方法名称
        config: ModelParallelConfig（用于 bf16/fp16 dtype 选择）
        backend: "te" 或 "torch"，控制 grouped GEMM 后端

    Returns:
        backend="te":    (gemm_A: te.GroupedLinear, gemm_B: te.GroupedLinear)
        backend="torch": (W_A: Tensor[E, in, dim], W_B: Tensor[E, dim, out])
    """
    if backend == "te":
        return _create_grouped_lora_gemm_te(
            num_experts, in_features, out_features, dim,
            lora_A_init_method, lora_B_init_method, config,
        )
    else:
        return _create_grouped_lora_gemm_torch(
            num_experts, in_features, out_features, dim,
            lora_A_init_method, lora_B_init_method, config,
        )


def _create_grouped_lora_gemm_te(num_experts, in_features, out_features, dim,
                                   lora_A_init_method, lora_B_init_method, config):
    """创建 TE GroupedLinear 实例对（bit-exact，默认后端）。"""
    import transformer_engine.pytorch as te

    gemm_A = te.GroupedLinear(
        num_gemms=num_experts,
        in_features=in_features,
        out_features=dim,
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
    )
    gemm_B = te.GroupedLinear(
        num_gemms=num_experts,
        in_features=dim,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
    )

    a_init_fn = get_init_fn(lora_A_init_method)
    b_init_fn = get_init_fn(lora_B_init_method)
    for i in range(num_experts):
        a_init_fn(getattr(gemm_A, f'weight{i}'))
        b_init_fn(getattr(gemm_B, f'weight{i}'))

    if config is not None:
        if config.bf16:
            gemm_A.bfloat16()
            gemm_B.bfloat16()
        elif config.fp16:
            gemm_A.half()
            gemm_B.half()

    return gemm_A, gemm_B


def _create_grouped_lora_gemm_torch(num_experts, in_features, out_features, dim,
                                      lora_A_init_method, lora_B_init_method, config):
    """创建 stacked weight tensor 对，用于 torch.nn.functional.grouped_mm。"""
    dtype = torch.bfloat16
    if config is not None and config.fp16:
        dtype = torch.float16

    W_A = torch.empty(num_experts, in_features, dim, dtype=dtype)
    W_B = torch.zeros(num_experts, dim, out_features, dtype=dtype)

    a_init_fn = get_init_fn(lora_A_init_method)
    b_init_fn = get_init_fn(lora_B_init_method)
    for i in range(num_experts):
        temp_a = torch.empty(dim, in_features, dtype=dtype)   # [rank, in_features]
        a_init_fn(temp_a)                                      # fan_in = in_features ✓
        W_A[i].copy_(temp_a.t())                               # 存为 [in_features, rank]

        temp_b = torch.empty(out_features, dim, dtype=dtype)   # [out_features, rank]
        b_init_fn(temp_b)                                      # fan_in = rank ✓
        W_B[i].copy_(temp_b.t())                               # 存为 [rank, out_features]

    return W_A, W_B


def get_init_fn_expert(init_method: str, is_expert: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get initialization function by method name with expert-awareness.

    Args:
        init_method: Name of the initialization method ('xavier', 'normal', 'kaiming', or 'zero').
        is_expert: Whether this is an expert layer.

    Returns:
        Initialization function that can be applied to a tensor.

    Raises:
        NotImplementedError: If init_method is not supported.
    """
    from megatron.bridge.peft.utils import init_method_normal, init_method_const

    if init_method == "xavier":
        init_fn = nn.init.xavier_normal_
    elif init_method == "normal":
        init_fn = init_method_normal(0.2)
    elif init_method == "kaiming":
        init_fn = init_method_kaiming_uniform_expert(math.sqrt(5), is_expert)
    elif init_method == "zero":
        init_fn = init_method_const(0.0)
    elif init_method == "constant":
        init_fn = init_method_const(val=0.5)
    else:
        raise NotImplementedError(f"init_method should be zero, normal, kaiming or xavier, got {init_method}")
    return init_fn


# ============================================================================
# Patch application
# ============================================================================

def apply_peft_utils_patch():
    """
    Apply PEFT utils patch.

    This patch:
    1. Exports new initialization functions to megatron.bridge.peft.utils module
    2. Patches ParallelLinearAdapter to use expert-aware initialization
    3. Simplifies sharded_state_dict replica_id handling
    """
    global _PEFT_UTILS_PATCH_APPLIED
    if _PEFT_UTILS_PATCH_APPLIED:
        return

    import megatron.bridge.peft.utils as utils_module
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    # Export new functions to utils module
    utils_module._calculate_fan_in_and_fan_out = _calculate_fan_in_and_fan_out
    utils_module._calculate_correct_fan = _calculate_correct_fan
    utils_module.calculate_gain = calculate_gain
    utils_module.kaiming_uniform_ = kaiming_uniform_
    utils_module.init_method_kaiming_uniform_expert = init_method_kaiming_uniform_expert
    utils_module.get_init_fn = get_init_fn
    utils_module.get_init_fn_expert = get_init_fn_expert

    # Add _get_init_fn_expert method to ParallelLinearAdapter
    def _get_init_fn_expert_method(self, init_method: str, is_expert: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization function by method name with expert-awareness."""
        return get_init_fn_expert(init_method, is_expert)

    ParallelLinearAdapter._get_init_fn_expert = _get_init_fn_expert_method

    # Patch _get_init_fn so ParallelLinearAdapter.__init__ picks expert-aware init.
    # __init__ sets self.is_expert before calling self._get_init_fn(...).
    def _patched_get_init_fn(self, init_method: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization function by method name with expert-awareness."""
        return get_init_fn_expert(init_method, getattr(self, "is_expert", False))

    ParallelLinearAdapter._get_init_fn = _patched_get_init_fn

    # Patch sharded_state_dict to simplify replica_id handling
    _original_sharded_state_dict = ParallelLinearAdapter.sharded_state_dict

    def _patched_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """
        Patched sharded_state_dict that simplifies replica_id handling.

        NOTE: When is_expert=True, the adapter is inside SequentialMLP.local_experts.
        SequentialMLP.sharded_state_dict() will handle replica_id adjustment uniformly
        for all tensors (both base weights and adapter weights). We should NOT adjust
        replica_id here to avoid conflicts with SequentialMLP's adjustment.
        """
        sharded_state_dict = {}

        linear_in_sd = self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata)
        linear_out_sd = self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)

        # No replica_id adjustment needed here - let SequentialMLP handle it

        if "linear_fc1" in self.base_linear_name:
            for k, v in linear_out_sd.items():
                if hasattr(v, "axis") and v.axis == 0:
                    v.axis = 1

        # NOTE: Below is a workaround to make sharded_state_dict work with TP.
        # Megatron-Core ColumnParallelLinear sets gather_output to True when TP=1.
        # When saving checkpoint, the sharded_tensor.axis will use the original value (0),
        # but when loading, since gather_output is True, axis will be None.
        # To be compatible with TP > 1, we need to skip replica_id modification here.

        # Handle TP=1 case - axis might need adjustment
        from megatron.core import parallel_state
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if tp_size == 1:
            for sd in [linear_in_sd, linear_out_sd]:
                for k, v in sd.items():
                    if hasattr(v, "axis"):
                        v.axis = None

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict

    ParallelLinearAdapter.sharded_state_dict = _patched_sharded_state_dict

    _PEFT_UTILS_PATCH_APPLIED = True
    print("[Patch] Applied PEFT utils patch with expert-aware initialization")
