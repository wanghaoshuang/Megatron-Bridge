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
针对 TE GroupedLinear 的 按专家（per-expert）LoRA 实现（在 moe_grouped_gemm=True 时）。

当 moe_grouped_gemm=True 时，MoE 专家层使用的是 TEGroupedMLP，其结构是扁平的（flat）：

linear_fc1 = TEColumnParallelGroupedLinear
linear_fc2 = TERowParallelGroupedLinear

该模块提供了 GroupedLoRALinear，用于对 TEGroupedLinear 进行封装，从而为每个专家（expert）提供独立的 LoRA adapter，
同时保持基础计算仍然使用 grouped GEMM 来实现加速。

核心设计：

基础权重前向计算：使用 grouped GEMM（保持不变，性能高）
LoRA adapter 前向计算：使用 torch.nn.functional.grouped_mm（stacked weight tensor [E, in, rank]）

torch.nn.functional.grouped_mm API（PyTorch >= 2.4）：
  output = torch.nn.functional.grouped_mm(A, B, offs=offs)
  - A:    [total_tokens, K]          contiguous bf16/fp16
  - B:    [num_experts, K, N]        contiguous bf16/fp16，N 需是 8 的倍数（bf16 16-byte 对齐）
  - offs: [num_experts] int32        每个 expert 在 A 中的 exclusive end row index
          即 offs = cumsum(tokens_per_expert)，长度 E（不是 E+1）

backward 已验证（PyTorch 2.11）：grad_inp / grad_A / grad_B 均正确。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Picklable callables for ShardedTensorFactory (must be module-level for pickle)
# ---------------------------------------------------------------------------

class _TorchBackendBuildFn:
    """Picklable build_fn for torch backend ShardedTensorFactory."""

    def __init__(self, num_experts, ep_rank, num_global_experts, sharded_offsets,
                 singleton, tp_axis_2d, etp_size, etp_rank, edp_rank, is_swiglu=False):
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.num_global_experts = num_global_experts
        self.sharded_offsets = sharded_offsets
        self.singleton = singleton
        self.tp_axis_2d = tp_axis_2d
        self.etp_size = etp_size
        self.etp_rank = etp_rank
        self.edp_rank = edp_rank
        self.is_swiglu = is_swiglu

    def __call__(self, key, data, replica_id, flattened_range):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
        result = {}
        for i in range(self.num_experts):
            expert_global_idx = self.ep_rank * self.num_experts + i
            if self.singleton:
                expert_sharded_offsets = self.sharded_offsets
            else:
                expert_sharded_offsets = (
                    *self.sharded_offsets,
                    (len(self.sharded_offsets), expert_global_idx, self.num_global_experts),
                )

            slice_data = data[i]
            rank_offsets = list(expert_sharded_offsets)
            prepend_axis_num = len(expert_sharded_offsets)
            if self.etp_size > 1:
                rank_offsets.append((self.tp_axis_2d + prepend_axis_num, self.etp_rank, self.etp_size))

            if not self.is_swiglu:
                sh_ten = ShardedTensor.from_rank_offsets(
                    key, slice_data, *rank_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
                if (hasattr(sh_ten, 'replica_id')
                        and isinstance(sh_ten.replica_id, tuple)
                        and len(sh_ten.replica_id) == 3):
                    sh_ten.replica_id = (*sh_ten.replica_id[:2], self.edp_rank)
                result[f'{key}.expert{i}'] = sh_ten
            else:
                # Inline swiglu split: split along dim 0 into W and V halves,
                # producing ShardedTensor directly (no nested ShardedTensorFactory).
                sh_ten = ShardedTensor.from_rank_offsets(
                    key, slice_data, *rank_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                )
                swiglu_axis = 0
                local_axis_size = slice_data.shape[swiglu_axis]
                sg_rank_offset = sh_ten.global_offset[swiglu_axis + prepend_axis_num] // local_axis_size
                sg_axis_frag = sh_ten.axis_fragmentations[swiglu_axis + prepend_axis_num]

                tensor_w, tensor_v = torch.chunk(slice_data, 2, dim=swiglu_axis)

                if self.singleton:
                    offset_w = (swiglu_axis + prepend_axis_num, sg_rank_offset, sg_axis_frag)
                    offset_v = (swiglu_axis + prepend_axis_num, sg_rank_offset, sg_axis_frag)
                    w_key = f'{key}_w'
                    v_key = f'{key}_v'
                else:
                    offset_w = (swiglu_axis + prepend_axis_num, sg_rank_offset, sg_axis_frag * 2)
                    offset_v = (swiglu_axis + prepend_axis_num, sg_rank_offset + sg_axis_frag, sg_axis_frag * 2)
                    w_key = key
                    v_key = key

                sh_ten_w = ShardedTensor.from_rank_offsets(
                    w_key, tensor_w, *expert_sharded_offsets, offset_w,
                    replica_id=replica_id, prepend_axis_num=prepend_axis_num,
                )
                sh_ten_v = ShardedTensor.from_rank_offsets(
                    v_key, tensor_v, *expert_sharded_offsets, offset_v,
                    replica_id=replica_id, prepend_axis_num=prepend_axis_num,
                )
                if (hasattr(sh_ten_w, 'replica_id')
                        and isinstance(sh_ten_w.replica_id, tuple)
                        and len(sh_ten_w.replica_id) == 3):
                    sh_ten_w.replica_id = (*sh_ten_w.replica_id[:2], self.edp_rank)
                    sh_ten_v.replica_id = (*sh_ten_v.replica_id[:2], self.edp_rank)
                result[f'{key}.expert{i}'] = [sh_ten_w, sh_ten_v]
        return result


class _TorchBackendMergeFn:
    """Picklable merge_fn for torch backend ShardedTensorFactory."""

    def __init__(self, num_experts):
        self.num_experts = num_experts

    def __call__(self, sub_state_dict):
        tensors = []
        for i in range(self.num_experts):
            for k, v in sub_state_dict.items():
                if f'.expert{i}' in k:
                    # swiglu case: v is a list [tensor_w, tensor_v] → cat back
                    if isinstance(v, list):
                        tensors.append(torch.cat(v, dim=0))
                    else:
                        tensors.append(v)
                    break
        if len(tensors) == self.num_experts:
            return torch.stack(tensors)
        values = list(sub_state_dict.values())
        processed = [torch.cat(v, dim=0) if isinstance(v, list) else v for v in values]
        return torch.stack(processed)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_te_grouped_linear(module: nn.Module) -> bool:
    """检查一个模块是否为 TE GroupedLinear（其具有 weight0..weightN 这样的多权重结构，而不是单一的 .weight）。"""
    return (
        hasattr(module, 'num_gemms')
        and hasattr(module, 'weight0')
        and not hasattr(module, 'weight')
    )


def is_quantized_grouped_linear(module: nn.Module) -> bool:
    """检查一个模块是否为 QuantizedGroupedLinear 的封装器（wrapper）。"""
    from megatron_bridge_patch.peft.lora_quantized_linear import QuantizedGroupedLinear
    return isinstance(module, QuantizedGroupedLinear)


def get_num_experts_from_module(module: nn.Module) -> int:
    """从 TE GroupedLinear 或其封装器中获取专家数量（num_gemms）。"""
    if is_quantized_grouped_linear(module):
        return module.base_linear.num_gemms
    if is_te_grouped_linear(module):
        return module.num_gemms
    raise ValueError(f"Cannot determine num_experts from {type(module)}")


# ---------------------------------------------------------------------------
# GroupedLoRALinear
# ---------------------------------------------------------------------------

class GroupedLoRALinear(nn.Module):
    """用于 TE GroupedLinear 的 LoRA 封装器，支持按专家独立的 adapter。

    基础权重的计算仍然使用 grouped GEMM 来实现加速；
    LoRA adapter 的计算根据 backend 选择：
      - "te": 使用 TE GroupedLinear（gemm_A/gemm_B 是 te.GroupedLinear 子模块）
      - "torch": 使用 torch.nn.functional.grouped_mm（gemm_A/gemm_B 是 nn.Parameter）

    参数说明：

    to_wrap：原始的 TEGroupedLinear（或 QuantizedGroupedLinear）模块
    gemm_A：LoRA A（TE GroupedLinear 或 Tensor[E, in, dim]，取决于 backend）
    gemm_B：LoRA B（TE GroupedLinear 或 Tensor[E, dim, out]，取决于 backend）
    num_experts：专家数量（即 num_gemms，local）
    alpha：LoRA 缩放因子
    dim：LoRA rank
    linear_name：当前线性层的全限定名（用于 state_dict 键映射）
    input_is_parallel：基础层是否为行并行（影响 state_dict 的 TP axis 映射）
    backend："te" 或 "torch"
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        gemm_A,
        gemm_B,
        num_experts: int,
        alpha: float,
        dim: int,
        linear_name: str,
        input_is_parallel: bool = False,
        backend: str = "te",
    ) -> None:
        super().__init__()
        self.to_wrap = to_wrap
        self.backend = backend

        if backend == "te":
            # gemm_A / gemm_B 是 te.GroupedLinear 子模块
            self.gemm_A = gemm_A
            self.gemm_B = gemm_B
        else:
            # gemm_A / gemm_B 是 stacked weight tensors，存为 nn.Parameter
            self.gemm_A = nn.Parameter(gemm_A)  # [E, in_features, dim]
            self.gemm_B = nn.Parameter(gemm_B)  # [E, dim, out_features]

        self._adapter_enabled = True
        self.num_experts = num_experts
        self.alpha = alpha
        self.dim = dim
        self.linear_name = linear_name
        self.input_is_parallel = input_is_parallel

    def enable_adapter_layers(self) -> None:
        """Enable the adapter layers, allowing them to contribute to the forward pass output."""
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        """Disable the adapter layers, making the forward pass return only the base module output."""
        self._adapter_enabled = False

    def base_linear_forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """执行基础线性层的前向计算（grouped GEMM）。

        返回值为 (linear_output, bias, layernorm_output)。
        对于 TEGroupedLinear，layernorm_output == x（不包含 layernorm）。
        """
        linear_output = self.to_wrap(x, *args, **kwargs)
        assert isinstance(linear_output, tuple), (
            f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        )

        bias = None
        layernorm_output = x

        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output

        return linear_output, bias, layernorm_output

    def _build_offs(
        self, tokens_per_expert: Union[List[int], torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """构造 torch.nn.functional.grouped_mm 所需的 offs tensor。

        offs[i] = expert i 在输入中的 exclusive end row index
                = cumsum(tokens_per_expert)，长度 E（不是 E+1）
        """
        if isinstance(tokens_per_expert, (list, tuple)):
            tpe = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device)
        else:
            tpe = tokens_per_expert.to(dtype=torch.int32, device=device)
        return torch.cumsum(tpe, dim=0).int()

    def forward(
        self, x: torch.Tensor, tokens_per_expert, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """基础 grouped GEMM + LoRA 前向传播。

        参数：
        x：输入张量 [total_tokens, hidden_size]
        tokens_per_expert：每个专家的 token 数量（list of int 或 int32 tensor）
        """
        # 1. 基础前向，走 grouped GEMM，一次计算所有专家的输出
        linear_output, bias, layernorm_output = self.base_linear_forward(
            x, tokens_per_expert, *args, **kwargs
        )
        if not self._adapter_enabled:
            return linear_output, bias

        # 2. LoRA adapter forward
        if self.backend == "te":
            # TE GroupedLinear path: call gemm_A/gemm_B as modules
            intermediate = self.gemm_A(layernorm_output, tokens_per_expert)
            adapter_output = self.gemm_B(intermediate, tokens_per_expert)
        else:
            # torch.nn.functional.grouped_mm path
            inp = layernorm_output.contiguous()
            offs = self._build_offs(tokens_per_expert, inp.device)
            intermediate = torch.nn.functional.grouped_mm(inp, self.gemm_A, offs=offs)
            adapter_output = torch.nn.functional.grouped_mm(intermediate, self.gemm_B, offs=offs)

        adapter_output = adapter_output.reshape(linear_output.shape)

        # Seg0 bypass: zero adapter output at seg0 positions if mask is set
        seg1_mask = getattr(self, '_seg1_mask_permuted', None)
        if seg1_mask is not None:
            adapter_output = adapter_output * seg1_mask.unsqueeze(-1).to(adapter_output.dtype)

        # Fused scale+add (out-of-place): computes linear_output + (alpha/dim) * adapter_output
        # in a single kernel, avoiding a separate allocation for the scaled adapter_output.
        # Cannot use in-place add_ because linear_output is the output of a TE custom Function
        # (_GroupedLinearBackward) which forbids in-place modification.
        return torch.add(linear_output, adapter_output, alpha=(self.alpha / self.dim)), bias

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """LoRA 训练中只保存 adapter 权重，不保存 base 权重（base 权重是冻结的）。

        键格式与 SequentialMLP 格式对齐：
          local_experts.{i}.adapter.linear_in.weight   → gemm_A[i]
          local_experts.{i}.adapter.linear_out.weight  → gemm_B[i]
        """
        if destination is None:
            destination = {}

        for i in range(self.num_experts):
            a_key = f"{prefix}local_experts.{i}.adapter.linear_in.weight"
            b_key = f"{prefix}local_experts.{i}.adapter.linear_out.weight"
            if self.backend == "te":
                destination[a_key] = getattr(self.gemm_A, f'weight{i}')
                destination[b_key] = getattr(self.gemm_B, f'weight{i}')
            else:
                destination[a_key] = self.gemm_A[i] if keep_vars else self.gemm_A[i].detach()
                destination[b_key] = self.gemm_B[i] if keep_vars else self.gemm_B[i].detach()

        return destination

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """从 state_dict 加载 LoRA 权重。

        将 local_experts.{i}.adapter.linear_in/out.weight 映射到 gemm_A/B。

        注意 shape 约定差异：
          - torch 后端: gemm_A[i] shape = [in_features, rank]（我们的 layout）
          - TE 后端: weight{i} shape = [rank, in_features]（TE 转置约定）
        如果 checkpoint 是用 torch 后端保存的，加载到 TE 后端时需要转置。
        """
        for i in range(self.num_experts):
            a_key = f"{prefix}local_experts.{i}.adapter.linear_in.weight"
            b_key = f"{prefix}local_experts.{i}.adapter.linear_out.weight"

            if a_key in state_dict:
                w = state_dict.pop(a_key)
                if self.backend == "te":
                    target = getattr(self.gemm_A, f'weight{i}')
                    # TE weight shape: [rank, in_features]; ckpt may be [in_features, rank]
                    if w.shape != target.shape and w.shape == target.shape[::-1]:
                        w = w.t()
                    target.data.copy_(w)
                else:
                    self.gemm_A.data[i].copy_(w)
            elif strict:
                missing_keys.append(a_key)

            if b_key in state_dict:
                w = state_dict.pop(b_key)
                if self.backend == "te":
                    target = getattr(self.gemm_B, f'weight{i}')
                    # TE weight shape: [out_features, rank]; ckpt may be [rank, out_features]
                    if w.shape != target.shape and w.shape == target.shape[::-1]:
                        w = w.t()
                    target.data.copy_(w)
                else:
                    self.gemm_B.data[i].copy_(w)
            elif strict:
                missing_keys.append(b_key)

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict[str, Any]] = None,
    ) -> ShardedStateDict:
        """采用 SequentialMLP 格式的 adapter 键的分片（sharded）state dict。

        在 LoRA 训练中，只会保存 adapter 的权重（基础权重是冻结的，由预训练模型提供）。

        ShardedTensor.key 的格式对齐原始 TEGroupedMLP.sharded_state_dict 的期望：
          - 非 singleton: {name}.adapter.xxx
          - singleton: {expert_global_idx}.adapter.xxx

        对于 torch backend，使用 ShardedTensorFactory 包装 3D nn.Parameter，
        使得 factory.data 指向实际参数（optimizer id() 匹配），
        build_fn 将其拆分为 per-expert 2D ShardedTensor（checkpoint 兼容）。
        """
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        linear_name = prefix.rstrip(".")

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        edp_rank = parallel_state.get_expert_data_parallel_rank()

        sharded_state_dict = {}
        num_global_experts = ep_size * self.num_experts
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)

        etp_group = parallel_state.get_expert_tensor_parallel_group()

        if self.backend != "te":
            # Torch backend: gemm_A/gemm_B 是 3D nn.Parameter [E, in, rank] / [E, rank, out]
            # 使用 ShardedTensorFactory 使 .data 指向实际参数（解决 optimizer id() 匹配问题）
            sharded_state_dict.update(self._sharded_state_dict_torch_backend(
                linear_name, sharded_offsets, ep_size, ep_rank, edp_rank,
                num_global_experts, singleton_local_shards, etp_group,
            ))
            return sharded_state_dict

        for i in range(self.num_experts):
            expert_global_idx = ep_rank * self.num_experts + i

            if singleton_local_shards:
                expert_sharded_offsets = sharded_offsets
            else:
                expert_sharded_offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), expert_global_idx, num_global_experts),
                )

            adapter_prefix = f"{linear_name}.local_experts.{i}.adapter."

            weight_a = getattr(self.gemm_A, f'weight{i}')
            weight_b = getattr(self.gemm_B, f'weight{i}')

            temp_state_dict = {
                'linear_in.weight': weight_a,
                'linear_out.weight': weight_b,
            }

            if self.input_is_parallel:
                tp_axis_map = {'linear_in.weight': 1, 'linear_out.weight': 0}
            else:
                tp_axis_map = {'linear_in.weight': 0, 'linear_out.weight': 0}

            adapter_sd = make_sharded_tensors_for_checkpoint(
                temp_state_dict,
                adapter_prefix,
                tp_axis_map,
                expert_sharded_offsets,
                tp_group=etp_group,
            )

            if 'linear_fc1' in self.linear_name:
                for k, v in adapter_sd.items():
                    if 'linear_out.weight' in k:
                        adapter_sd[k] = apply_swiglu_sharded_factory(
                            v, expert_sharded_offsets, singleton_local_shards
                        )

            if singleton_local_shards:
                replace_prefix_for_sharding(
                    adapter_sd,
                    f"{linear_name}.local_experts.{i}.",
                    f"{expert_global_idx}.",
                )
            else:
                replace_prefix_for_sharding(
                    adapter_sd,
                    f"{linear_name}.local_experts.{i}.",
                    f"{linear_name}.",
                )

            for k, sh_ten in adapter_sd.items():
                if hasattr(sh_ten, 'replica_id') and len(sh_ten.replica_id) == 3:
                    if singleton_local_shards and '.adapter.' in k:
                        sh_ten.replica_id = (0, 0, edp_rank)
                    else:
                        sh_ten.replica_id = (*sh_ten.replica_id[:2], edp_rank)

            sharded_state_dict.update(adapter_sd)

        return sharded_state_dict

    def _sharded_state_dict_torch_backend(
        self, linear_name, sharded_offsets, ep_size, ep_rank, edp_rank,
        num_global_experts, singleton_local_shards, etp_group,
    ) -> ShardedStateDict:
        """Torch backend: 用 ShardedTensorFactory 包装 3D 参数。

        factory.data = self.gemm_A / self.gemm_B（实际 nn.Parameter），
        build_fn 将 3D tensor 拆分为 per-expert 2D ShardedTensor。
        这样 optimizer 的 id(param) 匹配能正确工作。
        """
        from megatron.core.dist_checkpointing.mapping import (
            ShardedTensor, ShardedTensorFactory,
        )
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
        from megatron.core import parallel_state

        etp_size = parallel_state.get_expert_tensor_parallel_world_size()
        etp_rank = parallel_state.get_expert_tensor_parallel_rank()

        sharded_state_dict = {}

        for param, weight_name in [
            (self.gemm_A, 'linear_in.weight'),
            (self.gemm_B, 'linear_out.weight'),
        ]:
            # 确定 TP axis（在 2D slice 上的 axis）
            if weight_name == 'linear_in.weight':
                tp_axis_2d = 1 if self.input_is_parallel else 0
            else:
                tp_axis_2d = 0

            # Capture variables for closures
            _num_experts = self.num_experts
            _ep_rank = ep_rank
            _num_global_experts = num_global_experts
            _sharded_offsets = sharded_offsets
            _singleton = singleton_local_shards
            _linear_name = linear_name
            _weight_name = weight_name
            _tp_axis_2d = tp_axis_2d
            _etp_size = etp_size
            _etp_rank = etp_rank
            _edp_rank = edp_rank
            # Swiglu split needed for linear_fc1.adapter.linear_out.weight to match
            # TE backend checkpoint format. Cannot call apply_swiglu_sharded_factory here
            # because it creates nested ShardedTensorFactory that apply_factories can't expand.
            _is_swiglu = ('linear_fc1' in self.linear_name and weight_name == 'linear_out.weight')

            _build_fn = _TorchBackendBuildFn(
                _num_experts, _ep_rank, _num_global_experts, _sharded_offsets,
                _singleton, _tp_axis_2d, _etp_size, _etp_rank, _edp_rank, _is_swiglu,
            )
            _merge_fn = _TorchBackendMergeFn(_num_experts)

            # Factory key: use the adapter key format
            factory_key = f"{linear_name}.adapter.{weight_name}"
            # Use a unique Python dict key for each param
            dict_key = f"{linear_name}.adapter.{weight_name}"

            factory = ShardedTensorFactory(
                key=factory_key,
                data=param,
                build_fn=_build_fn,
                merge_fn=_merge_fn,
                replica_id=(0, 0, edp_rank),
            )
            sharded_state_dict[dict_key] = factory

        return sharded_state_dict
