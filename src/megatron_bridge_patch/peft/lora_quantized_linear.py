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

"""用于 MoE 专家权重的 INT4 量化线性层。

该模块提供 QuantizedLinear 封装器，将权重以 INT4 打包格式存储，
并在前向计算过程中按需（on-the-fly）进行反量化。

关键设计决策：
- 打包后的 INT4 数据直接存储在 weight.data 中（以 int32 张量形式），
  以便 Transformer Engine 的 CPU 权重 offloading 能够正确地在 CPU 和 GPU 之间迁移数据。
  scale（缩放因子）占用较小，单独存储。
- 不缓存反量化后的权重，以尽量降低 GPU 显存占用。
"""

import torch
import torch.nn as nn
from typing import Optional

from .lora_int4_kernel import quantize_per_column_int4, dequantize_per_column_int4


class _QuantizedLinearBase(nn.Module):
    """INT4 量化线性层的基类，用于执行实际的前反向。

    为所有变体（普通 Linear、ColumnParallelLinear、RowParallelLinear）
    提供共享的量化/反量化/前向计算逻辑。

    子类只需在 __init__ 中设置 _orig_out_features 和 _orig_in_features，
    并调用 super().__init__()。

    打包后的 INT4 数据直接存储在 ``base_linear.weight.data`` 中（以 int32 张量形式）
    """

    def __init__(self, base_linear, group_size=32):
        super().__init__()
        self.base_linear = base_linear
        self.group_size = group_size

        # Scale buffer
        self.register_buffer('weight_scale', None)
        # Shape metadata
        self.register_buffer('weight_shape', None)

        # 跟踪量化的状态
        self._is_quantized = False
        
        self.quantize() # 初始化时直接量化为int4

    @property
    def is_quantized(self):
        return self._is_quantized

    def quantize(self):
        """将权重张量量化为 INT4 打包格式。

        打包后的 int32 数据会原地替换 base_linear.weight.data。

        注意：在替换之前，会将该权重参数的 requires_grad 设为 False，
        因为 PyTorch 要求参与梯度跟踪的参数必须是浮点类型。在 LoRA 训练中，
        基础权重始终是冻结的。
        """
        if self._is_quantized:
            return
        # print(f"===进行int4量化") # DEBUG

        weight = self.base_linear.weight.data.clone()
        self.base_linear.weight.requires_grad = False # LoRA场景下冻结参数
        weight_packed, weight_scale, weight_shape = quantize_per_column_int4(
            weight, group_size=self.group_size
        )

        self.base_linear.weight.data = weight_packed
        self.weight_scale = weight_scale
        self.weight_shape = weight_shape
        self._is_quantized = True

    def dequantize(self):
        """将 INT4 打包权重反量化为 bfloat16。

        不使用缓存，以尽量降低 GPU 显存占用。反量化后的权重
        每次都会根据打包的 int32 数据重新计算得到。
        """
        if not self._is_quantized:
            return self.base_linear.weight.data
        return dequantize_per_column_int4(
            self.base_linear.weight.data,
            self.weight_scale,
            self.weight_shape,
            group_size=self.group_size,
        )

    def forward(self, x, *args, **kwargs):
        """带有按需（on-the-fly）反量化的前向计算。

        对于 TE 层：使用 saved_tensors_hooks 拦截 TE 的 save_for_backward，
        将 bf16 weight 替换为标记以节省显存。在 backward 时按需反量化。
        这样 bf16 weight 仅在 forward/backward 计算的短暂时刻存在，
        训练迭代间全部为 int32 打包数据，最大程度节省显存。

        对于非 TE 层（Megatron 原生层）：使用原 swap-restore 模式。
        """
        if not self._is_quantized:
            return self.base_linear(x, *args, **kwargs)

        # 判断 base_linear 是否为 TE 层
        _is_te = hasattr(self.base_linear, 'te_return_bias')

        if _is_te:
            # print("===进行forward反量化") # DEBUG
            weight_bf16 = self.dequantize() # forward时先执行反量化
            packed_weight = self.base_linear.weight.data
            self.base_linear.weight.data = weight_bf16 

            # 记录 bf16 weight 的 data_ptr，用于在 pack_hook 中识别
            weight_dptr = weight_bf16.data_ptr()

            # 闭包引用，供 hooks 使用
            _quant_base = self

            def _pack_hook(tensor): # forward时触发
                """Forward 时拦截 save_for_backward：将 bf16 weight 替换为标记。"""
                if isinstance(tensor, torch.Tensor) and tensor.data_ptr() == weight_dptr:
                    return ("__int4_packed__",) # 正常TE会保存一个weight(bf16)用于backward，这里保存一个标记
                return tensor

            def _unpack_hook(saved): # backward时触发
                """Backward 时拦截 saved_tensors 访问：按需反量化。

                不缓存反量化结果——每次调用都重新反量化，使 bf16 临时缓冲随
                autograd graph node 执行完毕后立即释放，避免在整个 backward
                过程中持续占用显存。TE linear 的 backward 对每个权重只访问
                一次（仅用于计算 dX = dY @ W^T），无需缓存。
                """
                if isinstance(saved, tuple) and len(saved) == 1 and saved[0] == "__int4_packed__":
                    # print("===进行backward反量化")
                    return _quant_base.dequantize()
                return saved

            with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
                output = self.base_linear(x, *args, **kwargs)

            # Forward 结束后立即还原 int32，bf16 weight 可被 GC 回收
            self.base_linear.weight.data = packed_weight
            return output
        else:
            # 非 TE 层：标准前向传播
            weight_bf16 = self.dequantize()
            packed_weight = self.base_linear.weight.data
            self.base_linear.weight.data = weight_bf16
            output = self.base_linear(x, *args, **kwargs)
            self.base_linear.weight.data = packed_weight
            return output

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None, **kwargs):
        """将 sharded_state_dict 的处理委托给基础线性层。
        对于 PEFT/LoRA 训练，基础权重本身就会被 adapter 的 key 过滤机制排除，因此这里直接返回一个空字典以避免形状问题。adapter 权重会由 ParallelLinearAdapter 单独处理。
        """
        return {}

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """将 state_dict_for_save_checkpoint 委托给底层的 linear 层。
        其原因与 sharded_state_dict 相同——在 PEFT 训练中，由于基础权重会被过滤掉，因此返回空字典。
        """
        return {}


class QuantizedGroupedLinear(nn.Module):
    """INT4 量化 wrapper，用于 TE GroupedLinear (weight0, weight1, ..., weight{N-1})，用于执行实际的前反向。

    与 QuantizedLinear 的区别：TE GroupedLinear 没有单一 .weight，
    而是有 num_gemms 个 weightN 参数，需要逐个处理。
    """

    def __init__(self, base_linear, group_size=32):
        super().__init__()
        self.base_linear = base_linear
        self.group_size = group_size
        self.num_gemms = base_linear.num_gemms
        self._orig_in_features = base_linear.in_features
        self._orig_out_features = base_linear.out_features

        # 为每个 expert 权重注册 scale 和 shape buffer
        for i in range(self.num_gemms):
            self.register_buffer(f'weight_scale_{i}', None)
            self.register_buffer(f'weight_shape_{i}', None)

        # 跟踪量化的状态
        self._is_quantized = False

        self.quantize() # 初始化时直接量化为int4

    @property
    def config(self):
        """转发 base_linear 的 config 属性，供 get_adapter_attributes_from_linear 等函数使用。"""
        return self.base_linear.config

    @property
    def parallel_mode(self):
        """转发 base_linear 的 parallel_mode 属性。"""
        return getattr(self.base_linear, 'parallel_mode', None)

    @property
    def in_features(self):
        return self._orig_in_features

    @property
    def out_features(self):
        return self._orig_out_features

    @property
    def is_quantized(self):
        return self._is_quantized

    def quantize(self):
        """将权重张量量化为 INT4 打包格式。

        打包后的 int32 数据会原地替换 base_linear.weight.data。

        注意：在替换之前，会将该权重参数的 requires_grad 设为 False，
        因为 PyTorch 要求参与梯度跟踪的参数必须是浮点类型。在 LoRA 训练中，
        基础权重始终是冻结的。
        """
        if self._is_quantized:
            return
        # print(f"===进行grouped int4量化") # DEBUG

        for i in range(self.num_gemms): # 遍历这个rank上的所有专家权重
            w_param = getattr(self.base_linear, f'weight{i}')
            w_param.requires_grad = False # LoRA场景下冻结参数
            weight_packed, weight_scale, weight_shape = quantize_per_column_int4(
                w_param.data.clone(), group_size=self.group_size
            )
            w_param.data = weight_packed
            setattr(self, f'weight_scale_{i}', weight_scale)
            setattr(self, f'weight_shape_{i}', weight_shape)

        self._is_quantized = True

    def _dequantize_expert(self, i):
        """反量化第 i 个 expert 的权重。"""
        w_param = getattr(self.base_linear, f'weight{i}')
        scale = getattr(self, f'weight_scale_{i}')
        shape = getattr(self, f'weight_shape_{i}')
        return dequantize_per_column_int4(w_param.data, scale, shape, group_size=self.group_size)

    def forward(self, x, *args, **kwargs):
        """带有按需（on-the-fly）反量化的前向计算。
        原来的TE会在前向最后保存权重用于反向，_pack_hook替换原来的保存权重的函数节省显存，
        _unpack_hook替换反向的拿去权重的函数，直接进行反量化。
        不使用缓存，以尽量降低 GPU 显存占用。反量化后的权重
        每次都会根据打包的 int32 数据重新计算得到。
        """
        if not self._is_quantized:
            return self.base_linear(x, *args, **kwargs)

        # 这里不需要判断是否为TE层，在moe_grouped_gemm=True时，
        # base_linear始终是TEGroupedLinear(te.pytorch.GroupedLinear)
        # 反量化所有 num_gemms 权重
        packed_weights = []
        bf16_data_ptrs = set()
        for i in range(self.num_gemms):
            w_param = getattr(self.base_linear, f'weight{i}')
            packed_weights.append(w_param.data) # 缓存int4的权重
            w_bf16 = self._dequantize_expert(i)
            w_param.data = w_bf16 # 反量化的bf16的权重
            bf16_data_ptrs.add(w_bf16.data_ptr())

        # 闭包引用，供 hooks 使用
        _quant_self = self

        def _pack_hook(tensor): # forward时触发
            """Forward 时拦截 save_for_backward：将 bf16 weight 替换为标记。"""
            if isinstance(tensor, torch.Tensor) and tensor.data_ptr() in bf16_data_ptrs:
                # 找到对应的 expert index
                for j in range(_quant_self.num_gemms):
                    wj = getattr(_quant_self.base_linear, f'weight{j}')
                    if wj.data_ptr() == tensor.data_ptr():
                        return ("__int4_packed_grouped__", j) 
                        # 正常TE会保存一个weight(bf16)用于backward，这里保存一个标记，
                        # 这里返回的是一个tuple，因此_unpack_hook中读取的也是tuple
                return ("__int4_packed_grouped__", -1)
            return tensor

        def _unpack_hook(saved): # backward时触发
            """Backward 时拦截 saved_tensors 访问：按需反量化。

            不缓存反量化结果——每次调用都重新反量化，使 bf16 临时缓冲随
            autograd graph node 执行完毕后立即释放，避免在整个 backward
            过程中持续占用显存。TE linear 的 backward 对每个权重只访问
            一次（仅用于计算 dX = dY @ W^T），无需缓存。
            """
            if isinstance(saved, tuple) and len(saved) == 2 and saved[0] == "__int4_packed_grouped__":
                idx = saved[1]
                if 0 <= idx < _quant_self.num_gemms:
                    return _quant_self._dequantize_expert(idx)
            return saved

        with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
            output = self.base_linear(x, *args, **kwargs)

        # Forward 结束后立即还原 int32，bf16 weight 可被 GC 回收
        for i in range(self.num_gemms):
            w_param = getattr(self.base_linear, f'weight{i}')
            w_param.data = packed_weights[i]

        return output

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None, **kwargs):
        return {}

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return {}


class QuantizedLinear(_QuantizedLinearBase):
    """Quantized wrapper for nn.Linear layers."""

    def __init__(self, base_linear, group_size=32):
        self._orig_out_features = base_linear.out_features
        self._orig_in_features = base_linear.in_features
        super().__init__(base_linear, group_size=group_size)

    @property
    def config(self):
        """转发 base_linear 的 config 属性，供 get_adapter_attributes_from_linear 等函数使用。"""
        return self.base_linear.config

    @property
    def parallel_mode(self):
        """转发 base_linear 的 parallel_mode 属性。"""
        return getattr(self.base_linear, 'parallel_mode', None)

    @property
    def in_features(self):
        return self._orig_in_features

    @property
    def out_features(self):
        return self._orig_out_features

    @property
    def bias(self):
        return self.base_linear.bias
