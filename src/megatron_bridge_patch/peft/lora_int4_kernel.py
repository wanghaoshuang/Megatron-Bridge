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

"""用于 LoRA 微调中 MoE 专家权重的 INT4 的kernel实现。

该模块提供基于 Triton 加速的函数，用于将 MoE 专家权重量化为 INT4 格式，
并在计算时将其反量化回 bfloat16。

主要特性：
- 按列进行对称量化，支持可配置的分组大小
- 使用打包的 INT4 存储（每个 int32 存储 8 个值），提高内存效率
- 使用 Triton 内核实现 GPU 加速的反量化
"""

import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


# ============================================================================
# Triton Kernels
# ============================================================================

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_ROW': 32, 'BLOCK_PACKED_COL': 16}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROW': 64, 'BLOCK_PACKED_COL': 16}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROW': 32, 'BLOCK_PACKED_COL': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROW': 64, 'BLOCK_PACKED_COL': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROW': 128, 'BLOCK_PACKED_COL': 16}, num_stages=4, num_warps=8),
        ],
        key=['M', 'packed_n', 'num_groups', 'GROUP_SIZE'],
    )
    @triton.jit
    def _dequantize_int4_kernel(
        packed_ptr,
        scale_ptr,
        out_ptr,
        M,
        packed_n,
        num_groups,
        GROUP_SIZE: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
        BLOCK_PACKED_COL: tl.constexpr,
    ):
        """Triton 内核：将 INT4 打包权重反量化为 bf16。

        每个程序（program）处理一个 tile，尺寸为 [BLOCK_ROW 行, BLOCK_PACKED_COL 个打包列]。
        每个打包列包含 8 个 INT4 值，因此输出 tile 的宽度为 BLOCK_PACKED_COL * 8。

        对于每个打包的 int32 值，会提取其中的 8 个半字节（nibble），将其转换为有符号数，
        再乘以对应分组的缩放因子（per-group scale），并以 bf16 格式存储。
        """
        pid_m = tl.program_id(0)
        pid_pc = tl.program_id(1)

        row_start = pid_m * BLOCK_ROW # 当前线程块负责哪一块数据
        pc_start = pid_pc * BLOCK_PACKED_COL

        row_offs = row_start + tl.arange(0, BLOCK_ROW) # 构造索引
        pc_offs = pc_start + tl.arange(0, BLOCK_PACKED_COL)

        row_mask = row_offs < M # 边界保护，最后一块可能不满
        pc_mask = pc_offs < packed_n

        # 读取int32的packed数据，每个元素是一个int32: [BLOCK_ROW, BLOCK_PACKED_COL]
        packed_ptrs = packed_ptr + row_offs[:, None] * packed_n + pc_offs[None, :]
        packed_vals = tl.load(packed_ptrs, mask=row_mask[:, None] & pc_mask[None, :], other=0)

        # int4数据拆分
        for i in range(8):
            # 取第i个4bit
            nibble = (packed_vals >> (i * 4)) & 0xF
            # 转成有符号数 [-8, 7]
            signed_val = nibble.to(tl.float32) - 8.0

            # 确认第i列属于哪个group
            actual_col = pc_offs * 8 + i  # [BLOCK_PACKED_COL]
            group_idx = actual_col // GROUP_SIZE  # [BLOCK_PACKED_COL]

            # 取这个group的scale: [BLOCK_ROW, BLOCK_PACKED_COL]
            scale_ptrs = scale_ptr + row_offs[:, None] * num_groups + group_idx[None, :]
            scale_val = tl.load(scale_ptrs, mask=row_mask[:, None] & pc_mask[None, :], other=1.0).to(tl.float32)

            # 反量化
            result = signed_val * scale_val

            # 写回bf16
            N = packed_n * 8
            out_ptrs = out_ptr + row_offs[:, None] * N + actual_col[None, :]
            tl.store(out_ptrs, result.to(tl.bfloat16), mask=row_mask[:, None] & pc_mask[None, :])


def _dequantize_int4_triton(
    weight_packed,
    weight_scale,
    out_features,
    in_features,
    group_size=32,
):
    """Triton-accelerated INT4 dequantization."""
    M = out_features
    packed_n = weight_packed.shape[1]
    num_groups = weight_scale.shape[1]
    # 保存反量化的权重
    output = torch.empty(M, in_features, dtype=torch.bfloat16, device=weight_packed.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_ROW']),
        triton.cdiv(packed_n, meta['BLOCK_PACKED_COL']),
    )
    _dequantize_int4_kernel[grid](
        weight_packed, weight_scale, output,
        M=M, packed_n=packed_n, num_groups=num_groups,
        GROUP_SIZE=group_size,
    )

    return output


# ============================================================================
# PyTorch implementations (used for quantize and as dequantize fallback)
# ============================================================================

def _quantize_int4_torch(weight, group_size=32):
    """PyTorch INT4 quantization (scale computation + packing)."""
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    w = weight.float()
    w_grouped = w.view(out_features, num_groups, group_size)

    group_max = w_grouped.abs().amax(dim=-1, keepdim=True)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)

    w_q = (w_grouped / scale).round().clamp(-8, 7)

    w_q_flat = w_q.view(out_features, -1)
    w_q_unsigned = (w_q_flat + 8).to(torch.uint8)

    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"
    w_q_reshaped = w_q_unsigned.view(out_features, in_features // 8, 8).to(torch.int32)
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)

    shifts = torch.arange(8, device=weight.device) * 4
    for i in range(8):
        packed |= (w_q_reshaped[:, :, i] & 0xF) << (shifts[i])

    weight_packed = packed
    weight_scale = scale.squeeze(-1).to(torch.float16)

    return weight_packed, weight_scale


def _dequantize_int4_torch( # torch实现的反量化，尽量用triton实现的，因为反量化要执行很多次，量化只执行一次
    weight_packed,
    weight_scale,
    out_features,
    in_features,
    group_size=32,
):
    """PyTorch fallback for INT4 dequantization."""
    shifts = torch.arange(8, device=weight_packed.device) * 4
    packed_unsqueezed = weight_packed.unsqueeze(-1)
    unpacked = ((packed_unsqueezed >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(out_features, in_features)
    unpacked = unpacked - 8

    scale = weight_scale.float().view(out_features, -1)
    num_groups = scale.shape[1]
    elements_per_group = in_features // num_groups
    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)

    result = unpacked * scale_expanded
    return result.to(torch.bfloat16)


# ============================================================================
# Public API
# ============================================================================

def quantize_per_column_int4(
    weight,
    group_size=32,
):
    """将 bfloat16/float16 权重量化为 INT4 打包格式。

    参数：
        weight: 输入权重张量，形状为 [out_features, in_features]
        group_size: 每个量化分组的元素数量（默认：32）

    返回：
        weight_packed: 将 INT4 数值打包到 int32 张量中
                    形状：[out_features, in_features // 8]
        weight_scale: 每个分组对应的缩放因子
                    形状：[out_features, num_groups]
        weight_shape: 原始张量形状（int64 张量）
                    形状：[2]
    """
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

    assert in_features % group_size == 0, (
        f"in_features={in_features} must be divisible by group_size={group_size}"
    )
    assert in_features % 8 == 0, (
        f"in_features={in_features} must be divisible by 8"
    )

    weight_packed, weight_scale = _quantize_int4_torch(weight, group_size)

    return weight_packed, weight_scale, weight_shape


def dequantize_per_column_int4(
    weight_packed,
    weight_scale,
    weight_shape=None,
    group_size=32,
):
    """将 INT4 打包权重反量化为 bfloat16。

    在可用时使用 Triton 内核进行 GPU 加速；
    在 CPU 上或未安装 Triton 时回退到 PyTorch 实现。
    不进行缓存——每次调用都会重新计算，以节省内存。

    参数：
        weight_packed: INT4 打包后的权重，形状为 [out_features, in_features // 8]
        weight_scale: 每个分组对应的缩放因子，形状为 [out_features, num_groups]
        weight_shape: 原始形状 [2]（可选）
        group_size: 每个量化分组的元素数量

    返回：
        反量化后的权重张量，形状为 [out_features, in_features]，数据类型为 bfloat16
    """
    out_features = weight_packed.shape[0]
    in_features = weight_packed.shape[1] * 8

    if weight_shape is not None:
        out_features = weight_shape[0].item()
        in_features = weight_shape[1].item()

    if HAVE_TRITON and weight_packed.is_cuda:
        return _dequantize_int4_triton(weight_packed, weight_scale, out_features, in_features, group_size)
    else:
        return _dequantize_int4_torch(weight_packed, weight_scale, out_features, in_features, group_size)
