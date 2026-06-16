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

"""FlashMaskAttention - a sparse-attention drop-in replacement for Megatron's
``DotProductAttention``.

This is a placeholder implementation that uses ``F.scaled_dot_product_attention``
with a custom (currently causal) bool mask. It preserves the exact input/output
contract of ``megatron.core.transformer.dot_product_attention.DotProductAttention``
so it can be swapped in at runtime without changing the surrounding
``SelfAttention`` module or its weights.

Tensor layout convention (matches Megatron):
    query / key / value : [s, b, np, hn]   (np = num_heads_per_tp_partition)
    output              : [s, b, h]        (h  = np * hn)

Replace the inner SDPA call with a real FlashMask CUDA / Triton kernel later
without touching call sites.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


def _build_sparse_prepend_memory_pattern(
    seq_len: int,
    group_size: int,
    segment_size: int,
    device: torch.device,
) -> Tensor:
    """Build the sparse attention mask in *prepend-memory* layout.

    ``seq_len`` is the number of **original** (non-memory) tokens.  The total
    mask dimension is ``seq_len + seq_len // group_size`` because one memory
    token is inserted after every ``group_size`` original tokens.

    The mask matrix rows/columns are ordered as::

        [m_0, m_1, ..., m_{J-1}, t_0, t_1, ..., t_{I-1}]

    where ``I = seq_len`` and ``J = seq_len // group_size``.  This matches the
    ``sparse_mask_v2.py::build_sparse_mask`` permutation.

    Mask rules (from ``sparse_mask_v2.py::_attend``):

    ① t -> t  :  sliding window   qi - (s-1) <= ki <= qi
    ② t -> m  :  only memory before the window   (ki+1)*g <= qi - (s-1)
    ③ m -> t  :  same segment & before group end   seg_m(qi)==seg_t(ki) and ki < (qi+1)*g
    ④ m -> m  :  causal   ki < qi

    Args:
        seq_len:     number of **original** tokens (excluding memory tokens).
        group_size:  ``g`` — tokens per compression group.
        segment_size: ``s`` — regular tokens per segment.  Must divide by ``group_size``.
        device:      target torch device.

    Returns:
        Bool tensor of shape ``[I+J, I+J]`` where ``True`` means *attend*.
    """
    s = segment_size
    g = group_size
    assert s % g == 0, f"segment_size ({s}) must be divisible by group_size ({g})"
    assert seq_len % g == 0, f"seq_len ({seq_len}) must be divisible by group_size ({g})"

    I = seq_len                       # number of regular (original) tokens
    J = seq_len // g                  # number of memory tokens
    total_len = I + J

    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    # Index grids -------------------------------------------------------
    # Memory block: rows [0, J), cols [0, J)
    qi_m = torch.arange(J, device=device).unsqueeze(1)  # [J, 1]
    kj_m = torch.arange(J, device=device).unsqueeze(0)  # [1, J]

    # Regular-token block: rows [J, total_len), cols [J, total_len)
    qi_t = torch.arange(I, device=device).unsqueeze(1)  # [I, 1]
    ki_t = torch.arange(I, device=device).unsqueeze(0)  # [1, I]

    # Cross blocks
    ki_t_for_mq = torch.arange(I, device=device).unsqueeze(0)  # [1, I]  (m row, t col)
    kj_m_for_tq = torch.arange(J, device=device).unsqueeze(0)  # [1, J]  (t row, m col)

    # ④ m -> m : causal  (ki < qi)
    mask[:J, :J] = kj_m < qi_m

    # ③ m -> t : same segment & ki < (qi+1)*g
    seg_m = (qi_m * g) // s       # segment of memory query
    seg_t_col = ki_t_for_mq // s  # segment of token key
    mask[:J, J:] = (seg_m == seg_t_col) & (ki_t_for_mq < (qi_m + 1) * g)

    # ② t -> m : (ki+1)*g <= qi - (s-1)
    mask[J:, :J] = (kj_m_for_tq + 1) * g <= qi_t - (s - 1)

    # ① t -> t : sliding window  qi - (s-1) <= ki <= qi
    mask[J:, J:] = (qi_t - (s - 1) <= ki_t) & (ki_t <= qi_t)

    return mask


class FlashMaskAttention(MegatronModule):
    """Drop-in replacement for ``DotProductAttention`` using SDPA + sparse mask.

    Interface (forward signature, return shape) is intentionally identical to
    ``megatron.core.transformer.dot_product_attention.DotProductAttention``
    so it can be assigned in place of ``self_attention.core_attention``.

    Subclasses must override ``build_flash_mask(seq_len_q, seq_len_k, device, causal)``
    to return a bool tensor where ``True`` means *attend*.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attn_mask_type: AttnMaskType,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.config = config
        self.attn_mask_type = attn_mask_type

        assert config.context_parallel_size == 1, (
            "FlashMaskAttention placeholder does not support context parallel; "
            "use TE attention path or extend the kernel."
        )

        # Match DotProductAttention's per-partition bookkeeping so callers
        # that introspect these attributes still work.
        if pg_collection is None:
            from megatron.core.process_groups_config import ProcessGroupCollection

            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        self.pg_collection = pg_collection
        tp_size = pg_collection.tp.size()

        projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_partition = divide(projection_size, tp_size)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, tp_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, tp_size)

        self.attention_dropout = (
            attention_dropout if attention_dropout is not None else config.attention_dropout
        )
        self.softmax_scale = softmax_scale  # if None, SDPA uses 1/sqrt(d_k)

    def build_flash_mask(self,
                        seq_len_q: int,
                        seq_len_k: int,
                        device: torch.device,
                        causal: bool = False) -> Tensor:
        """Build the attention mask for this module.

        Args:
            seq_len_q: query sequence length.
            seq_len_k: key sequence length.
            device: target torch device.
            causal: whether to build a causal mask (default ``False``).
        """
        pass

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Compute SDPA with a (sparse) bool mask.

        query/key/value: ``[s, b, np_or_ng, hn]``. Returns ``[s, b, h]``.
        """
        assert packed_seq_params is None, "FlashMaskAttention does not support packed sequences yet."
        assert attention_bias is None, "FlashMaskAttention does not support attention_bias."

        # GQA: expand kv heads to match q heads.
        repeat = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        sq, b, np_, hn = query.shape
        sk = key.size(0)

        # [s, b, np, hn] -> [b, np, s, hn]
        q_ = query.permute(1, 2, 0, 3).contiguous()
        k_ = key.permute(1, 2, 0, 3).contiguous()
        v_ = value.permute(1, 2, 0, 3).contiguous()

        causal = self.attn_mask_type == AttnMaskType.causal or attn_mask_type == AttnMaskType.causal
        mask = self.build_flash_mask(sq, sk, device=query.device, causal=causal)
        # broadcastable to [b, np, sq, sk]
        attn_mask = mask.unsqueeze(0).unsqueeze(0)

        dropout_p = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # mask already encodes causality
            scale=self.softmax_scale,
        )
        # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, h]
        out = out.permute(2, 0, 1, 3).contiguous().view(sq, b, self.hidden_size_per_partition)
        return out


class MemorySparseAttention(FlashMaskAttention):
    """Sparse attention with interleaved memory tokens (keeps rule ②).

    Regular tokens attend within their sliding window *and* to memory tokens
    from earlier segments.
    """
    def __init__(self,
        config: TransformerConfig,
        attn_mask_type: AttnMaskType,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        pg_collection=None,
        group_size: Optional[int] = None,
        segment_size: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            attn_mask_type=attn_mask_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            pg_collection=pg_collection
        )
        self.segment_size = segment_size
        self.group_size = group_size

    def build_flash_mask(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> Tensor:
        assert self.group_size is not None and self.segment_size is not None, \
            "group_size and segment_size must be set for MemorySparseAttention"
        if seq_len_q != seq_len_k:
            raise ValueError(
                "Sparse memory mask requires seq_len_q == seq_len_k, "
                f"got {seq_len_q} vs {seq_len_k}"
            )
        # seq_len_q here is the *expanded* sequence length (including memory tokens).
        # _build_sparse_prepend_memory_pattern expects the *original* token count.
        original_seq_len = seq_len_q * self.group_size // (self.group_size + 1)
        mask = _build_sparse_prepend_memory_pattern(original_seq_len, self.group_size, self.segment_size, device)
        return mask
    


class SlidingWindowAttention(FlashMaskAttention):
    """Sliding-window-only sparse attention (skips rule ②).

    Regular tokens attend only within their sliding window and never to memory
    tokens from previous segments.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attn_mask_type: AttnMaskType,
        window_size: int,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        pg_collection=None,
    ):
        super().__init__(
            config=config,
            attn_mask_type=attn_mask_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            pg_collection=pg_collection,
        )
        self.window_size = window_size

    def build_flash_mask(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> Tensor:
        w = self.window_size
        qi = torch.arange(seq_len_q, device=device).unsqueeze(1)
        ki = torch.arange(seq_len_k, device=device).unsqueeze(0)
        # Sliding window: each query attends to keys within [qi - w + 1, qi]
        mask = (qi - w + 1 <= ki) & (ki <= qi)
        return mask
        
