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


_SPARSE_MASK_CACHE: dict = {}


def _build_sparse_memory_pattern(
    seq_len: int,
    group_size: int,
    segment_size: int,
    device: torch.device,
) -> Tensor:
    """Build the interleaved (token + memory) sparse mask of shape ``[seq_len, seq_len]``.

    The expanded sequence has K = seq_len // (group_size + 1) memory tokens
    inserted at positions ``k*(g+1) + g``. ``segment_size`` is in *original*
    (un-expanded) token units and must satisfy ``segment_size % group_size == 0``.

    Mirrors ``experiments/sparse_attention/sparse_mask.py::_attend``.
    """
    g = group_size
    s = segment_size
    if g < 1:
        raise ValueError(f"group_size must be >= 1, got {g}")
    if s % g != 0:
        raise ValueError(f"segment_size ({s}) must be divisible by group_size ({g})")

    cache_key = (seq_len, g, s, device)
    cached = _SPARSE_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    K = seq_len // (g + 1)
    pos = torch.arange(seq_len, device=device)

    # is_mem[p] == True iff position p is a memory token slot.
    in_head = pos < K * (g + 1)
    is_mem = in_head & ((pos % (g + 1)) == g)

    # Original-token global index i for non-memory positions.
    # head: i = (p // (g+1)) * g + (p % (g+1))
    # tail: i = p - K
    i_idx = torch.where(
        in_head,
        (pos // (g + 1)) * g + (pos % (g + 1)),
        pos - K,
    )
    # Memory global index j for memory positions: j = p // (g+1).
    j_idx = pos // (g + 1)

    # Pairwise broadcast: rows = query, cols = key.
    qm = is_mem[:, None]
    km = is_mem[None, :]
    qi = i_idx[:, None]
    ki = i_idx[None, :]
    qj = j_idx[:, None]
    kj = j_idx[None, :]

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # (1) t -> t : sliding window.
    tt = (~qm) & (~km)
    # mask |= tt & (qi // s == ki // s) & (ki <= qi)
    # int(qi - (s - 1) <= ki <= qi) sliding window: attend to previous s-1 tokens
    mask |= tt & (qi - (s - 1) <= ki) & (ki <= qi)

    # (2) t -> m :  仅关注 sliding window 之前的 memory token
    # debuggggggg
    tm = (~qm) & km
    mask |= tm & ((ki + 1) * g <= qi - (s - 1))

    # (3) m -> t : same segment, within own group.
    mt = qm & (~km)
    mask |= mt & ((qj * g) // s == ki // s) & (ki < (qj + 1) * g)

    # (4) m -> m : causal.
    mm = qm & km
    mask |= mm & (kj < qj)

    _SPARSE_MASK_CACHE[cache_key] = mask
    return mask


def build_flash_mask(
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
    causal: bool = True,
    sparse_pattern: Optional[Tensor] = None,
    group_size: Optional[int] = None,
    segment_size: Optional[int] = None,
) -> Tensor:
    """Build a bool attention mask of shape ``[s_q, s_k]``.

    True positions are *kept* (consistent with ``F.scaled_dot_product_attention``'s
    ``attn_mask`` semantics when ``dtype=bool``: True allows attention).

    Args:
        seq_len_q: query sequence length.
        seq_len_k: key sequence length.
        device:    target device.
        causal:    enable causal (lower-triangular) mask.
        sparse_pattern: optional extra bool mask of shape ``[s_q, s_k]`` that
            will be AND-ed with the (causal/sparse) mask.
        group_size: if set together with ``segment_size``, use the interleaved
            (token + memory) sparse mask defined in
            ``experiments/sparse_attention/sparse_mask.py`` instead of a plain
            causal mask. Requires ``seq_len_q == seq_len_k``.
        segment_size: number of *original* tokens per segment (``s`` in the
            reference). Must be divisible by ``group_size``.

    Returns:
        Bool tensor of shape ``[s_q, s_k]``.
    """
    if group_size is not None and segment_size is not None:
        if seq_len_q != seq_len_k:
            raise ValueError(
                "Sparse memory mask requires seq_len_q == seq_len_k, "
                f"got {seq_len_q} vs {seq_len_k}"
            )
        mask = _build_sparse_memory_pattern(seq_len_q, group_size, segment_size, device)
    elif causal:
        mask = torch.ones(seq_len_q, seq_len_k, device=device, dtype=torch.bool).tril_()
    else:
        mask = torch.ones(seq_len_q, seq_len_k, device=device, dtype=torch.bool)
    if sparse_pattern is not None:
        mask = mask & sparse_pattern.to(device=device, dtype=torch.bool)
    return mask


class FlashMaskAttention(MegatronModule):
    """Drop-in replacement for ``DotProductAttention`` using SDPA + sparse mask.

    Interface (init signature, forward signature, return shape) is intentionally
    identical to ``megatron.core.transformer.dot_product_attention.DotProductAttention``
    so it can be assigned in place of ``self_attention.core_attention``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection=None,
        group_size: Optional[int] = None,
        segment_size: Optional[int] = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

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
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, tp_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, tp_size)

        self.attention_dropout = (
            attention_dropout if attention_dropout is not None else config.attention_dropout
        )
        self.softmax_scale = softmax_scale  # if None, SDPA uses 1/sqrt(d_k)
        self.group_size = group_size
        self.segment_size = segment_size

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
        mask = build_flash_mask(
            sq, sk,
            device=query.device,
            causal=causal,
            sparse_pattern=None,
            group_size=self.group_size,
            segment_size=self.segment_size,
        )
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
