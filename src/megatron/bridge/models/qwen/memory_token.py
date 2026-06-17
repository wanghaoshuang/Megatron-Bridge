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

"""Memory-token augmentation for the SparseAttention-distillation student.

A ``MemoryTokenInjector`` is registered on the student model only. It runs as a
forward hook on the embedding layer; after the original embedding produces a
``[s, b, h]`` activation, the hook replaces every ``g``-th slot (a placeholder
inserted by :func:`expand_batch_for_memory_tokens`) with a learnable, mean-MLP
summary of its preceding group, yielding ``[s', b, h]`` with ``s' = s + s // g``.

Teacher inputs / outputs are unchanged (length ``s``); the per-layer
core-attention outputs of the student are collapsed back to ``s`` via
:func:`collapse_to_original_positions` before being aligned with the teacher
for the KL term.

Layout convention follows Megatron: tensors are ``[s, b, h]`` (sbh), which is
the embedding output layout used by ``GPTModel``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MemoryQkvProjection(nn.Module):
    """Wraps an existing ``linear_qkv`` and appends a per-memory-token projection.

    After the base ``linear_qkv`` produces ``mixed_qkv, bias``, this module
    applies a separate linear projection (with kaiming-initialized weights)
    to the QKV values at memory-token positions only.  Real-token QKV values
    are left untouched.

    This follows the prepend layout: memory tokens sit at the first ``K``
    positions of the hidden-states sequence, where ``K = s_expanded // (g + 1)``.

    The module satisfies :class:`~megatron.core.transformer.attention.LinearQkvInterface`
    so it can replace ``self.linear_qkv`` on a ``SelfAttention`` instance.
    """

    def __init__(
        self,
        linear_qkv: nn.Module,
        qkv_out_dim: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.linear_qkv = linear_qkv
        self.group_size = group_size
        # Independent projection for memory-token QKV, same in/out dimensions.
        self.memory_proj = nn.Linear(qkv_out_dim, qkv_out_dim, bias=False)
        nn.init.zeros_(self.memory_proj.weight)

    @property
    def config(self):
        """Delegate config to the wrapped linear_qkv so LoRA utilities can access it."""
        return self.linear_qkv.config

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, object]:
        mixed_qkv, bias = self.linear_qkv(hidden_states)

        g = self.group_size
        s_expanded = mixed_qkv.size(0)
        K = s_expanded // (g + 1)
        if K == 0:
            return mixed_qkv, bias

        # Project only the memory-token positions (first K rows).
        mem_qkv = mixed_qkv[:K]                        # [K, b, qkv_dim]
        projected = self.memory_proj(mem_qkv)           # [K, b, qkv_dim]

        # Replace in-place on a clone to keep autograd clean.
        result = mixed_qkv.clone()
        result[:K] = projected.to(dtype=result.dtype)
        return result, bias

    def backward_dw(self) -> None:
        self.linear_qkv.backward_dw()



class PrependMemoryTokenInjector(nn.Module):
    """Insert trainable memory tokens prepended to the sequence.

    Unlike :class:`MemoryTokenInjector` where memory slots are interleaved
    inside the sequence (one after every ``g`` real tokens), this variant
    expects all ``K`` memory placeholders to sit at the *head* of the
    expanded embedding output::

        [M_0, M_1, …, M_{K-1}, t_0, t_1, …, t_{s-1}]

    where ``K = s // g`` and the expanded length is ``s' = K + s``.

    For each memory slot ``k``, the injector:
      - Collects the corresponding group of ``g`` real tokens (indices
        ``k*g … k*g + g - 1`` in the *original* sequence, i.e.
        ``K + k*g … K + k*g + g - 1`` in the expanded tensor).
      - Applies a SwiGLU FFN and mean-pools over the group to produce
        ``m_k``.
      - Overwrites the placeholder at position ``k`` with ``m_k``.
    """

    def __init__(self, hidden_size: int, group_size: int) -> None:
        super().__init__()
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")
        self.hidden_size = hidden_size
        self.group_size = group_size
        # SwiGLU FFN matching Qwen3-30B-A3B expert FFN shape (moe_intermediate_size=768)
        moe_intermediate_size = 768
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        nn.init.kaiming_normal_(self.gate_proj.weight)
        nn.init.kaiming_normal_(self.up_proj.weight)
        nn.init.kaiming_normal_(self.down_proj.weight)

    def forward(self, embed_out: torch.Tensor) -> torch.Tensor:
        """Args:
            embed_out: ``[s', b, h]`` embedding output for the *expanded* batch,
                where the first ``K`` positions are memory placeholders and the
                remaining ``s = s' - K`` positions are real tokens.

        Returns:
            Tensor of shape ``[s', b, h]`` with memory positions overwritten.
        """
        s_expanded, b, h = embed_out.shape
        g = self.group_size
        # K memory slots at the head, followed by s real tokens.
        # s_expanded = K + s, where s >= K*g (at least K full groups).
        # Solve: K = (s_expanded - s_expanded % g) // g  ... but more directly:
        # s = s_expanded - K, and K*g <= s < (K+1)*g.
        # => K*g <= s_expanded - K < (K+1)*g
        # => K*(g+1) <= s_expanded < K*(g+1) + g
        K = s_expanded // (g + 1)
        # Validate: s = s_expanded - K must have at least K*g real tokens.
        s = s_expanded - K
        if s < K * g:
            raise RuntimeError(
                f"PrependMemoryTokenInjector: expanded length {s_expanded} "
                f"implies K={K} but only {s} real tokens (< {K}*{g}) — "
                f"was prepend_batch_for_memory_tokens used?"
            )

        if K == 0:
            return embed_out

        # Layout: [M_0..M_{K-1}, t_0..t_{s-1}]
        memory_slots = embed_out[:K]                      # [K, b, h]
        real_tokens = embed_out[K:]                        # [s, b, h]

        # Gather each group of g real tokens: group k is real_tokens[k*g : k*g+g]
        groups = real_tokens[: K * g].view(K, g, b, h)    # [K, g, b, h]

        # m_k = mean over group of FFN(real_tokens) using SwiGLU
        flat = groups.reshape(K * g, b, h)
        ffn_out = self.down_proj(nn.functional.silu(self.gate_proj(flat)) * self.up_proj(flat))
        ffn_out = ffn_out.view(K, g, b, h)
        memory = ffn_out.mean(dim=1)                       # [K, b, h]

        # Replace memory placeholders with computed memory vectors.
        result = embed_out.clone()
        result[:K] = memory.to(dtype=result.dtype)

        return result


def prepend_batch_for_memory_tokens(
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    group_size: int,
    pad_token_id: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Expand a per-rank batch by prepending one placeholder per ``g``-token group.

    Instead of interleaving memory slots inside the sequence (as
    :func:`expand_batch_for_memory_tokens` does), this variant places all ``K``
    memory placeholders at the *head* of the sequence, yielding layout::

        [M_0, M_1, …, M_{K-1}, t_0, t_1, …, t_{s-1}]

    where ``K = s // g``.  The total output length is ``K + s``.

    - tokens: prepend ``K`` copies of ``pad_token_id``.
    - position_ids: prepend position ids matching the interleaved layout
      (``k*g + g - 1`` for each memory slot ``k``), so RoPE angles are
      preserved even though the memory tokens are physically at the head.
    - labels: prepend ``K`` copies of ``-100``.
    - loss_mask: prepend ``K`` copies of ``0.0``.
    - attention_mask: ``None`` is passed through; if a mask was supplied,
      raise — the memory-token recipe assumes
      ``skip_getting_attention_mask_from_dataset=True``.
    """
    if tokens is None:
        return tokens, position_ids, labels, loss_mask, attention_mask

    g = group_size
    if g < 2:
        raise ValueError(f"group_size must be >= 2, got {g}")
    b, s = tokens.shape
    K = s // g
    if K == 0:
        return tokens, position_ids, labels, loss_mask, attention_mask

    # tokens: prepend K pad tokens
    pad_tok = torch.full((b, K), pad_token_id, dtype=tokens.dtype, device=tokens.device)
    new_tokens = torch.cat([pad_tok, tokens], dim=1)

    # position_ids: memory slot k gets the same position as in the interleaved
    # layout, i.e. the last position of its group: k*g + g - 1.
    mem_pos = position_ids[:, :K * g].view(b, K, g)[:, :, -1]  # [b, K]
    new_pos = torch.cat([mem_pos, position_ids], dim=1)

    # labels: prepend K copies of -100
    if labels is not None:
        ign_lab = torch.full((b, K), -100, dtype=labels.dtype, device=labels.device)
        new_labels = torch.cat([ign_lab, labels], dim=1)
    else:
        new_labels = None

    # loss_mask: prepend K zeros
    if loss_mask is not None:
        zero_lm = torch.zeros(b, K, dtype=loss_mask.dtype, device=loss_mask.device)
        new_loss_mask = torch.cat([zero_lm, loss_mask], dim=1)
    else:
        new_loss_mask = None

    if attention_mask is not None:
        raise NotImplementedError(
            "MemoryToken expansion expects attention_mask=None "
            "(skip_getting_attention_mask_from_dataset=True)."
        )

    return new_tokens, new_pos, new_labels, new_loss_mask, None


def collapse_prepend_to_original_positions(
    attn_out: torch.Tensor,
    group_size: int,
    original_seq_len: int,
) -> torch.Tensor:
    """Drop memory-token rows from a prepend-layout ``[s', b, h]`` to recover ``[s, b, h]``.

    In the prepend layout the first ``K = s // g`` positions are memory tokens;
    dropping them recovers the original sequence.
    """
    s_expanded = attn_out.size(0)
    g = group_size
    K = s_expanded - original_seq_len
    if K < 0:
        raise RuntimeError(
            f"collapse_prepend_to_original_positions: expanded length {s_expanded} "
            f"is shorter than original {original_seq_len}"
        )
    expected_K = original_seq_len // g
    if K != expected_K:
        raise RuntimeError(
            f"collapse_prepend_to_original_positions: expected K={expected_K} "
            f"memory slots (orig={original_seq_len}, g={g}), got K={K}"
        )
    if K == 0:
        return attn_out
    return attn_out[K:]


def register_prepend_memory_token_injector(
    model: "nn.Module",
    group_size: int,
    name: str = "prepend_memory_token_injector",
) -> "PrependMemoryTokenInjector":
    """Attach a :class:`PrependMemoryTokenInjector` to ``model`` and wire it
    into the student's embedding forward path.

    Behaviour mirrors :func:`register_memory_token_injector` but uses the
    prepend-layout injector.
    """
    if not hasattr(model, "embedding") or model.embedding is None:
        raise RuntimeError("register_prepend_memory_token_injector: model has no .embedding")

    embedding = model.embedding
    config = embedding.config
    hidden_size = config.hidden_size

    injector = PrependMemoryTokenInjector(hidden_size=hidden_size, group_size=group_size)
    weight = embedding.word_embeddings.weight
    injector.to(device=weight.device, dtype=weight.dtype)
    setattr(model, name, injector)

    sequence_parallel = bool(getattr(config, "sequence_parallel", False))
    tp_group = getattr(embedding, "tp_group", None)
    tp_world_size = tp_group.size() if tp_group is not None else 1

    base_forward = embedding.forward

    def patched_forward(input_ids, position_ids, tokentype_ids=None):
        emb = base_forward(input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids)
        if sequence_parallel and tp_world_size > 1:
            from megatron.core import tensor_parallel as mpu_tp

            emb_full = mpu_tp.gather_from_sequence_parallel_region(emb, group=tp_group)
            emb_full = injector(emb_full)
            emb = mpu_tp.scatter_to_sequence_parallel_region(emb_full, group=tp_group)
        else:
            emb = injector(emb)
        return emb

    embedding.forward = patched_forward
    return injector
