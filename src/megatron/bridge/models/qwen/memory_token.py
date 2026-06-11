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


class MemoryTokenInjector(nn.Module):
    """Insert one trainable memory token at the end of every group of ``g`` tokens.

    For an embedding output of shape ``[s, b, h]`` and group size ``g``:

    - Process only the first ``K * g`` tokens, where ``K = s // g``.
    - Apply a single linear ``W: h -> h`` to those ``K*g`` tokens.
    - Mean-pool over each group of ``g`` to obtain ``m_k`` of shape ``[K, b, h]``.
    - Append ``m_k`` after each group, producing ``K * (g+1)`` tokens.
    - The trailing ``s - K*g`` tokens (if any) are passed through unchanged.

    The hook expects that :func:`expand_batch_for_memory_tokens` has already
    inserted placeholder ``pad`` tokens at the same positions, so that the
    incoming embedding tensor has length ``s' = K * (g+1) + (s - K*g)``.

    The hook *replaces* the activations at the placeholder positions with
    ``m_k``, leaving real-token activations untouched.
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
            embed_out: ``[s', b, h]`` embedding output for the *expanded* batch.

        Returns:
            Tensor of shape ``[s', b, h]`` with memory positions overwritten.
        """
        s_expanded, b, h = embed_out.shape
        g = self.group_size
        # Solve s' = K*(g+1) + r where 0 <= r < g; r is the trailing remainder
        # of the *original* (pre-expansion) sequence, which is < g by construction.
        K = s_expanded // (g + 1)
        r = s_expanded - K * (g + 1)
        if r >= g:
            # Should never happen if expand_batch_for_memory_tokens was used.
            raise RuntimeError(
                f"MemoryTokenInjector: unexpected expanded length {s_expanded} "
                f"for group_size {g}"
            )

        if K == 0:
            return embed_out

        head = embed_out[: K * (g + 1)]                  # [K*(g+1), b, h]
        tail = embed_out[K * (g + 1):]                   # [r, b, h], untouched

        # Reshape head into groups; positions 0..g-1 are real tokens, g is memory.
        head = head.view(K, g + 1, b, h)
        real = head[:, :g, :, :]                         # [K, g, b, h]

        # m_k = mean over group of FFN(real_tokens) using SwiGLU
        flat = real.reshape(K * g, b, h)
        ffn_out = self.down_proj(nn.functional.silu(self.gate_proj(flat)) * self.up_proj(flat))
        ffn_out = ffn_out.view(K, g, b, h)
        memory = ffn_out.mean(dim=1, keepdim=True)       # [K, 1, b, h]

        # Replace memory slot (index g) with the computed memory vectors.
        # `head` is a view; clone before in-place assignment to keep autograd
        # graph clean and avoid clobbering an upstream buffer.
        head = head.clone()
        # Cast to match the residual stream dtype (handles bf16/fp16 + Float16Module).
        head[:, g, :, :] = memory.squeeze(1).to(dtype=head.dtype)
        head = head.view(K * (g + 1), b, h)

        if tail.numel() == 0:
            return head
        return torch.cat([head, tail], dim=0)


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


def prepend_memory_position_mask(s_expanded: int, group_size: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[s']`` mask: ``True`` at memory-token positions for the prepend layout.

    Memory tokens occupy the first ``K = (s_expanded * g) // (g + 1)`` positions
    (equivalently ``K = s_expanded // (g + 1)``).
    """
    g = group_size
    K = s_expanded // (g + 1)
    mask = torch.zeros(s_expanded, dtype=torch.bool, device=device)
    if K > 0:
        mask[:K] = True
    return mask


def memory_position_mask(s_expanded: int, group_size: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[s']`` mask: ``True`` at memory-token positions."""
    g = group_size
    K = s_expanded // (g + 1)
    mask = torch.zeros(s_expanded, dtype=torch.bool, device=device)
    if K > 0:
        idx = torch.arange(K, device=device) * (g + 1) + g
        mask[idx] = True
    return mask


def expand_batch_for_memory_tokens(
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
    """Expand a per-rank batch by inserting one placeholder per ``g``-token group.

    Inserted at the end of every complete group of ``g`` tokens. Tail tokens
    (length ``s - K*g``) are kept as-is.

    - tokens: insert ``pad_token_id`` (only used by the embedding lookup; the
      activation is overwritten by :class:`MemoryTokenInjector`).
    - position_ids: insert a copy of the preceding token's position id
      (``k*g + g - 1``), so RoPE rotates the memory slot to that angle.
    - labels: insert ``-100`` (ignored by CE; ``loss_mask=0`` also makes them
      irrelevant).
    - loss_mask: insert ``0.0``, leaving real-token mask values untouched.
    - attention_mask: ``None`` is passed through (causal mask is regenerated by
      attention itself for the new length); if a mask was supplied, raise — the
      memory-token recipe assumes ``skip_getting_attention_mask_from_dataset=True``.
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

    real_len = K * g
    tail_len = s - real_len

    def _expand(x: torch.Tensor, fill_value, dtype=None) -> torch.Tensor:
        # x: [b, s]
        head = x[:, :real_len].view(b, K, g)
        pad_col = torch.full(
            (b, K, 1),
            fill_value,
            dtype=x.dtype if dtype is None else dtype,
            device=x.device,
        )
        head = torch.cat([head, pad_col], dim=2).view(b, K * (g + 1))
        if tail_len:
            head = torch.cat([head, x[:, real_len:]], dim=1)
        return head

    new_tokens = _expand(tokens, pad_token_id)

    # position_ids: insert k*g + g - 1 at each memory slot.
    head_pos = position_ids[:, :real_len].view(b, K, g)
    last_pos = head_pos[:, :, -1:]                       # [b, K, 1]
    new_pos = torch.cat([head_pos, last_pos], dim=2).view(b, K * (g + 1))
    if tail_len:
        new_pos = torch.cat([new_pos, position_ids[:, real_len:]], dim=1)

    new_labels = _expand(labels, -100) if labels is not None else None
    new_loss_mask = (
        _expand(loss_mask, 0.0, dtype=loss_mask.dtype) if loss_mask is not None else None
    )

    if attention_mask is not None:
        raise NotImplementedError(
            "MemoryToken expansion expects attention_mask=None "
            "(skip_getting_attention_mask_from_dataset=True)."
        )

    return new_tokens, new_pos, new_labels, new_loss_mask, None


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


def register_memory_token_injector(
    model: "nn.Module",
    group_size: int,
    name: str = "memory_token_injector",
) -> "MemoryTokenInjector":
    """Attach a :class:`MemoryTokenInjector` to ``model`` and wire it into the
    student's embedding forward path.

    Behaviour:
      - The injector is added as a *real* submodule of ``model`` (so DDP picks
        up its parameters when called *before* DDP wrapping).
      - ``model.embedding.forward`` is wrapped so that, after the original
        embedding produces its (sequence-parallel-scattered) output, the
        injector is applied on the un-scattered ``[s', b, h]`` tensor and the
        result is re-scattered to match the rest of the pipeline.

    Returns the injector module.
    """
    if not hasattr(model, "embedding") or model.embedding is None:
        raise RuntimeError("register_memory_token_injector: model has no .embedding")

    embedding = model.embedding
    config = embedding.config
    hidden_size = config.hidden_size

    injector = MemoryTokenInjector(hidden_size=hidden_size, group_size=group_size)
    # Place on the same device/dtype as the embedding's weight.
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

    # Bind in place; the hook is invoked by GPTModel._preprocess.
    embedding.forward = patched_forward
    return injector


def collapse_to_original_positions(
    attn_out: torch.Tensor,
    group_size: int,
    original_seq_len: int,
) -> torch.Tensor:
    """Drop memory-token rows from ``[s', b, h]`` to recover ``[s, b, h]``.

    ``s' - s == s // g``. The memory positions are at indices ``k*(g+1) + g``
    for ``k in [0, K)``; everything else is preserved.
    """
    s_expanded = attn_out.size(0)
    g = group_size
    K = s_expanded // (g + 1)
    expected = original_seq_len + K
    if s_expanded != expected:
        raise RuntimeError(
            f"collapse_to_original_positions: expected expanded length {expected} "
            f"(orig={original_seq_len}, K={K}, g={g}), got {s_expanded}"
        )
    if K == 0:
        return attn_out

    keep = torch.ones(s_expanded, dtype=torch.bool, device=attn_out.device)
    idx = torch.arange(K, device=attn_out.device) * (g + 1) + g
    keep[idx] = False
    return attn_out.index_select(0, torch.nonzero(keep, as_tuple=False).squeeze(1))


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
