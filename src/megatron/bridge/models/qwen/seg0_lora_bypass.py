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

"""Seg0 LoRA bypass utilities for MemorySparseAttention (MSA) distillation layers.

When MSA is active, seg0 tokens (the first ``segment_size`` positions in each
sequence) should behave identically to the base pretrained model — no LoRA
contribution.  This module provides utilities to build seg1 masks and install
hooks/masks on LoRA adapter modules (LoRALinear, LoRATopKRouter) in every
TransformerLayer that uses MSA.

Data format: BSHD only (``[sq, B, H]``), matching the distillation training setup.
"""

from typing import List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------


def build_seg1_mask_bshd(
    sq: int,
    B: int,
    seg0_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    cp_rank: int = 0,
) -> Tensor:
    """Build float mask ``[sq * B]``: 1.0 for seg1+ positions, 0.0 for seg0.

    In BSHD mode each batch element is a separate sequence of length ``sq``.
    The global position is ``cp_rank * sq + local_pos`` (CP not yet supported
    in MSA, but kept for forward-compat).

    Returns a flattened ``[sq * B]`` mask matching MoE's flattened token layout.
    """
    pos = torch.arange(sq, device=device, dtype=torch.long)
    if cp_rank > 0:
        pos = pos + cp_rank * sq
    mask_1d = (pos >= seg0_tokens).to(dtype)
    # [sq] -> [sq, B] -> [sq * B]
    mask = mask_1d.unsqueeze(1).expand(sq, B).contiguous().view(-1)
    return mask


# ---------------------------------------------------------------------------
# Adapter hooks
# ---------------------------------------------------------------------------


def _make_lazy_seg0_hook():
    """Create a forward hook for ``LoRALinear.adapter`` that zeros seg0 positions.

    The hook lazily builds the seg1 mask on the first call and caches it as
    ``adapter._seg1_mask``.  Subsequent calls reuse the cached mask as long
    as the tensor shape matches.

    The adapter output has shape ``[sq, B, H]`` (BSHD format).  The mask is
    ``[sq * B]`` reshaped to ``[sq, B, 1, ...]`` for broadcasting.
    """
    def hook(module, inp, output):
        seg0_tokens = getattr(module, '_seg0_tokens', 0)
        if seg0_tokens <= 0:
            return output

        sq = output.shape[0]
        B = output.shape[1]

        cached_mask = getattr(module, '_seg1_mask', None)
        if cached_mask is not None and cached_mask.shape[0] == sq * B:
            mask = cached_mask
        else:
            mask = build_seg1_mask_bshd(
                sq, B, seg0_tokens,
                device=output.device,
                dtype=output.dtype,
            )
            module._seg1_mask = mask

        # Reshape mask: [sq * B] -> [sq, B, 1, ...]
        view_shape = (sq, B) + (1,) * (output.dim() - 2)
        return output * mask.view(view_shape).to(output.dtype)
    return hook


def _make_lazy_seg0_router_hook():
    """Create a forward hook for ``LoRATopKRouter.adapter`` that zeros seg0 positions.

    The router adapter output has shape ``[S * B, num_experts]`` (flattened
    from the BSHD layout).  To build the mask we need both S and B.  The
    batch size B is stored on the adapter by :func:`install_seg0_bypass_hooks`
    during hook installation (derived from the parent
    ``LoRATopKRouter``'s config).
    """
    def hook(module, inp, output):
        seg0_tokens = getattr(module, '_seg0_tokens', 0)
        if seg0_tokens <= 0:
            return output

        # output shape: [S * B, num_experts]
        T = output.shape[0]

        cached_mask = getattr(module, '_seg1_mask', None)
        if cached_mask is not None and cached_mask.shape[0] == T:
            mask = cached_mask
        else:
            # Infer sq and B from the stored batch size.
            # _batch_size is set during hook installation by install_seg0_bypass_hooks.
            B = getattr(module, '_batch_size', None)
            if B is None or B <= 0:
                # Fallback: cannot determine batch size; skip masking.
                return output
            sq = T // B
            if sq * B != T:
                # Shape mismatch; skip masking to avoid incorrect zeros.
                return output
            mask = build_seg1_mask_bshd(
                sq, B, seg0_tokens,
                device=output.device,
                dtype=output.dtype,
            )
            module._seg1_mask = mask

        # mask: [S * B], output: [S * B, num_experts]
        return output * mask.unsqueeze(-1).to(output.dtype)
    return hook


# ---------------------------------------------------------------------------
# Layer-level hook installation / cleanup
# ---------------------------------------------------------------------------


def _infer_batch_size_from_layer(layer: torch.nn.Module) -> int:
    """Try to infer the batch size that will be used during training.

    This is used by the router hook to reconstruct the per-sequence position
    from the flattened ``[S * B, num_experts]`` adapter output.  Returns 1
    as a safe default (correct for micro_batch_size=1 with BSHD layout).
    """
    # micro_batch_size is typically 1 in the distillation setup.
    # We read it from the layer's config if available.
    config = getattr(layer, 'config', None)
    if config is not None:
        return getattr(config, 'micro_batch_size', 1)
    return 1


def install_seg0_bypass_hooks(layer: torch.nn.Module, seg0_tokens: int) -> List[torch.utils.hooks.RemovableHandle]:
    """Install seg0 LoRA bypass hooks on a single TransformerLayer.

    Finds all ``LoRALinear`` and ``LoRATopKRouter`` wrappers inside the layer
    and registers forward hooks on their ``.adapter`` sub-modules so that seg0
    positions receive zero LoRA contribution.

    Args:
        layer: A ``megatron.core.transformer.TransformerLayer`` instance.
        seg0_tokens: Number of leading tokens to bypass (== segment_size for MSA).

    Returns:
        List of ``RemovableHandle`` for all installed hooks.
    """
    from megatron.bridge.peft.lora_layers import LoRALinear, LoRATopKRouter

    B = _infer_batch_size_from_layer(layer)
    handles = []

    for name, module in layer.named_modules():
        if isinstance(module, LoRALinear) and module._adapter_enabled:
            adapter = module.adapter
            adapter._seg0_tokens = seg0_tokens
            h = adapter.register_forward_hook(_make_lazy_seg0_hook())
            handles.append(h)
        elif isinstance(module, LoRATopKRouter) and module._adapter_enabled:
            adapter = module.adapter
            adapter._seg0_tokens = seg0_tokens
            adapter._batch_size = B
            h = adapter.register_forward_hook(_make_lazy_seg0_router_hook())
            handles.append(h)

    return handles


def clear_seg0_bypass(layer: torch.nn.Module) -> None:
    """Remove seg0 bypass state from all adapters in a layer.

    Clears cached masks and ``_seg0_tokens`` attributes so they don't leak
    between forward passes or interfere with non-MSA inference.
    """
    from megatron.bridge.peft.lora_layers import LoRALinear, LoRATopKRouter

    for name, module in layer.named_modules():
        if isinstance(module, (LoRALinear, LoRATopKRouter)):
            adapter = module.adapter
            for attr in ('_seg0_tokens', '_seg1_mask', '_batch_size'):
                if hasattr(adapter, attr):
                    delattr(adapter, attr)
