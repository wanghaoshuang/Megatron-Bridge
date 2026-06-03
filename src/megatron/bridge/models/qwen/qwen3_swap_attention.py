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

"""Utilities to convert a standard Qwen3 GPTModel's attention into
FlashMaskAttention, and to register hooks that capture per-layer attention
outputs (the tensor right before ``linear_proj``).

Used by the SparseAttention distillation training pipeline to:
1. Build the *student* model B = swap_to_flashmask(model_A_copy)
2. Capture core_attention outputs from every decoder layer of both teacher
   and student for KL-loss alignment.
"""

from typing import Iterable, List

import torch
import torch.nn as nn

from megatron.bridge.models.qwen.sparse_attention import FlashMaskAttention


def _iter_decoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    """Yield the transformer decoder layers of a (possibly DDP-wrapped) GPTModel."""
    # Unwrap common wrappers (DDP / Float16Module / ModelOpt wrappers).
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    decoder = getattr(inner, "decoder", None)
    if decoder is None:
        return
    for layer in decoder.layers:
        yield layer


def swap_to_flashmask(model: nn.Module) -> nn.Module:
    """Replace ``self_attention.core_attention`` of every decoder layer with
    :class:`FlashMaskAttention`.

    The replacement preserves ``linear_qkv``, ``linear_proj``, ``q_layernorm``,
    ``k_layernorm`` and the rest of the SelfAttention module, so the existing
    pretrained weights remain valid.
    """
    for layer in _iter_decoder_layers(model):
        self_attn = getattr(layer, "self_attention", None)
        if self_attn is None or not hasattr(self_attn, "core_attention"):
            continue
        old = self_attn.core_attention
        new = FlashMaskAttention(
            config=self_attn.config,
            layer_number=getattr(self_attn, "layer_number", 1),
            attn_mask_type=getattr(old, "attn_mask_type", None),
            attention_type=getattr(old, "attention_type", "self"),
        ).to(device=next(self_attn.parameters()).device, dtype=next(self_attn.parameters()).dtype)
        self_attn.core_attention = new
    return model


class AttnOutputCollector:
    """Captures per-layer ``core_attention`` forward outputs via hooks.

    Hooks are attached to each decoder layer's ``self_attention.core_attention``.
    The captured tensor has shape ``[s, b, h]`` (output of the core attention,
    i.e., the *input* of ``linear_proj``).
    """

    def __init__(self) -> None:
        self.outputs: List[torch.Tensor] = []
        self._handles = []

    def clear(self) -> None:
        self.outputs.clear()

    def get(self) -> List[torch.Tensor]:
        return list(self.outputs)

    def attach(self, model: nn.Module) -> "AttnOutputCollector":
        for layer in _iter_decoder_layers(model):
            self_attn = getattr(layer, "self_attention", None)
            if self_attn is None or not hasattr(self_attn, "core_attention"):
                continue
            h = self_attn.core_attention.register_forward_hook(self._hook)
            self._handles.append(h)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, module, inputs, output):
        # output: Tensor [s, b, h]   (or tuple in some impls — take first)
        if isinstance(output, tuple):
            output = output[0]
        self.outputs.append(output)
