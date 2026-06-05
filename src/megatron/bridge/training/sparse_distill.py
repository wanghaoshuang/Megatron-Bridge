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

"""SparseAttention distillation forward step + KL loss.

Pipeline:
  - Student (B): Qwen3 with FlashMaskAttention.
  - Teacher (A): original Qwen3 with standard attention; eval-mode, no_grad.
  - Both attached with :class:`AttnOutputCollector` hooks. After a forward,
    we have ``[s, b, h]`` core-attention outputs from each decoder layer.
  - Total loss = LM CE + alpha * sum_l KL(softmax(s_l/T) || softmax(t_l/T)) * T^2

Usage::

    forward_step = SparseDistillForwardStep(
        teacher_models=teacher,                # list[GPTModel]
        student_collector=student_collector,
        teacher_collector=teacher_collector,
        alpha=1.0,
        temperature=1.0,
    )
    pretrain(cfg, forward_step_func=forward_step)
"""

from functools import partial
from typing import Iterable, List

import torch
import torch.nn.functional as F

from megatron.bridge.training.gpt_step import _forward_step_common
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState

from megatron.bridge.models.qwen.qwen3_swap_attention import AttnOutputCollector
from megatron.bridge.models.qwen.memory_token import (
    collapse_to_original_positions,
    expand_batch_for_memory_tokens,
)



def kl_per_layer(
    student_outs: List[torch.Tensor],
    teacher_outs: List[torch.Tensor],
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between per-layer core-attention outputs.

    Each tensor: ``[s, b, h]``. We treat the last dim as the distribution and
    apply ``softmax`` after temperature scaling. Per-token KL is averaged using
    ``loss_mask`` (``[b, s]``); contributions are summed across layers.

    Returns a scalar tensor (sum across layers).
    """
    assert len(student_outs) == len(teacher_outs), (
        f"layer count mismatch: student={len(student_outs)} teacher={len(teacher_outs)}"
    )
    if not student_outs:
        return torch.tensor(0.0, device=loss_mask.device, dtype=loss_mask.dtype)

    total = torch.zeros((), device=student_outs[0].device, dtype=student_outs[0].dtype)
    # loss_mask: [b, s] -> [s, b, 1]
    m = loss_mask.transpose(0, 1).unsqueeze(-1).to(student_outs[0])
    denom = m.sum().clamp_min(1.0)
    T = temperature

    for s_o, t_o in zip(student_outs, teacher_outs):
        # Match shapes / dtypes.
        if s_o.shape != t_o.shape:
            raise RuntimeError(f"attn output shape mismatch: {s_o.shape} vs {t_o.shape}")
        s_logp = F.log_softmax(s_o / T, dim=-1)
        t_p = F.softmax(t_o.detach() / T, dim=-1)
        # F.kl_div(reduction='none'): elementwise t_p * (log t_p - s_logp)
        kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1, keepdim=True)  # [s, b, 1]
        kl = (kl * m).sum() / denom
        total = total + kl * (T * T)

    return total


def _sparse_distill_loss(
    output_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    student_collector: AttnOutputCollector,
    teacher_collector: AttnOutputCollector,
    alpha: float,
    temperature: float,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
    teacher_loss_mask: torch.Tensor = None,
    group_size: int = 0,
    original_seq_len: int = 0,
):
    """Combined LM CE + KL distillation loss.

    Returns ``(loss, num_tokens, report_dict)``, matching Megatron-Bridge's
    loss-function contract used by ``forward_step``.

    When ``group_size > 0`` (memory-token mode), the student's per-layer
    attention outputs are length ``s'`` while the teacher's are length ``s``;
    the student tensors are collapsed back to ``s`` for the KL alignment, and
    ``teacher_loss_mask`` is used for the KL averaging. The LM CE term still
    uses the expanded ``loss_mask`` (memory positions have ``loss_mask=0``).
    """
    lm_loss, num_tokens, report = masked_next_token_loss(
        loss_mask,
        output_tensor,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )
    student_outs = student_collector.get()
    teacher_outs = teacher_collector.get()
    if group_size > 0 and original_seq_len > 0 and student_outs:
        # core_attention outputs are full-sequence tensors [s', b, h/tp] — they are
        # NOT sequence-parallel-scattered (the SP scatter happens at linear_proj's
        # reduce-scatter, which comes *after* core_attention). Gathering here would
        # double the sequence length and break collapse_to_original_positions.
        collapsed = [
            collapse_to_original_positions(s_o, group_size, original_seq_len)
            for s_o in student_outs
        ]
        student_outs = collapsed
    kl_mask = teacher_loss_mask if teacher_loss_mask is not None else loss_mask
    kl = kl_per_layer(
        student_outs,
        teacher_outs,
        loss_mask=kl_mask,
        temperature=temperature,
    )
    total = lm_loss + alpha * kl
    report["lm_loss"] = lm_loss.detach()
    report["kl_loss"] = kl.detach()
    report["total_loss"] = total.detach()

    # Free hook buffers for next iteration.
    student_collector.clear()
    teacher_collector.clear()
    return total, num_tokens, report


class SparseDistillForwardStep:
    """Forward step functor for SparseAttention distillation training.

    Compatible with ``pretrain(cfg, forward_step_func=...)``: it accepts
    ``(state, data_iterator, model)`` and returns ``(output_tensor, loss_func)``.

    If ``group_size > 0``, the student receives a *memory-token-expanded* batch
    (length ``s + s // g``), and the per-layer attention outputs are collapsed
    back to length ``s`` before the KL alignment with the teacher.
    """

    def __init__(
        self,
        teacher_models: List[torch.nn.Module],
        student_collector: AttnOutputCollector,
        teacher_collector: AttnOutputCollector,
        alpha: float = 1.0,
        temperature: float = 1.0,
        group_size: int = 0,
        pad_token_id: int = 0,
    ) -> None:
        self.teacher_models = teacher_models
        self.student_collector = student_collector
        self.teacher_collector = teacher_collector
        self.alpha = alpha
        self.temperature = temperature
        self.group_size = group_size
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model,
        return_schedule_plan: bool = False,
    ):
        # Reset collectors before each forward to avoid leftover from previous iter.
        self.student_collector.clear()
        self.teacher_collector.clear()

        wrapped_iter = _ReplayIterator(data_iterator)

        with torch.no_grad():
            for tm in self.teacher_models:
                tm.eval()
                _forward_step_common(state, wrapped_iter, tm, return_schedule_plan=False)

        # Capture teacher's (original) loss_mask + length before any student-side
        # batch expansion for use in KL averaging.
        teacher_loss_mask = None
        original_seq_len = 0
        if wrapped_iter._cache is not None:
            cached = wrapped_iter._cache
            if isinstance(cached, dict):
                if "loss_mask" in cached and cached["loss_mask"] is not None:
                    teacher_loss_mask = cached["loss_mask"]
                if "tokens" in cached and cached["tokens"] is not None:
                    original_seq_len = cached["tokens"].size(1)

        # When memory-token augmentation is enabled, transform the cached batch
        # for the student replay so it sees the expanded sequence.
        if self.group_size > 0:
            wrapped_iter.set_transform(self._student_transform)
        wrapped_iter.replay()

        output, loss_mask = _forward_step_common(
            state, wrapped_iter, model, return_schedule_plan=return_schedule_plan
        )

        loss_func = partial(
            _sparse_distill_loss,
            loss_mask=loss_mask,
            student_collector=self.student_collector,
            teacher_collector=self.teacher_collector,
            alpha=self.alpha,
            temperature=self.temperature,
            check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
            teacher_loss_mask=teacher_loss_mask,
            group_size=self.group_size,
            original_seq_len=original_seq_len,
        )
        return output, loss_func

    def _student_transform(self, batch):
        """Expand a batch dict in place for memory-token-augmented student input."""
        if not isinstance(batch, dict):
            return batch
        tokens = batch.get("tokens")
        if tokens is None:
            return batch
        new_tokens, new_pos, new_labels, new_loss_mask, new_attn = expand_batch_for_memory_tokens(
            tokens=tokens,
            position_ids=batch.get("position_ids"),
            labels=batch.get("labels"),
            loss_mask=batch.get("loss_mask"),
            attention_mask=batch.get("attention_mask"),
            group_size=self.group_size,
            pad_token_id=self.pad_token_id,
        )
        new_batch = dict(batch)
        new_batch["tokens"] = new_tokens
        new_batch["position_ids"] = new_pos
        if new_labels is not None:
            new_batch["labels"] = new_labels
        if new_loss_mask is not None:
            new_batch["loss_mask"] = new_loss_mask
        new_batch["attention_mask"] = new_attn
        return new_batch


class _ReplayIterator:
    """Wraps a data iterator so the most recently produced batch can be replayed
    once (used to feed the same batch into teacher and student).

    A ``transform`` callable can be set before :meth:`replay` so the next
    ``__next__`` returns ``transform(cached_batch)`` instead of the raw cache.
    """

    def __init__(self, base):
        self._base = base
        self._cache = None
        self._replay = False
        self._transform = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._replay and self._cache is not None:
            self._replay = False
            transform, self._transform = self._transform, None
            if transform is not None:
                return transform(self._cache)
            return self._cache
        item = next(self._base)
        self._cache = item
        return item

    def replay(self) -> None:
        self._replay = True

    def set_transform(self, fn) -> None:
        self._transform = fn
