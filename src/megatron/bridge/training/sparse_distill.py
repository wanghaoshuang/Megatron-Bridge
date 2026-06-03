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

    total = torch.zeros((), device=loss_mask.device, dtype=student_outs[0].dtype)
    # loss_mask: [b, s] -> [s, b, 1]
    m = loss_mask.transpose(0, 1).unsqueeze(-1).to(student_outs[0].dtype)
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
):
    """Combined LM CE + KL distillation loss.

    Returns ``(loss, num_tokens, report_dict)``, matching Megatron-Bridge's
    loss-function contract used by ``forward_step``.
    """
    lm_loss, num_tokens, report = masked_next_token_loss(
        loss_mask,
        output_tensor,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )
    kl = kl_per_layer(
        student_collector.get(),
        teacher_collector.get(),
        loss_mask=loss_mask,
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
    """

    def __init__(
        self,
        teacher_models: List[torch.nn.Module],
        student_collector: AttnOutputCollector,
        teacher_collector: AttnOutputCollector,
        alpha: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        self.teacher_models = teacher_models
        self.student_collector = student_collector
        self.teacher_collector = teacher_collector
        self.alpha = alpha
        self.temperature = temperature

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

        # 1) Run teacher under no_grad. We replay the same data via a tee'd iterator
        #    is non-trivial; instead, we wrap data_iterator so that the next() call
        #    is consumed once but the batch is captured and replayed for student.
        #    Simpler alternative: rely on identical token ids by running both
        #    forward passes inside _forward_step_common-style code. Here we run the
        #    student via _forward_step_common (which consumes the iterator), and
        #    BEFORE that, peek the batch via the teacher path using the same
        #    iterator. Easiest: wrap iterator so the same batch is yielded twice.
        wrapped_iter = _ReplayIterator(data_iterator)

        with torch.no_grad():
            for tm in self.teacher_models:
                tm.eval()
                _forward_step_common(state, wrapped_iter, tm, return_schedule_plan=False)
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
        )
        return output, loss_func


class _ReplayIterator:
    """Wraps a data iterator so the most recently produced batch can be replayed
    once (used to feed the same batch into teacher and student)."""

    def __init__(self, base):
        self._base = base
        self._cache = None
        self._replay = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._replay and self._cache is not None:
            self._replay = False
            return self._cache
        item = next(self._base)
        self._cache = item
        return item

    def replay(self) -> None:
        self._replay = True
