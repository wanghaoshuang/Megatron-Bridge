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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Load the module under test directly by file path, bypassing the
# megatron.bridge package __init__.py chain which pulls in transformers,
# megatron-core, etc.
_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "megatron"
    / "bridge"
    / "models"
    / "qwen"
    / "memory_token.py"
)
_spec = importlib.util.spec_from_file_location("memory_token", _MODULE_PATH)
_memory_token = importlib.util.module_from_spec(_spec)
# The module only imports torch and torch.nn, so this is safe.
_spec.loader.exec_module(_memory_token)
expand_batch_for_memory_tokens = _memory_token.expand_batch_for_memory_tokens
prepend_batch_for_memory_tokens = _memory_token.prepend_batch_for_memory_tokens
MemoryTokenInjector = _memory_token.MemoryTokenInjector


class TestExpandBatchForMemoryTokens:

    """Tests for expand_batch_for_memory_tokens."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_inputs(b, s, has_labels=True, has_loss_mask=True):
        tokens = torch.arange(s).unsqueeze(0).expand(b, -1).clone()
        position_ids = torch.arange(s).unsqueeze(0).expand(b, -1).clone()
        labels = torch.arange(100, 100 + s).unsqueeze(0).expand(b, -1).clone() if has_labels else None
        loss_mask = torch.ones(b, s) if has_loss_mask else None
        return tokens, position_ids, labels, loss_mask

    # ------------------------------------------------------------------
    # Basic shape & values: s divisible by g, no tail
    # ------------------------------------------------------------------

    def test_shape_no_tail(self):
        """s=6, g=3 → each of 2 groups gets 1 memory slot → new s' = 8."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=2, s=6)
        out = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 3, 999)
        new_tok, new_pos, new_lab, new_lm, new_am = out
        assert new_tok.shape == (2, 8)
        assert new_pos.shape == (2, 8)
        assert new_lab.shape == (2, 8)
        assert new_lm.shape == (2, 8)
        assert new_am is None

    def test_tokens_pad_value(self):
        """Memory slots in tokens must be pad_token_id."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        PAD = 42
        new_tok, *_ = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 3, PAD)
        # groups: [0,1,2, MEM, 3,4,5, MEM]  → indices 3 and 7
        assert new_tok[0, 3].item() == PAD
        assert new_tok[0, 7].item() == PAD
        # real tokens preserved
        assert new_tok[0, 0].item() == 0
        assert new_tok[0, 2].item() == 2
        assert new_tok[0, 4].item() == 3
        assert new_tok[0, 6].item() == 5

    def test_position_ids_replicate_last(self):
        """Memory-slot position_id == preceding token's position_id."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        new_tok, new_pos, *_ = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 3, 0)
        # group 0: pos [0,1,2, 2]; group 1: pos [3,4,5, 5]
        assert new_pos[0, 3].item() == 2
        assert new_pos[0, 7].item() == 5
        # real positions preserved
        assert new_pos[0, 0].item() == 0
        assert new_pos[0, 1].item() == 1
        assert new_pos[0, 2].item() == 2

    def test_labels_minus100(self):
        """Memory slots in labels must be -100."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        new_tok, new_pos, new_lab, *_ = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 3, 0)
        assert new_lab[0, 3].item() == -100
        assert new_lab[0, 7].item() == -100
        # real labels preserved
        assert new_lab[0, 0].item() == 100
        assert new_lab[0, 4].item() == 103

    def test_loss_mask_zero_at_memory(self):
        """Memory slots in loss_mask must be 0.0."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        _, _, _, new_lm, _ = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 3, 0)
        assert new_lm[0, 3].item() == 0.0
        assert new_lm[0, 7].item() == 0.0
        # real positions still 1.0
        assert new_lm[0, 0].item() == 1.0
        assert new_lm[0, 4].item() == 1.0

    # ------------------------------------------------------------------
    # With tail (s not divisible by g)
    # ------------------------------------------------------------------

    def test_shape_with_tail(self):
        """s=7, g=3 → K=2, tail_len=1 → new s' = 2*(3+1) + 1 = 9."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=7)
        new_tok, new_pos, new_lab, new_lm, _ = expand_batch_for_memory_tokens(
            tokens, pos, labels, loss_mask, None, 3, 999
        )
        assert new_tok.shape == (1, 9)

    def test_tail_preserved(self):
        """Trailing tokens (s - K*g) are kept unchanged at the end."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=7)
        PAD = 777
        new_tok, new_pos, new_lab, new_lm, _ = expand_batch_for_memory_tokens(
            tokens, pos, labels, loss_mask, None, 3, PAD
        )
        # tail is token 6 at index 8
        assert new_tok[0, 8].item() == 6
        assert new_pos[0, 8].item() == 6
        assert new_lab[0, 8].item() == 106
        assert new_lm[0, 8].item() == 1.0

    # ------------------------------------------------------------------
    # Optional fields: labels / loss_mask = None
    # ------------------------------------------------------------------

    def test_labels_none(self):
        tokens, pos, _, loss_mask = self._make_inputs(b=1, s=6, has_labels=False)
        _, _, new_lab, _, _ = expand_batch_for_memory_tokens(tokens, pos, None, loss_mask, None, 3, 0)
        assert new_lab is None

    def test_loss_mask_none(self):
        tokens, pos, labels, _ = self._make_inputs(b=1, s=6, has_loss_mask=False)
        _, _, _, new_lm, _ = expand_batch_for_memory_tokens(tokens, pos, labels, None, None, 3, 0)
        assert new_lm is None

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_tokens_none_returns_all_none_unchanged(self):
        """If tokens is None, return everything unchanged."""
        result = expand_batch_for_memory_tokens(None, None, None, None, None, 4, 0)
        assert result == (None, None, None, None, None)

    def test_group_size_too_small(self):
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 1, 0)

    def test_sequence_shorter_than_group(self):
        """s < g → K=0, return inputs unchanged."""
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=3)
        out = expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, None, 5, 999)
        new_tok, new_pos, new_lab, new_lm, new_am = out
        assert torch.equal(new_tok, tokens)
        assert torch.equal(new_pos, pos)
        assert torch.equal(new_lab, labels)
        assert torch.equal(new_lm, loss_mask)
        assert new_am is None

    def test_attention_mask_not_none_raises(self):
        tokens, pos, labels, loss_mask = self._make_inputs(b=1, s=6)
        attn = torch.ones(1, 6)
        with pytest.raises(NotImplementedError, match="attention_mask=None"):
            expand_batch_for_memory_tokens(tokens, pos, labels, loss_mask, attn, 3, 0)

    # ------------------------------------------------------------------
    # Multi-batch correctness
    # ------------------------------------------------------------------

    def test_multi_batch(self):
        """Different batch rows get independent but correct expansion."""
        b, s, g = 3, 6, 3
        tokens = torch.randint(0, 100, (b, s))
        pos = torch.arange(s).unsqueeze(0).expand(b, -1).clone()
        labels = torch.randint(0, 100, (b, s))
        loss_mask = torch.ones(b, s)

        new_tok, new_pos, new_lab, new_lm, _ = expand_batch_for_memory_tokens(
            tokens, pos, labels, loss_mask, None, g, 999
        )
        # shape
        assert new_tok.shape == (b, 8)
        # per-batch: real tokens match
        for bi in range(b):
            # group 0 real tokens
            assert torch.equal(new_tok[bi, :3], tokens[bi, :3])
            # memory slot
            assert new_tok[bi, 3].item() == 999
            # group 1 real tokens
            assert torch.equal(new_tok[bi, 4:7], tokens[bi, 3:6])
            # memory slot
            assert new_tok[bi, 7].item() == 999

            # position ids
            assert new_pos[bi, 3].item() == pos[bi, 2].item()
            assert new_pos[bi, 7].item() == pos[bi, 5].item()

    # ------------------------------------------------------------------
    # Exact full-sequence snapshot: g=2, s=5 (K=2, tail_len=1)
    # ------------------------------------------------------------------

    def test_full_snapshot_g2_s5(self):
        """Hand-computed reference for g=2, s=5."""
        # tokens:  [0, 1, 2, 3, 4]
        # groups:  [0,1, M, 2,3, M] + tail [4]
        # result:  [0, 1, M, 2, 3, M, 4]
        b = 1
        tokens = torch.tensor([[10, 20, 30, 40, 50]])
        pos = torch.tensor([[100, 101, 102, 103, 104]])
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        loss_mask = torch.ones(1, 5)
        PAD = 9999

        new_tok, new_pos, new_lab, new_lm, new_am = expand_batch_for_memory_tokens(
            tokens, pos, labels, loss_mask, None, 2, PAD
        )

        assert new_tok.tolist() == [[10, 20, PAD, 30, 40, PAD, 50]]
        assert new_pos.tolist() == [[100, 101, 101, 102, 103, 103, 104]]
        assert new_lab.tolist() == [[1, 2, -100, 3, 4, -100, 5]]
        assert new_lm.tolist() == [[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]
        assert new_am is None


class TestPrependBatchForMemoryTokens:

    """Tests for prepend_batch_for_memory_tokens."""

    # ------------------------------------------------------------------
    # Exact full-sequence snapshot: g=2, s=5 (K=2)
    # ------------------------------------------------------------------

    def test_full_snapshot_g2_s5(self):
        """Hand-computed reference for g=2, s=5 with prepend layout."""
        # tokens:  [10, 20, 30, 40, 50]
        # K = 5 // 2 = 2 → prepend 2 memory slots
        # result:  [M, M, 10, 20, 30, 40, 50]
        tokens = torch.tensor([[10, 20, 30, 40, 50]])
        pos = torch.tensor([[100, 101, 102, 103, 104]])
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        loss_mask = torch.ones(1, 5)
        PAD = 9999

        new_tok, new_pos, new_lab, new_lm, new_am = prepend_batch_for_memory_tokens(
            tokens, pos, labels, loss_mask, None, 2, PAD
        )

        assert new_tok.tolist() == [[PAD, PAD, 10, 20, 30, 40, 50]]
        # K=2, memory positions = last position in each group
        # group 0: pos 100,101 → mem pos 101; group 1: pos 102,103 → mem pos 103
        assert new_pos.tolist() == [[101, 103, 100, 101, 102, 103, 104]]
        assert new_lab.tolist() == [[-100, -100, 1, 2, 3, 4, 5]]
        assert new_lm.tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        assert new_am is None


class TestMemoryTokenInjector:
    """Tests for MemoryTokenInjector.forward."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_injector(hidden_size: int = 16, group_size: int = 3) -> MemoryTokenInjector:
        """Create a small injector for testing (hidden_size=16, not 768)."""
        inj = MemoryTokenInjector(hidden_size=hidden_size, group_size=group_size)
        # Override the 768-dim FFN with a smaller one for test speed.
        mid = 8
        inj.gate_proj = nn.Linear(hidden_size, mid, bias=False)
        inj.up_proj = nn.Linear(hidden_size, mid, bias=False)
        inj.down_proj = nn.Linear(mid, hidden_size, bias=False)
        nn.init.kaiming_normal_(inj.gate_proj.weight)
        nn.init.kaiming_normal_(inj.up_proj.weight)
        nn.init.kaiming_normal_(inj.down_proj.weight)
        return inj

    @staticmethod
    def _build_expanded_tensor(b: int, orig_s: int, g: int, h: int) -> torch.Tensor:
        """Build an [s', b, h] tensor mimicking the output of expand_batch_for_memory_tokens.

        Real-token positions get sequential values so they are distinguishable;
        memory-token positions (the placeholders) get zeros.
        """
        K = orig_s // g
        r = orig_s - K * g
        s_expanded = K * (g + 1) + r
        # Start with zeros (placeholder value)
        tensor = torch.zeros(s_expanded, b, h)
        # Fill real-token positions with unique values
        val = 0.0
        for k in range(K):
            for i in range(g):
                idx = k * (g + 1) + i
                val += 1.0
                tensor[idx, :, :] = val
        for i in range(r):
            idx = K * (g + 1) + i
            val += 1.0
            tensor[idx, :, :] = val
        return tensor

    # ------------------------------------------------------------------
    # Shape preservation
    # ------------------------------------------------------------------

    def test_shape_unchanged_no_tail(self):
        """s=6, g=3 → s'=8; output shape must equal input shape."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = torch.randn(8, 2, 16)  # K=2, g=3, s'=2*(3+1)=8
        out = inj(x)
        assert out.shape == x.shape

    def test_shape_unchanged_with_tail(self):
        """s=7, g=3 → K=2, r=1, s'=9; output shape must equal input shape."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = torch.randn(9, 2, 16)
        out = inj(x)
        assert out.shape == x.shape

    # ------------------------------------------------------------------
    # Real-token activations are not modified
    # ------------------------------------------------------------------

    def test_real_tokens_unchanged(self):
        """Only memory positions are overwritten; real-token activations stay the same."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = self._build_expanded_tensor(b=1, orig_s=7, g=3, h=16)
        out = inj(x)
        # Memory positions for g=3, K=2: indices 3 and 7
        # Real positions: everything else
        real_indices = [0, 1, 2, 4, 5, 6, 8]  # 0..2 (group0), 4..6 (group1), 8 (tail)
        for idx in real_indices:
            assert torch.allclose(out[idx], x[idx]), f"Real token at index {idx} was modified"

    # ------------------------------------------------------------------
    # Memory positions are overwritten (not zero)
    # ------------------------------------------------------------------

    def test_memory_positions_overwritten(self):
        """Memory-slot activations must be replaced (not the original zero placeholder)."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = self._build_expanded_tensor(b=1, orig_s=6, g=3, h=16)
        # Memory positions are 3 and 7 — they start as zeros
        assert torch.all(x[3] == 0)
        assert torch.all(x[7] == 0)
        out = inj(x)
        # After injection, they must be non-zero (FFN output of non-zero input)
        assert not torch.all(out[3] == 0), "Memory position 3 still zero after injection"
        assert not torch.all(out[7] == 0), "Memory position 7 still zero after injection"

    # ------------------------------------------------------------------
    # Memory positions match manual SwiGLU computation
    # ------------------------------------------------------------------

    def test_memory_values_match_manual_computation(self):
        """Memory values must equal the manual SwiGLU + mean computation."""
        h, g = 16, 3
        inj = self._make_injector(hidden_size=h, group_size=g)
        b = 1
        x = self._build_expanded_tensor(b=b, orig_s=6, g=g, h=h)
        out = inj(x)

        # Manually compute what the memory token should be for each group
        K = 6 // g  # = 2
        for k in range(K):
            # Real tokens for group k: positions k*(g+1) ... k*(g+1)+g-1
            real_indices = [k * (g + 1) + i for i in range(g)]
            real_tokens = torch.stack([x[idx, 0, :] for idx in real_indices])  # [g, h]

            # SwiGLU FFN
            flat = real_tokens  # [g, h]
            gate = inj.gate_proj(flat)
            up = inj.up_proj(flat)
            ffn_out = inj.down_proj(torch.nn.functional.silu(gate) * up)  # [g, h]
            expected = ffn_out.mean(dim=0)  # [h]

            mem_idx = k * (g + 1) + g
            actual = out[mem_idx, 0, :]
            assert torch.allclose(actual, expected, atol=1e-5), (
                f"Memory token at group {k} doesn't match manual computation"
            )

    # ------------------------------------------------------------------
    # Tail tokens are completely untouched
    # ------------------------------------------------------------------

    def test_tail_tokens_untouched(self):
        """Trailing remainder tokens must pass through exactly."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = self._build_expanded_tensor(b=1, orig_s=7, g=3, h=16)
        # K=2, r=1, tail is at index 8
        out = inj(x)
        # tail should be exactly the original
        assert torch.equal(out[8], x[8])

    # ------------------------------------------------------------------
    # K == 0 (sequence shorter than group): passthrough
    # ------------------------------------------------------------------

    def test_short_sequence_passthrough(self):
        """If s' < g+1 then K=0 and input is returned unchanged."""
        inj = self._make_injector(hidden_size=16, group_size=5)
        x = torch.randn(3, 1, 16)  # s'=3 < g+1=6 → K=0
        out = inj(x)
        assert torch.equal(out, x)

    # ------------------------------------------------------------------
    # group_size validation
    # ------------------------------------------------------------------

    def test_group_size_too_small_raises(self):
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            self._make_injector(hidden_size=16, group_size=1)

    # ------------------------------------------------------------------
    # Invalid expanded length raises RuntimeError
    # ------------------------------------------------------------------

    def test_invalid_expanded_length_raises(self):
        """If r >= g, the expanded length is inconsistent and should raise."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        # g=3 → valid s' must satisfy r = s' % (g+1) < g, i.e. r < 3
        # s'=7 → K=1, r=3 which is >= g → should raise
        x = torch.randn(7, 1, 16)
        with pytest.raises(RuntimeError, match="unexpected expanded length"):
            inj(x)

    # ------------------------------------------------------------------
    # Gradient flows through memory tokens
    # ------------------------------------------------------------------

    def test_gradient_flows(self):
        """Gradients must propagate from memory positions back to real-token inputs."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        x = self._build_expanded_tensor(b=1, orig_s=6, g=3, h=16)
        x.requires_grad_(True)
        out = inj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient on input"
        # Real token positions should have gradients (memory is computed from them)
        assert x.grad.abs().sum() > 0, "All gradients are zero"

    # ------------------------------------------------------------------
    # Different batch sizes
    # ------------------------------------------------------------------

    def test_multi_batch(self):
        """Works correctly with batch size > 1."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        b = 4
        x = self._build_expanded_tensor(b=b, orig_s=6, g=3, h=16)
        out = inj(x)
        assert out.shape == x.shape
        # Memory positions should differ across batches only if input differs
        # Here all batches have the same input, so memory should be the same
        mem_idx = 3  # first memory position
        for bi in range(1, b):
            assert torch.allclose(out[mem_idx, 0], out[mem_idx, bi], atol=1e-5)

    # ------------------------------------------------------------------
    # dtype handling (bf16)
    # ------------------------------------------------------------------

    def test_bf16_dtype(self):
        """Injector cast should work with bf16 input."""
        inj = self._make_injector(hidden_size=16, group_size=3)
        inj = inj.to(torch.bfloat16)
        x = self._build_expanded_tensor(b=1, orig_s=6, g=3, h=16).to(torch.bfloat16)
        out = inj(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape
