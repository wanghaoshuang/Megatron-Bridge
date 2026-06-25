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

"""LoRA adapter for MoE Router's gating weights (nn.Parameter based)."""

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

from .utils_patch import get_init_fn
def is_router_module(module: nn.Module) -> bool:
    """Check if a module is a MoE Router.
    
    A Router is identified by having:
    - A 'weight' nn.Parameter attribute (gating weights)
    - A 'gating' callable method
    - A 'routing' callable method
    
    Args:
        module: The module to check.
        
    Returns:
        True if the module is a Router, False otherwise.
    """
    return (
        hasattr(module, 'weight')
        and isinstance(module.weight, nn.Parameter)
        and hasattr(module, 'gating')
        and callable(module.gating)
        and hasattr(module, 'routing')
        and callable(module.routing)
    )


class RouterLoRAAdapter(nn.Module):
    """
    LoRA adapter for MoE Router's gating weights.
    
    Router uses bare nn.Parameter for gating:
    - weight: [num_experts, hidden_size]
    - bias: [num_experts] (optional)
    
    Original gating computation:
        logits = input @ weight.T + bias
    
    With LoRA:
        logits = input @ weight.T + bias + scale * (input @ lora_A @ lora_B)
    
    This adapter:
    1. Creates lora_A and lora_B parameters
    2. Freezes the original router weights
    3. Monkey-patches the router's gating method to include LoRA
    4. Registers itself as a submodule of the router for checkpoint compatibility
    
    Args:
        router: The Router module to adapt.
        dim: LoRA rank dimension.
        alpha: LoRA scaling factor (scale = alpha / dim).
        dropout: Dropout probability.
        dropout_position: Where to apply dropout ('pre' or 'post' LoRA computation).
        lora_A_init_method: Initialization method for lora_A ('kaiming' or 'normal').
        lora_B_init_method: Initialization method for lora_B ('zero' or 'normal').
        lora_dtype: Data type for LoRA weights. If None, uses router.weight.dtype.
    """
    
    def __init__(
        self,
        router: nn.Module,
        dim: int = 32,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["kaiming", "normal", "xavier", "zero"] = "xavier",
        lora_B_init_method: Literal["zero", "normal", "xavier", "kaiming"] = "zero",
        lora_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Validate router
        if not is_router_module(router):
            raise ValueError(
                "Module is not a valid Router. "
                "Router must have 'weight' (nn.Parameter), 'gating' (method), and 'routing' (method)."
            )
        
        self.dim = dim
        self.alpha = alpha
        self.scale = alpha / dim
        self.dropout_position = dropout_position
        
        # Freeze original router weights
        router.weight.requires_grad = False
        if hasattr(router, 'bias') and router.bias is not None:
            router.bias.requires_grad = False
        
        # Get dimensions from router weight
        # weight shape: [num_experts, hidden_size]
        num_experts, hidden_size = router.weight.shape
        # Store reference to router config (used for dtype and sequence_parallel)
        self._router_config = getattr(router, 'config', None)
        
        # Determine dtype - KEEP FLOAT32 for Router LoRA for numerical stability
        # Router gating weights are critical and benefit from higher precision
        dtype = lora_dtype if lora_dtype is not None else torch.float32
        params_dtype = torch.float32  # Force float32 for router LoRA
        
        # Create LoRA parameters on CPU first (like Router.weight)
        # This ensures all ranks get identical weights via Megatron's global seed sync
        # lora_A: [hidden_size, rank] -> input @ lora_A = [tokens, rank]
        # lora_B: [rank, num_experts] -> intermediate @ lora_B = [tokens, num_experts]
        self.lora_A = nn.Parameter(torch.empty(hidden_size, dim, dtype=params_dtype))
        self.lora_B = nn.Parameter(torch.empty(dim, num_experts, dtype=params_dtype))
        
        # Initialize on CPU - relies on Megatron's global seed synchronization
        # All ranks have identical CPU RNG state from torch.manual_seed(seed)
        self._init_lora_weights(lora_A_init_method, lora_B_init_method)
        
        # No dtype conversion needed - already created with correct dtype
        
        # Set sequence_parallel attribute for proper gradient all-reduce in SP mode
        if self._router_config is not None and hasattr(self._router_config, 'sequence_parallel'):
            setattr(self.lora_A, 'sequence_parallel', self._router_config.sequence_parallel)
            setattr(self.lora_B, 'sequence_parallel', self._router_config.sequence_parallel)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Store original gating method
        self._orig_gating = router.gating
        
        # Monkey-patch router's gating method
        router.gating = self.gating_with_lora
        
        # Register this adapter as router's submodule for checkpoint save/load
        router.add_module('adapter', self)
    def _init_lora_weights(
        self,
        lora_A_init_method: str,
        lora_B_init_method: str,
    ) -> None:
        """Initialize LoRA weights on CPU.
        
        Initializes on CPU to leverage Megatron's global seed synchronization.
        All ranks have identical CPU RNG state from torch.manual_seed(seed),
        so the same initialization produces identical weights across all ranks.
        
        This is the same approach used by Router.weight initialization.
        
        Args:
            lora_A_init_method: Initialization method for lora_A.
            lora_B_init_method: Initialization method for lora_B.
        """
        init_fn_A = get_init_fn(lora_A_init_method)
        init_fn_B = get_init_fn(lora_B_init_method)
        
        # Initialize on CPU - all ranks get identical values due to global seed sync
        init_fn_A(self.lora_A)
        init_fn_B(self.lora_B)
        
    def _get_router_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """Get the dtype to use for router computation.
        
        Respects the router's moe_router_dtype config if available.
        
        Args:
            input_dtype: The input tensor's dtype.
            
        Returns:
            The dtype to use for computation.
        """
        if self._router_config is not None:
            moe_router_dtype = getattr(self._router_config, 'moe_router_dtype', None)
            if moe_router_dtype == 'fp32':
                return torch.float32
            elif moe_router_dtype == 'fp64':
                return torch.float64
        return input_dtype
    
    def compute_lora(self, input: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """Compute LoRA contribution to gating logits.
        
        Args:
            input: Input tensor of shape [..., hidden_size].
            target_dtype: The dtype to cast the output to.
            
        Returns:
            LoRA output tensor of shape [..., num_experts].
        """
        # Get computation dtype
        router_dtype = self._get_router_dtype(input.dtype)
        
        # Apply pre-dropout if configured
        lora_input = input.to(router_dtype)
        if self.dropout_position == "pre":
            lora_input = self.dropout(lora_input)
        
        # LoRA computation: input @ lora_A @ lora_B
        # input: [..., hidden_size]
        # lora_A: [hidden_size, rank]
        # lora_B: [rank, num_experts]
        # output: [..., num_experts]
        input_shape = lora_input.shape
        lora_input_2d = lora_input.view(-1, input_shape[-1])
        
        # LoRA computation using standard matmul
        # lora_A: [hidden_size, rank] -> input @ lora_A = [tokens, rank]
        # lora_B: [rank, num_experts] -> intermediate @ lora_B = [tokens, num_experts]
        lora_A = self.lora_A.to(router_dtype)
        lora_B = self.lora_B.to(router_dtype)
        intermediate = torch.mm(lora_input_2d, lora_A)  # [tokens, rank]
        lora_output = torch.mm(intermediate, lora_B)    # [tokens, num_experts]
        
        # Reshape back to original batch dimensions
        lora_output = lora_output.view(*input_shape[:-1], -1)
        lora_output = lora_output * self.scale
        
        # Apply post-dropout if configured
        if self.dropout_position == "post":
            lora_output = self.dropout(lora_output)
        
        return lora_output.to(target_dtype)
    
    def gating_with_lora(self, input: torch.Tensor) -> torch.Tensor:
        """Replacement gating method that includes LoRA adaptation.

        This method is monkey-patched onto the router to replace its original
        gating method.

        Args:
            input: Input tensor of shape [..., hidden_size].

        Returns:
            Logits tensor of shape [..., num_experts].
        """
        # Get original gating output
        logits = self._orig_gating(input)

        # Add LoRA contribution
        lora_output = self.compute_lora(input, logits.dtype)

        # Seg0 bypass: zero LoRA output at seg0 positions if mask is set
        seg1_mask = getattr(self, '_seg1_mask', None)
        if seg1_mask is not None:
            view_shape = (lora_output.shape[0],) + (1,) * (lora_output.dim() - 1)
            lora_output = lora_output * seg1_mask.view(view_shape).to(lora_output.dtype)

        return logits + lora_output
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute LoRA output only.
        
        This is mainly for standalone testing. In practice, gating_with_lora
        is called by the router.
        
        Args:
            input: Input tensor.
            
        Returns:
            LoRA output tensor.
        """
        return self.compute_lora(input, input.dtype)
    
    @torch.no_grad()
    def merge_weights(self, router: nn.Module) -> None:
        """Merge LoRA weights into the router's original weights.
        
        After merging, the router can be used without the LoRA adapter.
        
        The math:
        - Original: logits = input @ weight.T
        - With LoRA: logits = input @ weight.T + scale * (input @ lora_A @ lora_B)
        - Merged: logits = input @ (weight + scale * lora_B.T @ lora_A.T).T
        
        Args:
            router: The router module to merge weights into.
        """
        # LoRA adds: input @ lora_A @ lora_B to logits
        # Original: logits = input @ weight.T
        # To merge: weight.T += scale * lora_A @ lora_B, so weight += scale * (lora_A @ lora_B).T
        # delta_weight = scale * lora_B.T @ lora_A.T
        # lora_A: [hidden_size, rank], lora_A.T: [rank, hidden_size]
        # lora_B: [rank, num_experts], lora_B.T: [num_experts, rank]
        # delta_weight: [num_experts, hidden_size]
        delta_weight = self.scale * torch.matmul(self.lora_B.t(), self.lora_A.t())
        router.weight.data += delta_weight.to(router.weight.dtype)
    
    @torch.no_grad()
    def unmerge_weights(self, router: nn.Module) -> None:
        """Unmerge LoRA weights from the router's original weights.
        
        Args:
            router: The router module to unmerge weights from.
        """
        delta_weight = self.scale * torch.matmul(self.lora_B.t(), self.lora_A.t())
        router.weight.data -= delta_weight.to(router.weight.dtype)
    
    def restore_original_gating(self, router: nn.Module) -> None:
        """Restore the router's original gating method.
        
        Args:
            router: The router module to restore.
        """
        router.gating = self._orig_gating
