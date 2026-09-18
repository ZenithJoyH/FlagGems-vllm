# Copyright 2026 FlagOS Contributors
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

"""Shape-bounded causal-conv prefill selection for T-Head PPU."""

from __future__ import annotations

import torch

from flag_gems.fused.causal_conv1d_update import (
    causal_conv1d_fn as _flaggems_causal_conv1d_fn,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as _native_causal_conv1d_fn,
)


_NATIVE_MIN_TOKENS_PER_REQUEST = 128


def is_available() -> bool:
    return callable(_native_causal_conv1d_fn) and callable(
        _flaggems_causal_conv1d_fn
    )


def _use_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    metadata,
    block_idx_first_scheduled_token: torch.Tensor | None,
    block_idx_last_scheduled_token: torch.Tensor | None,
    initial_state_idx: torch.Tensor | None,
    num_computed_tokens: torch.Tensor | None,
) -> bool:
    requests = query_start_loc.numel() - 1
    return (
        requests > 0
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.shape[0] == 512
        and x.stride(0) == 1
        and x.stride(1) > 1
        and weight.ndim == 2
        and weight.shape == (512, 4)
        and weight.stride(1) == 1
        and conv_states.ndim == 3
        and conv_states.shape[1:] == (512, 3)
        and metadata is not None
        and block_idx_first_scheduled_token is None
        and block_idx_last_scheduled_token is None
        and initial_state_idx is None
        and num_computed_tokens is None
        and x.shape[1] >= _NATIVE_MIN_TOKENS_PER_REQUEST * requests
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: bool | str | None = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = -1,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
) -> torch.Tensor:
    implementation = (
        _native_causal_conv1d_fn
        if _use_native(
            x,
            weight,
            conv_states,
            query_start_loc,
            metadata=metadata,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens,
        )
        else _flaggems_causal_conv1d_fn
    )
    return implementation(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens,
        block_size_to_align=block_size_to_align,
        metadata=metadata,
        validate_data=validate_data,
    )


__all__ = ["causal_conv1d_fn", "is_available"]
