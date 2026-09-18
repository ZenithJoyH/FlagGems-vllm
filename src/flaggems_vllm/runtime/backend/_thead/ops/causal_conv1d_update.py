# SPDX-License-Identifier: Apache-2.0
"""Graph-safe causal-conv decode-state update for accelerator backends."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    state_indices_ptr,
    output_ptr,
    dim: tl.constexpr,
    width: tl.constexpr,
    null_block_id: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_width: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_output_batch: tl.constexpr,
    stride_output_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    request = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < dim
    state_id = tl.load(state_indices_ptr + request)
    valid_state = (state_id >= 0) & (state_id != null_block_id)

    value = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for position in range(width - 1):
        state_offsets = (
            state_id * stride_state_batch
            + offsets * stride_state_dim
            + position * stride_state_width
        )
        weight_offsets = offsets * stride_weight_dim + position * stride_weight_width
        state_value = tl.load(
            state_ptr + state_offsets,
            mask=mask & valid_state,
            other=0.0,
        ).to(tl.float32)
        weight_value = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        value += state_value * weight_value

    x_offsets = request * stride_x_batch + offsets * stride_x_dim
    current = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    final_weight_offsets = (
        offsets * stride_weight_dim + (width - 1) * stride_weight_width
    )
    final_weight = tl.load(weight_ptr + final_weight_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    value += current * final_weight
    if HAS_BIAS:
        value += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if SILU:
        value *= tl.sigmoid(value)

    output_offsets = request * stride_output_batch + offsets * stride_output_dim
    tl.store(output_ptr + output_offsets, value, mask=mask & valid_state)
    tl.store(output_ptr + output_offsets, 0.0, mask=mask & ~valid_state)

    # Shift the width-1 state left and append the current token. Each program
    # owns a disjoint dimension tile for one selected state row.
    for position in range(width - 2):
        source_offsets = (
            state_id * stride_state_batch
            + offsets * stride_state_dim
            + (position + 1) * stride_state_width
        )
        target_offsets = (
            state_id * stride_state_batch
            + offsets * stride_state_dim
            + position * stride_state_width
        )
        shifted = tl.load(
            state_ptr + source_offsets,
            mask=mask & valid_state,
            other=0.0,
        )
        tl.store(state_ptr + target_offsets, shifted, mask=mask & valid_state)
    final_state_offsets = (
        state_id * stride_state_batch
        + offsets * stride_state_dim
        + (width - 2) * stride_state_width
    )
    tl.store(state_ptr + final_state_offsets, current, mask=mask & valid_state)


@triton.jit
def _causal_conv1d_varlen_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    dim: tl.constexpr,
    width: tl.constexpr,
    num_states: tl.constexpr,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_width: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_output_dim: tl.constexpr,
    stride_output_token: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    SILU: tl.constexpr,
):
    request = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < dim
    start = tl.load(query_start_loc_ptr + request)
    end = tl.load(query_start_loc_ptr + request + 1)
    if HAS_STATE_INDICES:
        state_id = tl.load(state_indices_ptr + request)
    else:
        state_id = request
    valid_state = (
        (state_id >= 0)
        & (state_id < num_states)
        & (state_id != pad_slot_id)
        & (state_id != null_block_id)
    )
    if HAS_INITIAL_STATE:
        use_initial = tl.load(has_initial_state_ptr + request).to(tl.int1)
    else:
        use_initial = True

    state_base = state_id * stride_state_batch + offsets * stride_state_dim
    history0 = tl.load(
        state_ptr + state_base,
        mask=mask & valid_state & use_initial,
        other=0.0,
    ).to(tl.float32)
    history1 = tl.load(
        state_ptr + state_base + stride_state_width,
        mask=mask & valid_state & use_initial,
        other=0.0,
    ).to(tl.float32)
    history2 = tl.load(
        state_ptr + state_base + 2 * stride_state_width,
        mask=mask & valid_state & use_initial,
        other=0.0,
    ).to(tl.float32)

    token = start
    while token < end:
        current = tl.load(
            x_ptr + offsets * stride_x_dim + token * stride_x_token,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight0 = tl.load(
            weight_ptr + offsets * stride_weight_dim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight1 = tl.load(
            weight_ptr + offsets * stride_weight_dim + stride_weight_width,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight2 = tl.load(
            weight_ptr + offsets * stride_weight_dim + 2 * stride_weight_width,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value = history0 * weight0 + history1 * weight1 + history2 * weight2
        final_weight = tl.load(
            weight_ptr
            + offsets * stride_weight_dim
            + (width - 1) * stride_weight_width,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        value += current * final_weight
        if HAS_BIAS:
            value += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if SILU:
            value *= tl.sigmoid(value)
        output_offsets = offsets * stride_output_dim + token * stride_output_token
        tl.store(output_ptr + output_offsets, value, mask=mask & valid_state)
        tl.store(output_ptr + output_offsets, 0.0, mask=mask & ~valid_state)
        history0 = history1
        history1 = history2
        history2 = current
        token += 1

    tl.store(state_ptr + state_base, history0, mask=mask & valid_state)
    tl.store(
        state_ptr + state_base + stride_state_width, history1, mask=mask & valid_state
    )
    tl.store(
        state_ptr + state_base + 2 * stride_state_width,
        history2,
        mask=mask & valid_state,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    null_block_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
) -> torch.Tensor:
    """Run the plain decode subset without device-to-host metadata reads."""

    del max_query_len, validate_data
    unsupported = (
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )
    if any(value is not None for value in unsupported):
        raise NotImplementedError(
            "This causal-conv Triton update supports plain decode only"
        )
    if x.ndim != 2:
        raise ValueError(f"Expected decode x [batch, dim], got {tuple(x.shape)}")
    if conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "Expected conv_state [states, dim, width-1] and weight [dim, width]"
        )
    batch, dim = x.shape
    width = weight.shape[1]
    if width != 4 or conv_state.shape[1:] != (dim, width - 1):
        raise ValueError(
            "causal-conv update requires width=4 and matching state; "
            f"got x={tuple(x.shape)}, state={tuple(conv_state.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    if conv_state_indices is None:
        raise ValueError("graph-safe causal-conv requires device state indices")
    if conv_state_indices.ndim != 1 or conv_state_indices.shape[0] < batch:
        raise ValueError("conv_state_indices must provide one entry per request")
    if activation not in (None, False, "silu", "swish"):
        raise ValueError(f"Unsupported causal-conv activation: {activation!r}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != dim):
        raise ValueError("causal-conv bias must have shape [dim]")

    output = torch.empty_like(x)
    block_d = 256
    grid = (batch, triton.cdiv(dim, block_d))
    _causal_conv1d_update_kernel[grid](
        x,
        conv_state,
        weight,
        bias if bias is not None else x,
        conv_state_indices,
        output,
        dim=dim,
        width=width,
        null_block_id=null_block_id,
        stride_x_batch=x.stride(0),
        stride_x_dim=x.stride(1),
        stride_state_batch=conv_state.stride(0),
        stride_state_dim=conv_state.stride(1),
        stride_state_width=conv_state.stride(2),
        stride_weight_dim=weight.stride(0),
        stride_weight_width=weight.stride(1),
        stride_output_batch=output.stride(0),
        stride_output_dim=output.stride(1),
        BLOCK_D=block_d,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish", True),
        num_warps=4,
        num_stages=1,
    )
    return output


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
    """Run variable-length prefill without host reads of sequence metadata."""

    del block_size_to_align, metadata, validate_data
    if any(
        value is not None
        for value in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError("causal_conv1d_fn does not support APC metadata")
    if x.ndim != 2:
        raise ValueError(f"Expected x [dim, total_tokens], got {tuple(x.shape)}")
    if conv_states.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "Expected conv_states [states, dim, width-1] and weight [dim, width]"
        )
    dim, _ = x.shape
    width = weight.shape[1]
    if width != 4 or conv_states.shape[1:] != (dim, width - 1):
        raise ValueError(
            "causal_conv1d_fn requires width=4 and matching state; "
            f"got x={tuple(x.shape)}, state={tuple(conv_states.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    if query_start_loc.ndim != 1 or query_start_loc.numel() < 2:
        raise ValueError("query_start_loc must contain at least one sequence")
    requests = query_start_loc.numel() - 1
    if cache_indices is not None and (
        cache_indices.ndim != 1 or cache_indices.shape[0] < requests
    ):
        raise ValueError("cache_indices must provide one state per request")
    if has_initial_state is not None and (
        has_initial_state.ndim != 1 or has_initial_state.shape[0] < requests
    ):
        raise ValueError("has_initial_state must provide one flag per request")
    if activation not in (None, False, "silu", "swish", True):
        raise ValueError(f"Unsupported causal-conv activation: {activation!r}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != dim):
        raise ValueError("causal-conv bias must have shape [dim]")

    output = torch.empty_like(x)
    block_d = 256
    grid = (requests, triton.cdiv(dim, block_d))
    _causal_conv1d_varlen_kernel[grid](
        x,
        conv_states,
        weight,
        bias if bias is not None else x,
        query_start_loc,
        cache_indices if cache_indices is not None else query_start_loc,
        has_initial_state if has_initial_state is not None else query_start_loc,
        output,
        dim=dim,
        width=width,
        num_states=conv_states.shape[0],
        pad_slot_id=pad_slot_id,
        null_block_id=null_block_id,
        stride_x_dim=x.stride(0),
        stride_x_token=x.stride(1),
        stride_state_batch=conv_states.stride(0),
        stride_state_dim=conv_states.stride(1),
        stride_state_width=conv_states.stride(2),
        stride_weight_dim=weight.stride(0),
        stride_weight_width=weight.stride(1),
        stride_output_dim=output.stride(0),
        stride_output_token=output.stride(1),
        BLOCK_D=block_d,
        HAS_BIAS=bias is not None,
        HAS_STATE_INDICES=cache_indices is not None,
        HAS_INITIAL_STATE=has_initial_state is not None,
        SILU=activation in ("silu", "swish", True),
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
