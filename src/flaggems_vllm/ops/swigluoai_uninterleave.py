# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed gate/up SwiGLU-OAI activation used by vLLM MoE and dense MLPs."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _swigluoai_uninterleave_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    output_width: tl.constexpr,
    clamp_limit: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // output_width
    column = offsets % output_width
    input_offsets = row * (2 * output_width) + column

    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + output_width, mask=mask, other=0
    ).to(tl.float32)
    gate = tl.minimum(gate, clamp_limit)
    up = tl.maximum(tl.minimum(up, clamp_limit), -clamp_limit)
    value = gate / (1.0 + tl.exp(-alpha * gate)) * (up + beta)
    tl.store(output_ptr + offsets, value, mask=mask)


def _storage_overlaps(input: torch.Tensor, out: torch.Tensor) -> bool:
    """Reject overlapping ranges, not disjoint views of one vLLM workspace."""
    if not torch._C._is_alias_of(input, out):
        return False
    if not out.is_contiguous():
        return True
    input_start = input.storage_offset()
    out_start = out.storage_offset()
    return (
        input_start < out_start + out.numel()
        and out_start < input_start + input.numel()
    )


def swigluoai_uninterleave(
    input: torch.Tensor,
    clamp_limit: float,
    alpha: float,
    beta: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return clamped SwiGLU-OAI or write it into an existing output buffer.

    This is inference-only. Gate and up occupy contiguous halves of each row;
    the name distinguishes that layout from interleaved MoE activations.
    """
    if input.ndim != 2 or input.shape[1] % 2:
        raise ValueError("input must be [tokens, 2 * width]")
    if input.device.type in ("cpu", "meta") or input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("input must be a supported accelerator float tensor")
    if not input.is_contiguous():
        raise NotImplementedError("input must be contiguous")
    if not all(math.isfinite(v) for v in (clamp_limit, alpha, beta)):
        raise ValueError("activation parameters must be finite")
    if clamp_limit < 0:
        raise ValueError("clamp_limit must be nonnegative")
    shape = (input.shape[0], input.shape[1] // 2)
    if out is None:
        out = torch.empty(shape, dtype=input.dtype, device=input.device)
    elif out.shape != shape or out.dtype != input.dtype or out.device != input.device:
        raise ValueError("out must match the expected shape, dtype and device")
    if out.numel() and _storage_overlaps(input, out):
        raise ValueError("out must not alias input")
    if not out.is_contiguous():
        raise NotImplementedError("out must be contiguous")
    if out.numel() == 0:
        return out

    # Fixed pointwise tile. No autotuning or workload-specific launch
    # branching is needed for this first portable implementation.
    block_size = 256
    with torch_device_fn.device(input.device):
        _swigluoai_uninterleave_kernel[
            (triton.cdiv(out.numel(), block_size),)
        ](
            input,
            out,
            out.numel(),
            shape[1],
            clamp_limit,
            alpha,
            beta,
            BLOCK_SIZE=block_size,
        )
    return out
