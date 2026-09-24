# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-head Gemma RMS normalization for packed MiniMax-M3 Q/K tensors."""

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def gemma_qk_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    x_stride_token: tl.constexpr,
    x_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    x_head_offset: tl.constexpr,
    out_head_offset: tl.constexpr,
    eps: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 128)
    x_offset = token * x_stride_token + (
        x_head_offset + head * 128 + dim
    ) * x_stride_dim
    x = tl.load(x_ptr + x_offset).to(tl.float32)
    weight = tl.load(weight_ptr + dim).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / 128.0 + eps)
    normed = x * inv_rms * (1.0 + weight)
    out_offset = token * out_stride_token + (
        out_head_offset + head * 128 + dim
    ) * out_stride_dim
    tl.store(out_ptr + out_offset, normed)
