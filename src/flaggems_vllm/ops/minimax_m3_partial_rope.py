# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""64-dimensional partial NeoX RoPE for MiniMax-M3 Q/K heads."""

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def partial_rope_64_kernel(
    x_ptr,
    cos_sin_ptr,
    positions_ptr,
    x_stride_token: tl.constexpr,
    x_stride_dim: tl.constexpr,
    head_offset: tl.constexpr,
    cos_stride_pos: tl.constexpr,
    cos_stride_dim: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 32)
    base = token * x_stride_token + (head_offset + head * 128) * x_stride_dim
    first = tl.load(x_ptr + base + dim * x_stride_dim).to(tl.float32)
    second = tl.load(x_ptr + base + (32 + dim) * x_stride_dim).to(tl.float32)
    position = tl.load(positions_ptr + token)
    cos = tl.load(cos_sin_ptr + position * cos_stride_pos + dim * cos_stride_dim)
    sin = tl.load(
        cos_sin_ptr + position * cos_stride_pos + (32 + dim) * cos_stride_dim
    )
    tl.store(x_ptr + base + dim * x_stride_dim, first * cos - second * sin)
    tl.store(x_ptr + base + (32 + dim) * x_stride_dim, second * cos + first * sin)
