# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged sparse-index cache insertion for MiniMax-M3."""

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def index_cache_insert_kernel(
    qkv_ptr,
    slot_mapping_ptr,
    index_cache_ptr,
    qkv_stride_token: tl.constexpr,
    qkv_stride_dim: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    index_k_offset: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.arange(0, 128)
    slot = tl.load(slot_mapping_ptr + token)
    valid = slot >= 0
    block = slot // block_size
    block_offset = slot - block * block_size
    value = tl.load(
        qkv_ptr
        + token * qkv_stride_token
        + (index_k_offset + dim) * qkv_stride_dim
    )
    cache_offset = (
        block * cache_stride_block
        + block_offset * cache_stride_token
        + dim * cache_stride_dim
    )
    tl.store(index_cache_ptr + cache_offset, value, mask=valid)
