# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged K/V cache insertion for MiniMax-M3."""

import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def kv_cache_insert_kernel(
    qkv_ptr,
    slot_mapping_ptr,
    kv_cache_ptr,
    qkv_stride_token: tl.constexpr,
    qkv_stride_dim: tl.constexpr,
    kv_stride_block: tl.constexpr,
    kv_stride_kv: tl.constexpr,
    kv_stride_token: tl.constexpr,
    kv_stride_head: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    k_offset: tl.constexpr,
    v_offset: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 128)
    slot = tl.load(slot_mapping_ptr + token)
    valid = slot >= 0
    block = slot // block_size
    block_offset = slot - block * block_size

    row = token * qkv_stride_token
    k = tl.load(
        qkv_ptr + row + (k_offset + head * 128 + dim) * qkv_stride_dim
    )
    v = tl.load(
        qkv_ptr + row + (v_offset + head * 128 + dim) * qkv_stride_dim
    )
    cache_base = (
        block * kv_stride_block
        + block_offset * kv_stride_token
        + head * kv_stride_head
        + dim * kv_stride_dim
    )
    tl.store(kv_cache_ptr + cache_base, k, mask=valid)
    tl.store(kv_cache_ptr + cache_base + kv_stride_kv, v, mask=valid)
