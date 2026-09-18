# SPDX-License-Identifier: Apache-2.0
"""Graph-safe BF16 paged-MQA logits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_bf16_kernel(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n: tl.constexpr,
    max_model_len: tl.constexpr,
    stride_bt: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    logical_block = tl.program_id(1)
    context_len = tl.load(context_lens_ptr + row)
    first_position = logical_block * BLOCK_SIZE
    if first_position >= context_len:
        return

    heads = tl.arange(0, NUM_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    positions = tl.arange(0, BLOCK_SIZE)

    q_offsets = row * NUM_HEADS * HEAD_DIM + heads[:, None] * HEAD_DIM + dims[None, :]
    query = tl.load(q_ptr + q_offsets, eviction_policy="evict_last")
    query_t = tl.trans(query)
    weight = tl.load(
        weights_ptr + row * NUM_HEADS + heads,
        eviction_policy="evict_last",
    )

    request = row // next_n
    # Block IDs are int32, but a large KV cache can expose more than
    # 524,288 blocks. Cast before multiplying by BLOCK_SIZE * HEAD_DIM so the
    # element offset cannot overflow signed int32 at high physical block IDs.
    physical_block = tl.load(block_table_ptr + request * stride_bt + logical_block).to(
        tl.int64
    )
    # A graph replay can expose a transient -1/unallocated entry while the
    # scheduler advances a compressed kpool sequence across a page boundary.
    # Never form a device pointer from an invalid table entry. Keeping this as
    # a device-side mask preserves static launch shapes and graph replay.
    valid_block = (physical_block >= 0) & (physical_block < NUM_BLOCKS)
    safe_physical_block = tl.where(valid_block, physical_block, 0)
    key_offsets = (
        safe_physical_block * BLOCK_SIZE * HEAD_DIM
        + positions[:, None] * HEAD_DIM
        + dims[None, :]
    )
    keys = tl.load(
        kv_cache_ptr + key_offsets,
        mask=valid_block,
        other=0.0,
        eviction_policy="evict_first",
    )
    scores = tl.dot(keys, query_t)
    scores = tl.maximum(scores, 0.0)
    values = tl.sum(scores * weight[None, :], axis=1)

    valid = (first_position + positions < context_len) & valid_block
    output_offsets = row * max_model_len + first_position + positions
    tl.store(logits_ptr + output_offsets, values, mask=valid)


def bf16_paged_mqa_logits_graph_safe(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata,
    *,
    max_context_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute BF16 paged-MQA without capture-time host synchronization.

    The launch grid depends only on static tensor/configuration shapes. Runtime
    context lengths and block mappings remain device inputs, so a graph captured
    with an empty history can replay correctly with a populated history.
    """

    del schedule_metadata
    if q.ndim == 3:
        q = q.unsqueeze(1)
    if q.ndim != 4:
        raise ValueError(f"Expected q rank 4, got shape {tuple(q.shape)}")
    batch, next_n, num_heads, head_dim = q.shape
    if num_heads not in (32, 64) or head_dim != 128:
        raise ValueError(
            "BF16 paged-MQA currently supports H in (32, 64), D=128; "
            f"got H={num_heads}, D={head_dim}"
        )
    if q.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        raise ValueError("BF16 paged-MQA requires BF16 query and cache")
    if kv_cache.shape[-1] != head_dim:
        raise ValueError(
            f"Paged-MQA cache head dim must be {head_dim}, got {kv_cache.shape[-1]}"
        )
    if context_lens.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("context_lens and block_table must use int32")
    if not q.is_contiguous() or not weights.is_contiguous():
        raise ValueError("q and weights must be contiguous for graph-stable strides")

    total_rows = batch * next_n
    if context_lens.ndim == 2:
        flat_context_lens = context_lens.reshape(-1)[:total_rows]
    elif next_n == 1:
        flat_context_lens = context_lens.reshape(-1)[:total_rows]
    else:
        raise ValueError("1D context_lens is only supported when next_n=1")

    logits = torch.full(
        (total_rows, max_context_len),
        float("-inf") if clean_logits else 0.0,
        dtype=torch.float32,
        device=q.device,
    )
    if total_rows == 0 or max_context_len == 0:
        return logits

    block_size = kv_cache.shape[1]
    if block_size not in (16, 32, 64):
        raise ValueError(f"Unsupported BF16 paged-MQA block size: {block_size}")
    # A serving framework may narrow the graph input block table to the static
    # capacity required by a capture batch, rather than allocating
    # max_context_len / block_size columns. Launch exactly that stable
    # capacity; runtime context_lens keeps unused logical blocks inactive.
    max_blocks = block_table.shape[-1]
    if max_blocks == 0:
        raise ValueError("block_table must expose at least one logical block")

    grid = (total_rows, max_blocks)
    _paged_mqa_bf16_kernel[grid](
        q,
        kv_cache,
        weights,
        flat_context_lens,
        block_table,
        logits,
        next_n=next_n,
        max_model_len=max_context_len,
        stride_bt=block_table.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_BLOCKS=kv_cache.shape[0],
        num_warps=4 if num_heads == 32 else 8,
        num_stages=1,
    )
    return logits


__all__ = ["bf16_paged_mqa_logits_graph_safe"]
