# SPDX-License-Identifier: Apache-2.0
"""Graph-safe gather for an unquantized BF16 paged indexer cache.

An unquantized cache has no scale bytes. Some callers nevertheless retain the
quantized-gather call shape and provide a float32 ``(num_tokens, 1)`` identity
scale workspace. This variant validates that workspace but performs only the
BF16 gather, without host synchronization.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cp_gather_indexer_k_bf16_kernel(
    k_cache_ptr,
    k_out_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_table_stride_batch: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    NUM_PHYSICAL_BLOCKS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES_PER_REQUEST: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    token_ids = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_ids < TOTAL_TOKENS

    # Resolve packed-token -> request entirely on device.  NUM_REQUESTS is a
    # shape-derived constexpr, so the loop is capture/replay stable and works
    # with repeated boundaries from zero-length requests.
    request_ids = tl.zeros((BLOCK_TOKENS,), dtype=tl.int32)
    for request in tl.static_range(1, NUM_REQUESTS):
        boundary = tl.load(cu_seqlen_ptr + request)
        request_ids += token_ids >= boundary

    request_starts = tl.load(
        cu_seqlen_ptr + request_ids,
        mask=token_mask,
        other=0,
    )
    request_offsets = token_ids - request_starts
    logical_pages = request_offsets // PAGE_SIZE
    page_offsets = request_offsets % PAGE_SIZE
    table_mask = token_mask & (logical_pages < MAX_PAGES_PER_REQUEST)
    physical_blocks = tl.load(
        block_table_ptr
        + request_ids * block_table_stride_batch
        + logical_pages * block_table_stride_page,
        mask=table_mask,
        other=-1,
    )
    valid_blocks = (
        table_mask & (physical_blocks >= 0) & (physical_blocks < NUM_PHYSICAL_BLOCKS)
    )

    dims = tl.arange(0, HEAD_DIM)
    physical_blocks_i64 = physical_blocks.to(tl.int64)
    page_offsets_i64 = page_offsets.to(tl.int64)
    token_ids_i64 = token_ids.to(tl.int64)
    source_offsets = (
        physical_blocks_i64[:, None] * cache_stride_block
        + page_offsets_i64[:, None] * cache_stride_token
        + dims[None, :] * cache_stride_dim
    )
    output_offsets = (
        token_ids_i64[:, None] * out_stride_token + dims[None, :] * out_stride_dim
    )
    mask = valid_blocks[:, None]
    values = tl.load(k_cache_ptr + source_offsets, mask=mask, other=0.0)
    tl.store(k_out_ptr + output_offsets, values, mask=token_mask[:, None])


def cp_gather_indexer_k_bf16_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor | None,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
) -> None:
    """Gather a paged BF16 indexer cache into the packed BF16 output."""

    if k_cache.dtype != torch.bfloat16 or k_cache.ndim != 3:
        raise ValueError("indexer cache must be a 3-D bfloat16 tensor")
    if k_fp8.dtype != torch.bfloat16 or k_fp8.ndim != 2:
        raise ValueError("gathered indexer keys must be a 2-D bfloat16 tensor")
    if k_cache.shape[-1] != k_fp8.shape[-1]:
        raise ValueError(
            f"Indexer head mismatch: cache={k_cache.shape[-1]} output={k_fp8.shape[-1]}"
        )
    if (
        k_fp8_scale is None
        or k_fp8_scale.dtype != torch.float32
        or k_fp8_scale.ndim != 2
        or k_fp8_scale.shape != (k_fp8.shape[0], 1)
    ):
        raise ValueError(
            "BF16 indexer gather requires a float32 identity-scale "
            "workspace shaped (num_tokens, 1)"
        )
    if block_table.ndim != 2 or cu_seqlen.ndim != 1:
        raise ValueError("block_table must be 2-D and cu_seqlen must be 1-D")
    if cu_seqlen.shape[0] != block_table.shape[0] + 1:
        raise ValueError("cu_seqlen length must equal batch size plus one")

    total_tokens = k_fp8.shape[0]
    if total_tokens == 0:
        return
    head_dim = k_fp8.shape[1]
    if head_dim > 128 or head_dim != triton.next_power_of_2(head_dim):
        raise ValueError(
            f"BF16 indexer gather requires power-of-two head_dim <=128, got {head_dim}"
        )

    block_tokens = 16 if total_tokens < 512 else 32
    grid = (triton.cdiv(total_tokens, block_tokens),)
    _cp_gather_indexer_k_bf16_kernel[grid](
        k_cache,
        k_fp8,
        block_table,
        cu_seqlen,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        TOTAL_TOKENS=total_tokens,
        NUM_REQUESTS=block_table.shape[0],
        NUM_PHYSICAL_BLOCKS=k_cache.shape[0],
        PAGE_SIZE=k_cache.shape[1],
        MAX_PAGES_PER_REQUEST=block_table.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )


__all__ = ["cp_gather_indexer_k_bf16_cache"]
