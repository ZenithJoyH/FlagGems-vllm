# SPDX-License-Identifier: Apache-2.0
"""Device-side token mapping helpers for pooled sparse indexers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_pools_to_tokens_kernel(
    group_ids_ptr,
    group_valid_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    output_ptr,
    groups: tl.constexpr,
    topk: tl.constexpr,
    pool_size: tl.constexpr,
    page_cols: tl.constexpr,
    stride_group_row: tl.constexpr,
    stride_group_col: tl.constexpr,
    stride_valid_row: tl.constexpr,
    stride_valid_col: tl.constexpr,
    stride_page_row: tl.constexpr,
    stride_page_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_col: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < topk
    group_columns = columns // pool_size
    group_ids = tl.load(
        group_ids_ptr + row * stride_group_row + group_columns * stride_group_col,
        mask=mask & (group_columns < groups),
        other=-1,
    ).to(tl.int64)
    valid = tl.load(
        group_valid_ptr + row * stride_valid_row + group_columns * stride_valid_col,
        mask=mask & (group_columns < groups),
        other=0,
    ).to(tl.int1)
    token_ids = group_ids * pool_size + columns % pool_size
    if HAS_PAGE_TABLE:
        safe_ids = tl.maximum(0, tl.minimum(token_ids, page_cols - 1))
        mapped = tl.load(
            page_table_ptr + row * stride_page_row + safe_ids * stride_page_col,
            mask=mask,
            other=-1,
        ).to(tl.int32)
    elif HAS_TOPK_OFFSETS:
        mapped = (token_ids + tl.load(topk_offsets_ptr + row)).to(tl.int32)
    else:
        mapped = token_ids.to(tl.int32)
    mapped = tl.where(valid, mapped, -1)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_col,
        mapped,
        mask=mask,
    )


@triton.jit
def _append_tail_to_topk_kernel(
    topk_result_ptr,
    seq_lens_ptr,
    pool_lens_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    output_ptr,
    history_len: tl.constexpr,
    out_cols: tl.constexpr,
    pool_size: tl.constexpr,
    page_cols: tl.constexpr,
    stride_topk_row: tl.constexpr,
    stride_topk_col: tl.constexpr,
    stride_page_row: tl.constexpr,
    stride_page_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_col: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < out_cols
    is_history = columns < history_len
    if history_len > 0:
        history = tl.load(
            topk_result_ptr
            + row * stride_topk_row
            + tl.minimum(columns, history_len - 1) * stride_topk_col,
            mask=mask,
            other=-1,
        ).to(tl.int32)
    else:
        history = tl.full((BLOCK,), -1, tl.int32)
    tail_start = tl.load(pool_lens_ptr + row).to(tl.int64) * pool_size
    tail_offset = columns - history_len
    tail_count = tl.load(seq_lens_ptr + row).to(tl.int64) - tail_start
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = tail_start + tail_offset
    if HAS_PAGE_TABLE:
        safe_tail = tl.maximum(0, tl.minimum(tail_raw, page_cols - 1))
        tail = tl.load(
            page_table_ptr + row * stride_page_row + safe_tail * stride_page_col,
            mask=mask,
            other=-1,
        ).to(tl.int32)
    elif HAS_TOPK_OFFSETS:
        tail = (tail_raw + tl.load(topk_offsets_ptr + row)).to(tl.int32)
    else:
        tail = tail_raw.to(tl.int32)
    value = tl.where(is_history, history, -1)
    value = tl.where(is_tail, tail, value)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_col,
        value,
        mask=mask,
    )


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand each selected pool into its member token indices."""
    if topk % pool_size:
        raise ValueError("topk must be divisible by pool_size")
    if group_ids.ndim != 2 or group_valid.shape != group_ids.shape:
        raise ValueError("group_ids and group_valid must have matching rank-2 shapes")
    if group_ids.shape[1] != topk // pool_size:
        raise ValueError("group_ids width must equal topk // pool_size")
    if page_table is not None and topk_offsets is not None:
        raise ValueError("page_table and topk_offsets are mutually exclusive")
    rows, groups = group_ids.shape
    output = torch.empty((rows, topk), device=group_ids.device, dtype=torch.int32)
    block = min(256, triton.next_power_of_2(topk))
    grid = (rows, triton.cdiv(topk, block))
    _expand_pools_to_tokens_kernel[grid](
        group_ids,
        group_valid,
        page_table if page_table is not None else group_ids,
        topk_offsets if topk_offsets is not None else group_ids,
        output,
        groups=groups,
        topk=topk,
        pool_size=pool_size,
        page_cols=page_table.shape[1] if page_table is not None else 1,
        stride_group_row=group_ids.stride(0),
        stride_group_col=group_ids.stride(1),
        stride_valid_row=group_valid.stride(0),
        stride_valid_col=group_valid.stride(1),
        stride_page_row=page_table.stride(0) if page_table is not None else 0,
        stride_page_col=page_table.stride(1) if page_table is not None else 0,
        stride_output_row=output.stride(0),
        stride_output_col=output.stride(1),
        BLOCK=block,
        HAS_PAGE_TABLE=page_table is not None,
        HAS_TOPK_OFFSETS=topk_offsets is not None,
        num_warps=4,
        num_stages=1,
    )
    return output


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append the incomplete final pool to selected historical token indices."""
    if pool_size == 1:
        return topk_result
    if topk_result.ndim != 2:
        raise ValueError("topk_result must have rank 2")
    if page_table is not None and topk_offsets is not None:
        raise ValueError("page_table and topk_offsets are mutually exclusive")
    rows, history_len = topk_result.shape
    if seq_lens.numel() < rows or pool_lens.numel() < rows:
        raise ValueError("seq_lens and pool_lens must provide one value per row")
    out_cols = history_len + pool_size - 1
    output = torch.empty((rows, out_cols), device=topk_result.device, dtype=torch.int32)
    block = min(256, triton.next_power_of_2(out_cols))
    grid = (rows, triton.cdiv(out_cols, block))
    _append_tail_to_topk_kernel[grid](
        topk_result,
        seq_lens,
        pool_lens,
        page_table if page_table is not None else topk_result,
        topk_offsets if topk_offsets is not None else topk_result,
        output,
        history_len=history_len,
        out_cols=out_cols,
        pool_size=pool_size,
        page_cols=page_table.shape[1] if page_table is not None else 1,
        stride_topk_row=topk_result.stride(0),
        stride_topk_col=topk_result.stride(1),
        stride_page_row=page_table.stride(0) if page_table is not None else 0,
        stride_page_col=page_table.stride(1) if page_table is not None else 0,
        stride_output_row=output.stride(0),
        stride_output_col=output.stride(1),
        BLOCK=block,
        HAS_PAGE_TABLE=page_table is not None,
        HAS_TOPK_OFFSETS=topk_offsets is not None,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["append_tail_to_topk", "expand_pools_to_tokens"]
