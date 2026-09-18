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

"""T-Head specializations for small-head sparse MLA."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.flashmla_sparse import (
    flash_mla_sparse_fwd as _generic_flash_mla_sparse_fwd,
)
from flaggems_vllm.ops.flashmla_sparse import (
    triton_flash_mla_sparse_fwd as _triton_flash_mla_sparse_fwd,
)


_SPLITK_HQ = 4
_SPLITK_DQK = 576
_SPLITK_DV = 512
_SPLITK_TOPK = 2048
_SPLITK_BK = 64
_SPLITK_BDP = 256
_SPLITK_MAX_SQ = 64
_HQ4_PREFILL_MIN_SQ = 128
_HQ4_PREFILL_MAX_SQ = 2048


@triton.jit
def _flash_mla_sparse_splitk_stage1(
    q,
    kv,
    indices,
    topk_length,
    partial_acc,
    partial_max,
    partial_sum,
    stride_qm,
    stride_qh,
    stride_kvn,
    stride_tm,
    SQ,
    SKV,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    DV: tl.constexpr,
    TOPK: tl.constexpr,
    BDP: tl.constexpr,
    SM_SCALE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    query_id = pid // NUM_SPLITS
    split_id = pid % NUM_SPLITS
    offs_h = tl.arange(0, HQ)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BK)
    offs_td = tl.arange(0, 64)

    local_max = tl.full([HQ], float("-inf"), tl.float32)
    local_sum = tl.zeros([HQ], tl.float32)
    acc0 = tl.zeros([HQ, BDP], tl.float32)
    acc1 = tl.zeros([HQ, BDP], tl.float32)
    split_size: tl.constexpr = TOPK // NUM_SPLITS
    blocks_per_split: tl.constexpr = split_size // BK
    split_start = split_id * split_size
    valid_topk = tl.load(topk_length + query_id)
    index_base = indices + query_id * stride_tm

    if split_start < valid_topk:
        q_base = q + query_id * stride_qm
        q0 = tl.load(q_base + offs_h[:, None] * stride_qh + offs_d[None, :])
        q1 = tl.load(
            q_base + offs_h[:, None] * stride_qh + BDP + offs_d[None, :]
        )
        qt = tl.load(
            q_base + offs_h[:, None] * stride_qh + DV + offs_td[None, :]
        )

        for block_id in range(blocks_per_split):
            positions = split_start + block_id * BK + offs_t
            position_mask = positions < valid_topk
            kv_ids = tl.load(index_base + positions, mask=position_mask, other=-1)
            valid_ids = position_mask & (kv_ids >= 0) & (kv_ids < SKV)
            safe_ids = tl.where(valid_ids, kv_ids, 0)
            kv0 = tl.load(
                kv + safe_ids[None, :] * stride_kvn + offs_d[:, None],
                cache_modifier=".cg",
            )
            kv1 = tl.load(
                kv
                + safe_ids[None, :] * stride_kvn
                + BDP
                + offs_d[:, None],
                cache_modifier=".cg",
            )
            kvt = tl.load(
                kv
                + safe_ids[None, :] * stride_kvn
                + DV
                + offs_td[:, None],
                cache_modifier=".cg",
            )
            qk = tl.dot(q0, kv0, out_dtype=tl.float32)
            qk = tl.dot(q1, kv1, qk, out_dtype=tl.float32)
            qk = tl.dot(qt, kvt, qk, out_dtype=tl.float32) * SM_SCALE
            qk = tl.where(valid_ids[None, :], qk, float("-inf"))
            new_max = tl.maximum(local_max, tl.max(qk, axis=1))
            exp_qk = tl.where(
                valid_ids[None, :], tl.exp(qk - new_max[:, None]), 0.0
            )
            alpha = tl.where(
                local_max != float("-inf"), tl.exp(local_max - new_max), 0.0
            )
            local_sum = local_sum * alpha + tl.sum(exp_qk, axis=1)
            acc0 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv0.trans(),
                acc0 * alpha[:, None],
                out_dtype=tl.float32,
            )
            acc1 = tl.dot(
                exp_qk.to(tl.bfloat16),
                kv1.trans(),
                acc1 * alpha[:, None],
                out_dtype=tl.float32,
            )
            local_max = new_max

    stat_base = (query_id * NUM_SPLITS + split_id) * HQ + offs_h
    tl.store(partial_max + stat_base, local_max)
    tl.store(partial_sum + stat_base, local_sum)
    acc_base = (
        (query_id * NUM_SPLITS + split_id) * HQ + offs_h[:, None]
    ) * DV
    tl.store(partial_acc + acc_base + offs_d[None, :], acc0)
    tl.store(partial_acc + acc_base + BDP + offs_d[None, :], acc1)


@triton.jit
def _flash_mla_sparse_splitk_stage2(
    partial_acc,
    partial_max,
    partial_sum,
    attn_sink,
    output,
    max_logits,
    lse,
    HQ: tl.constexpr,
    DV: tl.constexpr,
    BDP: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    d_block = pid % 2
    head = (pid // 2) % HQ
    query_id = pid // (2 * HQ)
    offs_s = tl.arange(0, NUM_SPLITS)
    stat_offsets = (query_id * NUM_SPLITS + offs_s) * HQ + head
    split_max = tl.load(partial_max + stat_offsets)
    global_max = tl.max(split_max, axis=0)
    valid = global_max != float("-inf")
    weights = tl.where(
        split_max != float("-inf"), tl.exp(split_max - global_max), 0.0
    )
    split_sum = tl.load(partial_sum + stat_offsets)
    merged_sum = tl.sum(split_sum * weights, axis=0)
    denominator = merged_sum + tl.exp(tl.load(attn_sink + head) - global_max)

    offs_d = d_block * BDP + tl.arange(0, BDP)
    acc_offsets = (
        ((query_id * NUM_SPLITS + offs_s[:, None]) * HQ + head) * DV
        + offs_d[None, :]
    )
    partial = tl.load(partial_acc + acc_offsets)
    merged = tl.sum(partial * weights[:, None], axis=0) / denominator
    merged = tl.where(valid, merged, 0.0)
    output_offsets = (query_id * HQ + head) * DV + offs_d
    tl.store(output + output_offsets, merged.to(tl.bfloat16))
    if d_block == 0:
        stat_output_offset = query_id * HQ + head
        tl.store(
            max_logits + stat_output_offset,
            tl.where(valid, global_max, float("-inf")),
        )
        merged_lse = global_max + tl.log(merged_sum)
        tl.store(
            lse + stat_output_offset,
            tl.where(valid, merged_lse, float("inf")),
        )


def _can_use_thead_splitk(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> bool:
    return (
        q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
        and q.is_contiguous()
        and kv.is_contiguous()
        and indices.is_contiguous()
        and q.ndim == 3
        and kv.ndim == 3
        and indices.ndim == 3
        and 0 < q.shape[0] <= _SPLITK_MAX_SQ
        and q.shape[1:] == (_SPLITK_HQ, _SPLITK_DQK)
        and kv.shape[1:] == (1, _SPLITK_DQK)
        and indices.shape == (q.shape[0], 1, _SPLITK_TOPK)
        and d_v == _SPLITK_DV
        and attn_sink is not None
        and attn_sink.dtype == torch.float32
        and attn_sink.is_contiguous()
        and attn_sink.shape == (_SPLITK_HQ,)
        and topk_length is not None
        and topk_length.dtype == torch.int32
        and topk_length.is_contiguous()
        and topk_length.shape == (q.shape[0],)
    )


def _can_use_thead_hq4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
) -> bool:
    return (
        q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
        and q.is_contiguous()
        and kv.is_contiguous()
        and indices.is_contiguous()
        and _HQ4_PREFILL_MIN_SQ <= q.shape[0] <= _HQ4_PREFILL_MAX_SQ
        and q.shape[1:] == (_SPLITK_HQ, _SPLITK_DQK)
        and kv.ndim == 3
        and kv.shape[1:] == (1, _SPLITK_DQK)
        and indices.shape == (q.shape[0], 1, _SPLITK_TOPK)
        and d_v == _SPLITK_DV
        and attn_sink is not None
        and attn_sink.dtype == torch.float32
        and attn_sink.is_contiguous()
        and attn_sink.shape == (_SPLITK_HQ,)
        and topk_length is not None
        and topk_length.dtype == torch.int32
        and topk_length.is_contiguous()
        and topk_length.shape == (q.shape[0],)
    )


def _flash_mla_sparse_hq4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sq = q.shape[0]
    skv = kv.shape[0]
    output = torch.empty(
        (sq, _SPLITK_HQ, _SPLITK_DV), device=q.device, dtype=q.dtype
    )
    max_logits = torch.empty(
        (sq, _SPLITK_HQ), device=q.device, dtype=torch.float32
    )
    lse = torch.empty_like(max_logits)
    _triton_flash_mla_sparse_fwd.fn[(sq,)](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        sq,
        _SPLITK_HQ,
        _SPLITK_DQK,
        skv,
        _SPLITK_TOPK,
        True,
        True,
        BK=32,
        BH=_SPLITK_HQ,
        num_warps=4,
        num_stages=1,
    )
    return output, max_logits, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use T-Head HY4 specializations and preserve the generic fallback."""
    if _can_use_thead_hq4_prefill(
        q, kv, indices, d_v, attn_sink, topk_length
    ):
        return _flash_mla_sparse_hq4_prefill(
            q, kv, indices, sm_scale, attn_sink, topk_length
        )

    if not _can_use_thead_splitk(
        q, kv, indices, d_v, attn_sink, topk_length
    ):
        return _generic_flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

    sq = q.shape[0]
    skv = kv.shape[0]
    num_splits = 16 if sq <= 32 else 8
    partial_acc = torch.empty(
        (sq, num_splits, _SPLITK_HQ, _SPLITK_DV),
        device=q.device,
        dtype=torch.float32,
    )
    partial_max = torch.empty(
        (sq, num_splits, _SPLITK_HQ),
        device=q.device,
        dtype=torch.float32,
    )
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty(
        (sq, _SPLITK_HQ, _SPLITK_DV), device=q.device, dtype=q.dtype
    )
    max_logits = torch.empty(
        (sq, _SPLITK_HQ), device=q.device, dtype=torch.float32
    )
    lse = torch.empty_like(max_logits)

    _flash_mla_sparse_splitk_stage1[(sq * num_splits,)](
        q,
        kv,
        indices,
        topk_length,
        partial_acc,
        partial_max,
        partial_sum,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        sq,
        skv,
        HQ=_SPLITK_HQ,
        DQK=_SPLITK_DQK,
        DV=_SPLITK_DV,
        TOPK=_SPLITK_TOPK,
        BDP=_SPLITK_BDP,
        SM_SCALE=sm_scale,
        NUM_SPLITS=num_splits,
        BK=_SPLITK_BK,
        num_warps=4,
        num_stages=2,
    )
    _flash_mla_sparse_splitk_stage2[(sq * _SPLITK_HQ * 2,)](
        partial_acc,
        partial_max,
        partial_sum,
        attn_sink,
        output,
        max_logits,
        lse,
        HQ=_SPLITK_HQ,
        DV=_SPLITK_DV,
        BDP=_SPLITK_BDP,
        NUM_SPLITS=num_splits,
        num_warps=4,
        num_stages=1,
    )
    return output, max_logits, lse


__all__ = ["flash_mla_sparse_fwd"]
