# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"TLE-free block alignment for the T-Head compiler backend."

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_alignment_workspace(
    cumsum, counts, num_experts: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(cumsum + offsets, 0, mask=offsets < num_experts + 1)
    tl.store(counts + offsets, 0, mask=offsets < (num_experts + 1) * num_experts)


def moe_align_block_size_triton(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
):
    from flaggems_vllm.ops.moe_align_block_size import (
        _moe_align_block_size_triton_4stage,
    )

    if topk_ids.numel() == 0:
        _zero_alignment_workspace[(1,)](
            num_tokens_post_pad,
            num_tokens_post_pad,
            0,
            BLOCK=256,
        )
        return

    cumsum = torch.empty((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    counts = torch.empty(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    # Small scratch initialization: fixed tile, no torch compute/copy fallback.
    _zero_alignment_workspace[(triton.cdiv((num_experts + 1) * num_experts, 256),)](
        cumsum,
        counts,
        num_experts,
        BLOCK=256,
    )
    return _moe_align_block_size_triton_4stage(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        workspace=(cumsum, counts),
    )
