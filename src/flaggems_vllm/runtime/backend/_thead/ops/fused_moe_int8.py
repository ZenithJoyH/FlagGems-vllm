# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact integer accumulation and measured GEMM1 tiles for T-Head 810E."""
import functools
from typing import Any, Optional

import torch
from triton import language as tl

_EXPECTED_CONFIGS = {
    64: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 3,
    },
    8192: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 3,
    },
}


@functools.lru_cache(maxsize=16)
def _supports_gemm1_tile(device_index: int) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    supported = (
        props.name == "PPU-ZW810E"
        and props.major == 8
        and props.minor == 0
        and props.warp_size == 32
    )
    return supported


def _gemm1_config(
    A,
    B,
    C,
    A_scale,
    B_scale,
    top_k,
    config,
    compute_type,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    block_shape,
    B_bias,
    FUSE_SILU,
    direct_sum,
    mul_routed_weight,
):
    m = A.size(0)
    if not (
        0 < m <= 8192
        and "INT8_ACCUMULATOR" not in config
        and not config.get("SWAP_AB", False)
        and A.device.type == "cuda"
        and A.dtype == torch.int8
        and B.dtype == torch.int8
        and C.dtype == torch.bfloat16
        and compute_type == tl.bfloat16
        and tuple(A.shape) == (m, 6144)
        and tuple(B.shape) == (128, 768, 6144)
        and tuple(C.shape) == (m, 4, 768)
        and top_k == 4
        and use_int8_w8a8
        and per_channel_quant
        and not any(
            (
                use_fp8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                FUSE_SILU,
                direct_sum,
                mul_routed_weight,
            )
        )
        and block_shape is None
        and B_bias is None
        and A_scale is not None
        and B_scale is not None
        and A_scale.dtype == B_scale.dtype == torch.float32
        and tuple(A_scale.shape) == (m, 1)
        and tuple(B_scale.shape) == (128, 768, 1)
        and all(
            t.device == A.device and t.is_contiguous()
            for t in (A, B, C, A_scale, B_scale)
        )
    ):
        return config
    if not _supports_gemm1_tile(A.device.index):
        return config
    tuned = dict(config)
    tuned["INT8_ACCUMULATOR"] = True
    if config == _EXPECTED_CONFIGS.get(m):
        tuned["BLOCK_SIZE_K"] = 256
    return tuned


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: torch.Tensor | None = None,
    FUSE_SILU: bool = False,
    direct_sum: bool = False,
    out_top_k: int = 1,
) -> None:
    # Vendor modules may be discovered while the generic package imports.
    from flaggems_vllm.ops.fused_moe import (
        invoke_fused_moe_triton_kernel as _generic_invoke,
    )

    config = _gemm1_config(
        A,
        B,
        C,
        A_scale,
        B_scale,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape,
        B_bias,
        FUSE_SILU,
        direct_sum,
        mul_routed_weight,
    )
    return _generic_invoke(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
        FUSE_SILU=FUSE_SILU,
        direct_sum=direct_sum,
        out_top_k=out_top_k,
    )
