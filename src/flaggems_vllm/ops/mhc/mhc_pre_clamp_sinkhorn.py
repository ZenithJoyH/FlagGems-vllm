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

"""mHC pre operator used by the clamp-based XingChen4 checkpoint format."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_pre_clamp_sinkhorn_gemm"),
    key=["num_tokens", "hidden_size"],
    strategy=["log", "align32"],
)
@triton.jit
def _rms_bf16_gemm_kernel(
    residual_ptr,
    fn_ptr,
    mixes_ptr,
    num_tokens,
    hidden_size,
    rms_eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute linear(RMSNorm(residual).bf16, fn.bf16) in FP32."""
    hc_hidden_size = 4 * hidden_size
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < num_tokens

    sq_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, hc_hidden_size, BLOCK_K):
        k = k_start + offs_k
        residual = tl.load(
            residual_ptr + offs_m[:, None] * hc_hidden_size + k[None, :],
            mask=mask_m[:, None] & (k[None, :] < hc_hidden_size),
            other=0.0,
        ).to(tl.float32)
        sq_sum += tl.sum(residual * residual, axis=1)
    rms_inv = tl.rsqrt(sq_sum / hc_hidden_size + rms_eps)

    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, hc_hidden_size, BLOCK_K):
        k = k_start + offs_k
        mask_k = k < hc_hidden_size
        residual = tl.load(
            residual_ptr + offs_m[:, None] * hc_hidden_size + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        residual = (residual * rms_inv[:, None]).to(tl.bfloat16)
        weight = tl.load(
            fn_ptr + offs_n[:, None] * hc_hidden_size + k[None, :],
            mask=(offs_n[:, None] < 24) & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(residual, tl.trans(weight))

    tl.store(
        mixes_ptr + offs_m[:, None] * 24 + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & (offs_n[None, :] < 24),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_pre_clamp_sinkhorn_epilogue"),
    key=["num_tokens", "hidden_size"],
    strategy=["log", "align32"],
)
@triton.jit
def _mix_sinkhorn_reduce_kernel(
    residual_ptr,
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    post_ptr,
    comb_ptr,
    layer_input_ptr,
    num_tokens,
    hidden_size,
    hc_hidden_size,
    sinkhorn_eps: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Apply mix activations, clamped Sinkhorn, and the residual reduction."""
    token = tl.program_id(0)
    if token >= num_tokens:
        return

    mix_offset = token * 24
    scale_pre = tl.load(hc_scale_ptr)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    pre_0 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset) * scale_pre + tl.load(hc_base_ptr)
    ).to(tl.bfloat16)
    pre_1 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 1) * scale_pre
        + tl.load(hc_base_ptr + 1)
    ).to(tl.bfloat16)
    pre_2 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 2) * scale_pre
        + tl.load(hc_base_ptr + 2)
    ).to(tl.bfloat16)
    pre_3 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 3) * scale_pre
        + tl.load(hc_base_ptr + 3)
    ).to(tl.bfloat16)

    post_0 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 4) * scale_post
        + tl.load(hc_base_ptr + 4)
    ) * hc_post_mult_value
    post_1 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 5) * scale_post
        + tl.load(hc_base_ptr + 5)
    ) * hc_post_mult_value
    post_2 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 6) * scale_post
        + tl.load(hc_base_ptr + 6)
    ) * hc_post_mult_value
    post_3 = tl.sigmoid(
        tl.load(mixes_ptr + mix_offset + 7) * scale_post
        + tl.load(hc_base_ptr + 7)
    ) * hc_post_mult_value
    post_offset = token * 4
    tl.store(post_ptr + post_offset, post_0)
    tl.store(post_ptr + post_offset + 1, post_1)
    tl.store(post_ptr + post_offset + 2, post_2)
    tl.store(post_ptr + post_offset + 3, post_3)

    comb_offset = mix_offset + 8
    cm_00 = tl.clamp(
        tl.load(mixes_ptr + comb_offset) * scale_comb + tl.load(hc_base_ptr + 8),
        clamp_min,
        clamp_max,
    )
    cm_01 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 1) * scale_comb
        + tl.load(hc_base_ptr + 9),
        clamp_min,
        clamp_max,
    )
    cm_02 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 2) * scale_comb
        + tl.load(hc_base_ptr + 10),
        clamp_min,
        clamp_max,
    )
    cm_03 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 3) * scale_comb
        + tl.load(hc_base_ptr + 11),
        clamp_min,
        clamp_max,
    )
    cm_10 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 4) * scale_comb
        + tl.load(hc_base_ptr + 12),
        clamp_min,
        clamp_max,
    )
    cm_11 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 5) * scale_comb
        + tl.load(hc_base_ptr + 13),
        clamp_min,
        clamp_max,
    )
    cm_12 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 6) * scale_comb
        + tl.load(hc_base_ptr + 14),
        clamp_min,
        clamp_max,
    )
    cm_13 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 7) * scale_comb
        + tl.load(hc_base_ptr + 15),
        clamp_min,
        clamp_max,
    )
    cm_20 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 8) * scale_comb
        + tl.load(hc_base_ptr + 16),
        clamp_min,
        clamp_max,
    )
    cm_21 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 9) * scale_comb
        + tl.load(hc_base_ptr + 17),
        clamp_min,
        clamp_max,
    )
    cm_22 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 10) * scale_comb
        + tl.load(hc_base_ptr + 18),
        clamp_min,
        clamp_max,
    )
    cm_23 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 11) * scale_comb
        + tl.load(hc_base_ptr + 19),
        clamp_min,
        clamp_max,
    )
    cm_30 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 12) * scale_comb
        + tl.load(hc_base_ptr + 20),
        clamp_min,
        clamp_max,
    )
    cm_31 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 13) * scale_comb
        + tl.load(hc_base_ptr + 21),
        clamp_min,
        clamp_max,
    )
    cm_32 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 14) * scale_comb
        + tl.load(hc_base_ptr + 22),
        clamp_min,
        clamp_max,
    )
    cm_33 = tl.clamp(
        tl.load(mixes_ptr + comb_offset + 15) * scale_comb
        + tl.load(hc_base_ptr + 23),
        clamp_min,
        clamp_max,
    )

    row_max_0 = tl.maximum(tl.maximum(cm_00, cm_01), tl.maximum(cm_02, cm_03))
    row_max_1 = tl.maximum(tl.maximum(cm_10, cm_11), tl.maximum(cm_12, cm_13))
    row_max_2 = tl.maximum(tl.maximum(cm_20, cm_21), tl.maximum(cm_22, cm_23))
    row_max_3 = tl.maximum(tl.maximum(cm_30, cm_31), tl.maximum(cm_32, cm_33))
    cm_00, cm_01, cm_02, cm_03 = (
        tl.exp(cm_00 - row_max_0),
        tl.exp(cm_01 - row_max_0),
        tl.exp(cm_02 - row_max_0),
        tl.exp(cm_03 - row_max_0),
    )
    cm_10, cm_11, cm_12, cm_13 = (
        tl.exp(cm_10 - row_max_1),
        tl.exp(cm_11 - row_max_1),
        tl.exp(cm_12 - row_max_1),
        tl.exp(cm_13 - row_max_1),
    )
    cm_20, cm_21, cm_22, cm_23 = (
        tl.exp(cm_20 - row_max_2),
        tl.exp(cm_21 - row_max_2),
        tl.exp(cm_22 - row_max_2),
        tl.exp(cm_23 - row_max_2),
    )
    cm_30, cm_31, cm_32, cm_33 = (
        tl.exp(cm_30 - row_max_3),
        tl.exp(cm_31 - row_max_3),
        tl.exp(cm_32 - row_max_3),
        tl.exp(cm_33 - row_max_3),
    )

    for _ in range(sinkhorn_repeat):
        row_sum_0 = cm_00 + cm_01 + cm_02 + cm_03 + sinkhorn_eps
        row_sum_1 = cm_10 + cm_11 + cm_12 + cm_13 + sinkhorn_eps
        row_sum_2 = cm_20 + cm_21 + cm_22 + cm_23 + sinkhorn_eps
        row_sum_3 = cm_30 + cm_31 + cm_32 + cm_33 + sinkhorn_eps
        cm_00, cm_01, cm_02, cm_03 = (
            cm_00 / row_sum_0,
            cm_01 / row_sum_0,
            cm_02 / row_sum_0,
            cm_03 / row_sum_0,
        )
        cm_10, cm_11, cm_12, cm_13 = (
            cm_10 / row_sum_1,
            cm_11 / row_sum_1,
            cm_12 / row_sum_1,
            cm_13 / row_sum_1,
        )
        cm_20, cm_21, cm_22, cm_23 = (
            cm_20 / row_sum_2,
            cm_21 / row_sum_2,
            cm_22 / row_sum_2,
            cm_23 / row_sum_2,
        )
        cm_30, cm_31, cm_32, cm_33 = (
            cm_30 / row_sum_3,
            cm_31 / row_sum_3,
            cm_32 / row_sum_3,
            cm_33 / row_sum_3,
        )

        col_sum_0 = cm_00 + cm_10 + cm_20 + cm_30 + sinkhorn_eps
        col_sum_1 = cm_01 + cm_11 + cm_21 + cm_31 + sinkhorn_eps
        col_sum_2 = cm_02 + cm_12 + cm_22 + cm_32 + sinkhorn_eps
        col_sum_3 = cm_03 + cm_13 + cm_23 + cm_33 + sinkhorn_eps
        cm_00, cm_10, cm_20, cm_30 = (
            cm_00 / col_sum_0,
            cm_10 / col_sum_0,
            cm_20 / col_sum_0,
            cm_30 / col_sum_0,
        )
        cm_01, cm_11, cm_21, cm_31 = (
            cm_01 / col_sum_1,
            cm_11 / col_sum_1,
            cm_21 / col_sum_1,
            cm_31 / col_sum_1,
        )
        cm_02, cm_12, cm_22, cm_32 = (
            cm_02 / col_sum_2,
            cm_12 / col_sum_2,
            cm_22 / col_sum_2,
            cm_32 / col_sum_2,
        )
        cm_03, cm_13, cm_23, cm_33 = (
            cm_03 / col_sum_3,
            cm_13 / col_sum_3,
            cm_23 / col_sum_3,
            cm_33 / col_sum_3,
        )

    output_comb_offset = token * 16
    tl.store(comb_ptr + output_comb_offset, cm_00)
    tl.store(comb_ptr + output_comb_offset + 1, cm_01)
    tl.store(comb_ptr + output_comb_offset + 2, cm_02)
    tl.store(comb_ptr + output_comb_offset + 3, cm_03)
    tl.store(comb_ptr + output_comb_offset + 4, cm_10)
    tl.store(comb_ptr + output_comb_offset + 5, cm_11)
    tl.store(comb_ptr + output_comb_offset + 6, cm_12)
    tl.store(comb_ptr + output_comb_offset + 7, cm_13)
    tl.store(comb_ptr + output_comb_offset + 8, cm_20)
    tl.store(comb_ptr + output_comb_offset + 9, cm_21)
    tl.store(comb_ptr + output_comb_offset + 10, cm_22)
    tl.store(comb_ptr + output_comb_offset + 11, cm_23)
    tl.store(comb_ptr + output_comb_offset + 12, cm_30)
    tl.store(comb_ptr + output_comb_offset + 13, cm_31)
    tl.store(comb_ptr + output_comb_offset + 14, cm_32)
    tl.store(comb_ptr + output_comb_offset + 15, cm_33)

    hidden_offsets = tl.arange(0, BLOCK_H)
    for h_start in range(0, hidden_size, BLOCK_H):
        h = h_start + hidden_offsets
        mask_h = h < hidden_size
        residual_offset = token * hc_hidden_size + h
        residual_0 = tl.load(residual_ptr + residual_offset, mask=mask_h, other=0.0)
        residual_1 = tl.load(
            residual_ptr + residual_offset + hidden_size, mask=mask_h, other=0.0
        )
        residual_2 = tl.load(
            residual_ptr + residual_offset + 2 * hidden_size,
            mask=mask_h,
            other=0.0,
        )
        residual_3 = tl.load(
            residual_ptr + residual_offset + 3 * hidden_size,
            mask=mask_h,
            other=0.0,
        )
        weighted = (
            (residual_0 * pre_0).to(tl.bfloat16).to(tl.float32)
            + (residual_1 * pre_1).to(tl.bfloat16).to(tl.float32)
            + (residual_2 * pre_2).to(tl.bfloat16).to(tl.float32)
            + (residual_3 * pre_3).to(tl.bfloat16).to(tl.float32)
        )
        tl.store(
            layer_input_ptr + token * hidden_size + h,
            weighted,
            mask=mask_h,
        )


def _check_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_repeat: int,
) -> None:
    if residual.ndim < 2 or residual.shape[-2] != 4:
        raise ValueError("residual must have shape (..., 4, hidden_size)")
    if residual.shape[-1] <= 0:
        raise ValueError("hidden_size must be positive")
    if residual.dtype != torch.bfloat16:
        raise TypeError("residual must have dtype torch.bfloat16")
    if fn.dtype != torch.float32:
        raise TypeError("fn must have dtype torch.float32")
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise TypeError("hc_scale and hc_base must have dtype torch.float32")
    hidden_size = residual.shape[-1]
    if fn.shape != (24, 4 * hidden_size):
        raise ValueError(f"fn must have shape (24, {4 * hidden_size})")
    if hc_scale.shape != (3,) or hc_base.shape != (24,):
        raise ValueError("hc_scale and hc_base must have shapes (3,) and (24,)")
    if not all(x.is_contiguous() for x in (residual, fn, hc_scale, hc_base)):
        raise ValueError("all inputs must be contiguous")
    if not all(x.device == residual.device for x in (fn, hc_scale, hc_base)):
        raise ValueError("all inputs must be on the same device")
    if not isinstance(sinkhorn_repeat, int) or sinkhorn_repeat <= 0:
        raise ValueError("sinkhorn_repeat must be a positive integer")


def mhc_pre_clamp_sinkhorn(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    clamp_min: float,
    clamp_max: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the XingChen4 mHC pre block without a PyTorch compute fallback.

    The operator supports the production configuration ``hc_mult == 4`` with
    BF16 residuals and FP32 parameters. RMS-normalized activations and weights
    are rounded to BF16 before the projection, combination logits are clamped,
    and every Sinkhorn iteration performs row then column normalization.
    """
    logger.debug("GEMS MHC PRE CLAMP SINKHORN")
    _check_inputs(residual, fn, hc_scale, hc_base, sinkhorn_repeat)
    if clamp_min >= clamp_max:
        raise ValueError("clamp_min must be less than clamp_max")

    outer_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.numel() // (4 * hidden_size)
    output_device = residual.device
    mixes = torch.empty((num_tokens, 24), dtype=torch.float32, device=output_device)
    post = torch.empty((num_tokens, 4), dtype=torch.bfloat16, device=output_device)
    comb = torch.empty((num_tokens, 4, 4), dtype=torch.bfloat16, device=output_device)
    layer_input = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=output_device
    )
    if num_tokens == 0:
        return (
            post.view(*outer_shape, 4, 1),
            comb.view(*outer_shape, 4, 4),
            layer_input.view(*outer_shape, hidden_size),
        )

    gemm_grid = lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_M"]),)
    epilogue_grid = (num_tokens,)
    with torch_device_fn.device(output_device):
        _rms_bf16_gemm_kernel[gemm_grid](
            residual,
            fn,
            mixes,
            num_tokens,
            hidden_size,
            rms_eps=rms_eps,
        )
        _mix_sinkhorn_reduce_kernel[epilogue_grid](
            residual,
            mixes,
            hc_scale,
            hc_base,
            post,
            comb,
            layer_input,
            num_tokens,
            hidden_size,
            4 * hidden_size,
            sinkhorn_eps=rms_eps,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    return (
        post.view(*outer_shape, 4, 1),
        comb.view(*outer_shape, 4, 4),
        layer_input.view(*outer_shape, hidden_size),
    )
