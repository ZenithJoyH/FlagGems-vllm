# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 Q/K normalization, partial RoPE and paged cache insertion.

The public function keeps vLLM's packed QKV and mutation contract. Its
portable Triton implementation deliberately uses separate normalization and
RoPE kernels: the intermediate is rounded to the input dtype before rotary,
and no program reads a peer element while another program writes it.
"""

from __future__ import annotations

import torch

from flaggems_vllm.ops.minimax_m3_index_cache import (
    index_cache_insert_kernel as _index_cache_insert_kernel,
)
from flaggems_vllm.ops.minimax_m3_kv_cache import (
    kv_cache_insert_kernel as _kv_cache_insert_kernel,
)
from flaggems_vllm.ops.minimax_m3_partial_rope import (
    partial_rope_64_kernel as _rope_64_kernel,
)
from flaggems_vllm.ops.minimax_m3_qk_norm import (
    gemma_qk_norm_kernel as _gemma_norm_kernel,
)
from flaggems_vllm.runtime import torch_device_fn

_HEAD_DIM = 128


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.device.type in ("cpu", "meta") or tensor.device != device:
        raise ValueError(f"{name} must be on the same accelerator as qkv")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must match qkv ({dtype})")


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    """Apply MiniMax-M3 Gemma QK norm, partial NeoX RoPE, and cache writes."""
    if qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("qkv must be float16 or bfloat16")
    if qkv.device.type in ("cpu", "meta") or not qkv.is_contiguous():
        raise ValueError("qkv must be a contiguous accelerator tensor")
    if qkv.ndim != 2:
        raise ValueError("qkv must be a packed [tokens, width] tensor")
    if num_heads <= 0 or num_kv_heads <= 0 or num_index_heads < 0:
        raise ValueError("head counts must be positive")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.device != qkv.device
    ):
        raise ValueError("positions must be an integer tensor on the qkv device")
    if positions.ndim != 1 or positions.numel() != qkv.shape[0]:
        raise ValueError("positions length must match qkv token count")
    if not positions.is_contiguous():
        raise ValueError("positions must be contiguous")
    if rotary_dim != 64:
        raise ValueError("MiniMax-M3 preprocessing requires rotary_dim=64")
    accepted_cache_dtype = {
        torch.float16: ("auto", "float16", "fp16"),
        torch.bfloat16: ("auto", "bfloat16", "bf16"),
    }[qkv.dtype]
    if kv_cache_dtype not in accepted_cache_dtype:
        raise NotImplementedError(
            "MiniMax-M3 preprocessing requires a cache dtype matching qkv; "
            "FP8 cache storage is not implemented"
        )

    dtype = qkv.dtype
    for name, tensor in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos_sin_cache", cos_sin_cache),
    ):
        _check_tensor(name, tensor, dtype, qkv.device)
    if q_norm_weight.numel() != _HEAD_DIM or k_norm_weight.numel() != _HEAD_DIM:
        raise ValueError("q/k norm weights must each have 128 elements")
    if not q_norm_weight.is_contiguous() or not k_norm_weight.is_contiguous():
        raise ValueError("q/k norm weights must be contiguous")
    if (
        cos_sin_cache.ndim != 2
        or not cos_sin_cache.is_contiguous()
        or cos_sin_cache.shape[1] != rotary_dim
    ):
        raise ValueError("cos_sin_cache must be contiguous [max_pos, rotary_dim]")

    has_index = num_index_heads > 0
    expected_heads = num_heads + 2 * num_kv_heads
    if has_index:
        expected_heads += num_index_heads + 1
    if qkv.shape[1] != expected_heads * _HEAD_DIM:
        raise ValueError("qkv packed width does not match the configured head counts")
    q_size = num_heads * _HEAD_DIM
    kv_size = num_kv_heads * _HEAD_DIM
    index_q_size = num_index_heads * _HEAD_DIM
    for name, output, width in (
        ("q_out", q_out, q_size),
        ("index_q_out", index_q_out, index_q_size),
    ):
        if output is None:
            continue
        _check_tensor(name, output, dtype, qkv.device)
        if (
            output.shape != (qkv.shape[0], width)
            or not output.is_contiguous()
            or output.untyped_storage().data_ptr()
            == qkv.untyped_storage().data_ptr()
        ):
            raise ValueError(f"{name} must be a separate contiguous output")
    if (
        q_out is not None
        and index_q_out is not None
        and q_out.untyped_storage().data_ptr()
        == index_q_out.untyped_storage().data_ptr()
    ):
        raise ValueError("q_out and index_q_out must not alias")
    if qkv.shape[0] == 0:
        return

    if has_index:
        if index_q_norm_weight is None or index_k_norm_weight is None:
            raise ValueError("index branch requires both index norm weights")
        _check_tensor("index_q_norm_weight", index_q_norm_weight, dtype, qkv.device)
        _check_tensor("index_k_norm_weight", index_k_norm_weight, dtype, qkv.device)
        if (
            index_q_norm_weight.numel() != _HEAD_DIM
            or index_k_norm_weight.numel() != _HEAD_DIM
        ):
            raise ValueError("index norm weights must each have 128 elements")
        if (
            not index_q_norm_weight.is_contiguous()
            or not index_k_norm_weight.is_contiguous()
        ):
            raise ValueError("index norm weights must be contiguous")

    if kv_cache is not None:
        if not has_index:
            raise ValueError("KV insertion requires the sparse index branch")
        if slot_mapping is None or index_cache is None:
            raise ValueError("KV insertion requires slot_mapping and index_cache")
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        if block_size <= 0 or kv_cache.ndim != 5 or index_cache.ndim != 3:
            raise ValueError("invalid paged-cache shape or block_size")
        if (
            kv_cache.shape[1:] != (2, block_size, num_kv_heads, _HEAD_DIM)
            or index_cache.shape[1:] != (block_size, _HEAD_DIM)
        ):
            raise ValueError("paged-cache shape does not match head counts")
        _check_tensor("kv_cache", kv_cache, dtype, qkv.device)
        _check_tensor("index_cache", index_cache, dtype, qkv.device)
        for name, mapping in (
            ("slot_mapping", slot_mapping),
            ("index_slot_mapping", index_slot_mapping),
        ):
            if mapping.device != qkv.device or mapping.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must be an integer tensor on the qkv device")
            if mapping.ndim != 1 or mapping.numel() != qkv.shape[0]:
                raise ValueError(f"{name} length must match qkv token count")
            if not mapping.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    launches = [
        (0, q_norm_weight, num_heads, q_out, 0),
        (q_size, k_norm_weight, num_kv_heads, None, q_size),
    ]
    if has_index:
        index_q_offset = q_size + 2 * kv_size
        index_k_offset = index_q_offset + index_q_size
        launches.extend(
            [
                (
                    index_q_offset,
                    index_q_norm_weight,
                    num_index_heads,
                    index_q_out,
                    0,
                ),
                (index_k_offset, index_k_norm_weight, 1, None, index_k_offset),
            ]
        )

    with torch_device_fn.device(qkv.device):
        for x_offset, weight, heads, external_out, in_place_offset in launches:
            out = qkv if external_out is None else external_out
            out_offset = in_place_offset if external_out is None else 0
            _gemma_norm_kernel[(heads, qkv.shape[0])](
                qkv,
                weight,
                out,
                qkv.stride(0),
                qkv.stride(1),
                out.stride(0),
                out.stride(1),
                x_offset,
                out_offset,
                eps,
                num_warps=4,
            )
            _rope_64_kernel[(heads, qkv.shape[0])](
                out,
                cos_sin_cache,
                positions,
                out.stride(0),
                out.stride(1),
                out_offset,
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                num_warps=1,
            )

        if kv_cache is None:
            return

        _kv_cache_insert_kernel[(num_kv_heads, qkv.shape[0])](
            qkv,
            slot_mapping,
            kv_cache,
            qkv.stride(0),
            qkv.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            kv_cache.stride(4),
            q_size,
            q_size + kv_size,
            num_kv_heads,
            block_size,
            num_warps=4,
        )
        _index_cache_insert_kernel[(qkv.shape[0],)](
            qkv,
            index_slot_mapping,
            index_cache,
            qkv.stride(0),
            qkv.stride(1),
            index_cache.stride(0),
            index_cache.stride(1),
            index_cache.stride(2),
            index_k_offset,
            block_size,
            num_warps=4,
        )
