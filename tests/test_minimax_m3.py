# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical and mutation-contract checks for MiniMax-M3 preprocessing."""

import pytest
import torch

import flaggems_vllm

HEAD = 128
ROTARY = 64


def _device():
    return flaggems_vllm.runtime.device.name


def _cache(device, dtype):
    positions = torch.arange(32, device=device, dtype=torch.float32)
    inv = 1.0 / (
        5_000_000.0
        ** (torch.arange(0, ROTARY, 2, device=device).float() / ROTARY)
    )
    angles = positions[:, None] * inv[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)


def _reference(x, weight, positions, cache):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
    y = (y * (1.0 + weight.float())).to(x.dtype).float()
    selected = cache[positions.long()].float()
    cos = selected[:, None, : ROTARY // 2]
    sin = selected[:, None, ROTARY // 2 :]
    first, second = y[..., :32], y[..., 32:64]
    result = y.clone()
    result[..., :32] = first * cos - second * sin
    result[..., 32:64] = second * cos + first * sin
    return result.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_minimax_m3_preprocess(dtype, sparse):
    device = _device()
    tokens, q_heads, kv_heads, index_heads = 5, 4, 2, 2 if sparse else 0
    q_size, kv_size = q_heads * HEAD, kv_heads * HEAD
    index_size = index_heads * HEAD
    width = q_size + 2 * kv_size + (index_size + HEAD if sparse else 0)
    torch.manual_seed(1)
    qkv = torch.randn(tokens, width, device=device, dtype=dtype)
    original = qkv.clone()
    weights = [torch.randn(HEAD, device=device, dtype=dtype) * 0.1 for _ in range(4)]
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    cache = _cache(device, dtype)
    q_out = torch.empty(tokens, q_size, device=device, dtype=dtype) if sparse else None
    index_out = (
        torch.empty(tokens, index_size, device=device, dtype=dtype) if sparse else None
    )
    slots = torch.tensor([2, -1, 130, 5, 7], device=device, dtype=torch.int64)
    index_slots = torch.tensor([9, -1, 133, 3, 4], device=device, dtype=torch.int64)
    kv_cache = (
        torch.zeros(2, 2, 128, kv_heads, HEAD, device=device, dtype=dtype)
        if sparse
        else None
    )
    index_cache = (
        torch.zeros(2, 128, HEAD, device=device, dtype=dtype) if sparse else None
    )

    flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        weights[0],
        weights[1],
        cache,
        positions,
        q_heads,
        kv_heads,
        ROTARY,
        1e-6,
        index_q_norm_weight=weights[2] if sparse else None,
        index_k_norm_weight=weights[3] if sparse else None,
        num_index_heads=index_heads,
        slot_mapping=slots if sparse else None,
        index_slot_mapping=index_slots if sparse else None,
        kv_cache=kv_cache,
        index_cache=index_cache,
        block_size=128 if sparse else 0,
        q_out=q_out,
        index_q_out=index_out,
        kv_cache_dtype="auto",
    )

    q_ref = _reference(
        original[:, :q_size].view(tokens, q_heads, HEAD),
        weights[0],
        positions,
        cache,
    ).reshape(tokens, q_size)
    k_ref = _reference(
        original[:, q_size : q_size + kv_size].view(tokens, kv_heads, HEAD),
        weights[1],
        positions,
        cache,
    ).reshape(tokens, kv_size)
    q_actual = q_out if sparse else qkv[:, :q_size]
    torch.testing.assert_close(q_actual, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        qkv[:, q_size : q_size + kv_size], k_ref, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        qkv[:, q_size + kv_size : q_size + 2 * kv_size],
        original[:, q_size + kv_size : q_size + 2 * kv_size],
        rtol=0,
        atol=0,
    )
    if sparse:
        iq_start = q_size + 2 * kv_size
        iq_ref = _reference(
            original[:, iq_start : iq_start + index_size].view(
                tokens, index_heads, HEAD
            ),
            weights[2],
            positions,
            cache,
        ).reshape(tokens, index_size)
        ik_ref = _reference(
            original[:, -HEAD:].view(tokens, 1, HEAD),
            weights[3],
            positions,
            cache,
        ).reshape(tokens, HEAD)
        torch.testing.assert_close(index_out, iq_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(qkv[:, -HEAD:], ik_ref, rtol=1e-2, atol=1e-2)
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, 128)
                torch.testing.assert_close(
                    kv_cache[block, 0, offset],
                    k_ref[token].view(kv_heads, HEAD),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    kv_cache[block, 1, offset],
                    original[token, q_size + kv_size : q_size + 2 * kv_size].view(
                        kv_heads, HEAD
                    ),
                    rtol=0,
                    atol=0,
                )
        for token, slot in enumerate(index_slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, 128)
                torch.testing.assert_close(
                    index_cache[block, offset], ik_ref[token], rtol=0, atol=0
                )


def test_minimax_m3_rejects_fp8_cache():
    device = _device()
    qkv = torch.empty(1, 4 * HEAD, device=device, dtype=torch.bfloat16)
    weight = torch.empty(HEAD, device=device, dtype=torch.bfloat16)
    cache = torch.empty(1, ROTARY, device=device, dtype=torch.bfloat16)
    positions = torch.zeros(1, device=device, dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="FP8"):
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, weight, weight, cache, positions, 2, 1, ROTARY, 1e-6,
            kv_cache_dtype="fp8",
        )


def test_invalid_cache_contract_does_not_mutate_input():
    device = _device()
    qkv = torch.randn(1, 5 * HEAD, device=device, dtype=torch.bfloat16)
    original = qkv.clone()
    weight = torch.zeros(HEAD, device=device, dtype=torch.bfloat16)
    cache = _cache(device, torch.bfloat16)
    positions = torch.zeros(1, device=device, dtype=torch.int64)
    kv_cache = torch.empty(1, 2, 16, 1, HEAD, device=device, dtype=qkv.dtype)
    with pytest.raises(ValueError, match="slot_mapping"):
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            weight,
            weight,
            cache,
            positions,
            1,
            1,
            ROTARY,
            1e-6,
            index_q_norm_weight=weight,
            index_k_norm_weight=weight,
            num_index_heads=1,
            kv_cache=kv_cache,
            block_size=16,
        )
    torch.testing.assert_close(qkv, original, rtol=0, atol=0)


def test_minimax_m3_graph_replay():
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA-compatible graph capture is unavailable")
    device = _device()
    qkv = torch.randn(4, 4 * HEAD, device=device, dtype=torch.bfloat16)
    weight = torch.randn(HEAD, device=device, dtype=torch.bfloat16) * 0.1
    cache = _cache(device, torch.bfloat16)
    positions = torch.arange(4, device=device, dtype=torch.int64)
    original = qkv.clone()

    def invoke():
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, weight, weight, cache, positions, 2, 1, ROTARY, 1e-6
        )

    invoke()
    torch.cuda.synchronize()
    qkv.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    replacement = torch.randn_like(qkv)
    qkv.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        replacement[:, : 2 * HEAD].view(4, 2, HEAD), weight, positions, cache
    ).reshape(4, 2 * HEAD)
    torch.testing.assert_close(qkv[:, : 2 * HEAD], expected, rtol=1e-2, atol=1e-2)
