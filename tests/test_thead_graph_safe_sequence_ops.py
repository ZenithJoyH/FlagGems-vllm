# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for graph-safe sequence operators used by day-0 serving."""

import math

import pytest
import torch

from flaggems_vllm import (
    bf16_paged_mqa_logits_graph_safe,
    causal_conv1d_update,
    cp_gather_indexer_k_bf16_cache,
    fused_recurrent_kda,
    fused_safe_kda_gate,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    persist_prefill_tail,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="accelerator device required"
)


def _device():
    return torch.device("cuda")


def _hadamard_128(x):
    y = x.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        groups = 128 // (2 * stride)
        shaped = y.reshape(*y.shape[:-1], groups, 2, stride)
        first = shaped[..., 0, :]
        second = shaped[..., 1, :]
        y = torch.stack((first + second, first - second), dim=-2).reshape_as(y)
    return y / math.sqrt(128)


@torch.inference_mode()
def test_causal_conv1d_update_matches_reference():
    torch.manual_seed(1)
    device = _device()
    batch, states, dim = 2, 4, 128
    x = torch.randn(batch, dim, device=device, dtype=torch.bfloat16)
    state = torch.randn(states, dim, 3, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, 4, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)

    expected_state = state.clone()
    selected = expected_state[state_indices.long()].clone().float()
    expected = (selected * weight[:, :3].float().unsqueeze(0)).sum(-1)
    expected += x.float() * weight[:, 3].float()
    expected += bias.float()
    expected = torch.nn.functional.silu(expected).to(x.dtype)
    expected_state[state_indices.long(), :, :-1] = selected[:, :, 1:].to(state.dtype)
    expected_state[state_indices.long(), :, -1] = x

    actual = causal_conv1d_update(
        x,
        state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=state_indices,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@torch.inference_mode()
def test_bf16_indexer_gather_matches_reference():
    torch.manual_seed(2)
    device = _device()
    page_size, dim = 4, 128
    cache = torch.randn(6, page_size, dim, device=device, dtype=torch.bfloat16)
    block_table = torch.tensor([[2, 4], [1, 5]], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, 6, 9], device=device, dtype=torch.int32)
    output = torch.empty(9, dim, device=device, dtype=torch.bfloat16)
    scales = torch.ones(9, 1, device=device, dtype=torch.float32)

    expected = torch.empty_like(output)
    for request, (start, end) in enumerate(((0, 6), (6, 9))):
        for packed in range(start, end):
            offset = packed - start
            block = int(block_table[request, offset // page_size])
            expected[packed] = cache[block, offset % page_size]

    cp_gather_indexer_k_bf16_cache(
        cache, output, scales, block_table, cu_seqlen
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_graph_safe_bf16_mqa_masks_unallocated_block():
    torch.manual_seed(8)
    device = _device()
    block_size, heads, dim = 16, 32, 128
    query = torch.randn(1, 1, heads, dim, device=device, dtype=torch.bfloat16)
    cache = torch.randn(4, block_size, 1, dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(1, heads, device=device, dtype=torch.float32)
    context_lens = torch.tensor([[17]], device=device, dtype=torch.int32)
    block_table = torch.tensor([[3, -1]], device=device, dtype=torch.int32)

    actual = bf16_paged_mqa_logits_graph_safe(
        query,
        cache,
        weights,
        context_lens,
        block_table,
        None,
        max_context_len=32,
    )
    keys = cache[3, :, 0].float()
    scores = torch.relu(keys @ query[0, 0].float().T)
    expected = (scores * weights[0]).sum(-1)
    torch.testing.assert_close(actual[0, :16], expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[0, 16:], torch.zeros_like(actual[0, 16:]))


@torch.inference_mode()
def test_safe_kda_gate_matches_reference():
    torch.manual_seed(3)
    device = _device()
    tokens, heads, dim = 5, 4, 128
    gate = torch.randn(tokens, heads * dim, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    bias = torch.randn(heads, dim, device=device, dtype=torch.float32)

    actual = fused_safe_kda_gate(gate, a_log, dim, bias, lower_bound=-5.0)
    expected = -5.0 * torch.sigmoid(
        torch.exp(a_log)[None, :, None] * (gate.float().view(tokens, heads, dim) + bias)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@torch.inference_mode()
def test_recurrent_kda_decode_matches_reference():
    torch.manual_seed(4)
    device = _device()
    tokens, heads, dim, states = 2, 4, 128, 3
    q = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate = -torch.rand(1, tokens, heads, dim, device=device, dtype=torch.float32)
    beta = torch.rand(1, tokens, heads, device=device, dtype=torch.float32)
    initial_state = torch.randn(
        states, heads, dim, dim, device=device, dtype=torch.float32
    ) * 0.01
    expected_state = initial_state.clone()
    state_indices = torch.tensor([0, 2], device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    scale = dim**-0.5
    expected = torch.empty_like(v)

    for token in range(tokens):
        state_id = int(state_indices[token])
        for head in range(heads):
            q_vec = torch.nn.functional.normalize(q[0, token, head].float(), dim=0)
            k_vec = torch.nn.functional.normalize(k[0, token, head].float(), dim=0)
            state = expected_state[state_id, head]
            state = state * torch.exp(gate[0, token, head])[:, None]
            residual = v[0, token, head].float() - k_vec @ state
            state = state + beta[0, token, head] * k_vec[:, None] * residual[None, :]
            expected[0, token, head] = ((q_vec * scale) @ state).to(v.dtype)
            expected_state[state_id, head] = state

    actual, final_state = fused_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
    )
    assert final_state.data_ptr() == initial_state.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(final_state, expected_state, rtol=5e-3, atol=5e-3)


@torch.inference_mode()
def test_kpool_bf16_compress_matches_reference():
    torch.manual_seed(5)
    device = _device()
    pools, pool_size, dim = 2, 4, 128
    cache = torch.zeros(3, 16, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(pools, pool_size, dim, device=device, dtype=torch.bfloat16)
    score = torch.randn_like(key)
    ape = torch.randn(pool_size, dim, device=device, dtype=torch.float32)
    locations = torch.tensor([0, 17], device=device, dtype=torch.int64)

    probabilities = torch.softmax(score.float() + ape, dim=1)
    pooled = (key.float() * probabilities).sum(1).to(torch.bfloat16)
    expected = _hadamard_128(pooled).to(torch.bfloat16)

    kpool_compress_and_write_cache(
        cache, key, score, ape, locations, pool_size=pool_size
    )
    torch.testing.assert_close(cache[0, 0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cache[1, 1], expected[1], rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_prefill_tail_persists_only_incomplete_pool():
    torch.manual_seed(6)
    device = _device()
    dim, pool_size = 128, 4
    key = torch.randn(7, dim, device=device, dtype=torch.bfloat16)
    gate = torch.randn_like(key)
    slots = torch.tensor([0, 1, 2, 4, 5, 6, 7], device=device, dtype=torch.int64)
    tail = torch.full(
        (2, 2, pool_size, dim), -9, device=device, dtype=torch.bfloat16
    )

    persist_prefill_tail(key, gate, slots, tail, pool_size=pool_size)
    torch.testing.assert_close(tail[0, 0, :3], key[:3], rtol=0, atol=0)
    torch.testing.assert_close(tail[0, 1, :3], gate[:3], rtol=0, atol=0)
    torch.testing.assert_close(
        tail[0, :, 3], torch.full_like(tail[0, :, 3], -9), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tail[1], torch.full_like(tail[1], -9), rtol=0, atol=0
    )


@torch.inference_mode()
def test_kpool_decode_stashes_incomplete_token():
    torch.manual_seed(7)
    device = _device()
    dim, pool_size = 128, 4
    cache = torch.zeros(1, 16, dim, device=device, dtype=torch.bfloat16)
    tail = torch.zeros(1, 2, pool_size, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 1, dim, device=device, dtype=torch.bfloat16)
    score = torch.randn_like(key)
    ape = torch.randn(pool_size, dim, device=device, dtype=torch.float32)
    mapping = torch.zeros(1, 1, device=device, dtype=torch.int32)
    positions = torch.zeros_like(mapping)

    kpool_decode_update_and_maybe_write_cache_batched(
        cache,
        tail,
        mapping,
        key,
        score,
        ape,
        mapping,
        positions,
        pool_size=pool_size,
    )
    torch.testing.assert_close(tail[0, 0, 0], key[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(tail[0, 1, 0], score[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(cache, torch.zeros_like(cache), rtol=0, atol=0)
