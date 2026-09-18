# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for general variable-length sequence capabilities."""

import pytest
import torch

from flaggems_vllm import (
    append_tail_to_topk,
    causal_conv1d_fn,
    chunk_kda_with_safe_gate,
    expand_pools_to_tokens,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="accelerator device required"
)


def _device():
    return torch.device("cuda")


@torch.inference_mode()
def test_causal_conv1d_varlen_matches_reference():
    torch.manual_seed(9)
    device = _device()
    dim = 128
    lengths = (3, 2)
    boundaries = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    state_indices = torch.tensor([2, 0], device=device, dtype=torch.int32)
    has_initial = torch.tensor([True, False], device=device, dtype=torch.bool)
    x = torch.randn(dim, sum(lengths), device=device, dtype=torch.bfloat16)
    state = torch.randn(4, dim, 3, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, 4, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    expected_state = state.clone()
    expected = torch.empty_like(x)
    cursor = 0
    for request, length in enumerate(lengths):
        state_id = int(state_indices[request])
        history = (
            expected_state[state_id].clone().float()
            if bool(has_initial[request])
            else torch.zeros_like(expected_state[state_id], dtype=torch.float32)
        )
        for token in range(length):
            current = x[:, cursor + token].float()
            value = (history * weight[:, :3].float()).sum(-1)
            value += current * weight[:, 3].float() + bias.float()
            expected[:, cursor + token] = torch.nn.functional.silu(value).to(x.dtype)
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
        expected_state[state_id] = history.to(state.dtype)
        cursor += length

    actual = causal_conv1d_fn(
        x,
        weight,
        bias,
        state,
        boundaries,
        cache_indices=state_indices,
        has_initial_state=has_initial,
        activation="silu",
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@torch.inference_mode()
def test_chunk_kda_varlen_matches_reference():
    torch.manual_seed(10)
    device = _device()
    lengths = (2, 1)
    tokens, heads, dim = sum(lengths), 4, 128
    q = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.rand(1, tokens, heads, device=device, dtype=torch.float32)
    a_log = torch.randn(1, 1, heads, 1, device=device, dtype=torch.float32)
    bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    boundaries = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    initial_state = torch.randn(
        len(lengths), heads, dim, dim, device=device, dtype=torch.float32
    ) * 0.01
    original_state = initial_state.clone()
    expected_state = initial_state.clone()
    expected = torch.empty_like(v)
    scale = dim**-0.5
    cursor = 0
    for sequence, length in enumerate(lengths):
        for token in range(cursor, cursor + length):
            for head in range(heads):
                q_vec = torch.nn.functional.normalize(q[0, token, head].float(), dim=0)
                k_vec = torch.nn.functional.normalize(k[0, token, head].float(), dim=0)
                gate = -5.0 * torch.sigmoid(
                    torch.exp(a_log.flatten()[head])
                    * (raw_g[0, token, head].float() + bias[head])
                )
                state_head = expected_state[sequence, head] * torch.exp(gate)[:, None]
                residual = v[0, token, head].float() - k_vec @ state_head
                state_head += beta[0, token, head] * k_vec[:, None] * residual[None, :]
                expected[0, token, head] = ((q_vec * scale) @ state_head).to(v.dtype)
                expected_state[sequence, head] = state_head
        cursor += length

    actual, final_state = chunk_kda_with_safe_gate(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=boundaries,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(final_state, expected_state, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(initial_state, original_state, rtol=0, atol=0)


@torch.inference_mode()
def test_indexer_pool_mapping_matches_reference():
    device = _device()
    group_ids = torch.tensor([[2, 0], [1, 3]], device=device, dtype=torch.int32)
    group_valid = torch.tensor([[True, False], [True, True]], device=device)
    offsets = torch.tensor([10, 20], device=device, dtype=torch.int32)
    actual = expand_pools_to_tokens(
        group_ids,
        group_valid,
        topk=8,
        pool_size=4,
        topk_offsets=offsets,
    )
    expected = torch.tensor(
        [[18, 19, 20, 21, -1, -1, -1, -1], [24, 25, 26, 27, 32, 33, 34, 35]],
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    history = torch.tensor([[3, 4], [7, 8]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([7, 8], device=device, dtype=torch.int32)
    pool_lens = torch.tensor([1, 2], device=device, dtype=torch.int32)
    actual_tail = append_tail_to_topk(
        history,
        seq_lens,
        pool_lens,
        pool_size=4,
        topk_offsets=offsets,
    )
    expected_tail = torch.tensor(
        [[3, 4, 14, 15, 16], [7, 8, -1, -1, -1]],
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)
