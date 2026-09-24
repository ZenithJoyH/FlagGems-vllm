# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Preliminary dense MiniMax-M3 preprocessing benchmark (device required)."""

import pytest
import torch
import triton.testing

import flaggems_vllm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-compatible GPU needed")
@pytest.mark.parametrize("tokens", [1, 32, 128])
def test_minimax_m3_dense_benchmark(tokens):
    device = flaggems_vllm.runtime.device.name
    heads, kv_heads, dim, rotary = 4, 2, 128, 64
    qkv = torch.randn(
        tokens, (heads + 2 * kv_heads) * dim, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    cos_sin = torch.randn(256, rotary, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)

    def reference():
        chunks = qkv.split((heads * dim, kv_heads * dim, kv_heads * dim), -1)
        for x, count in ((chunks[0], heads), (chunks[1], kv_heads)):
            shaped = x.view(tokens, count, dim).float()
            normalized = shaped * torch.rsqrt(
                shaped.square().mean(-1, keepdim=True) + 1e-6
            )
            normalized = (normalized * (1.0 + weight.float())).to(qkv.dtype)
            cos, sin = cos_sin[positions].float().chunk(2, -1)
            first = normalized.float()[..., :32].clone()
            second = normalized.float()[..., 32:64].clone()
            normalized[..., :32] = first * cos[:, None, :] - second * sin[:, None, :]
            normalized[..., 32:64] = second * cos[:, None, :] + first * sin[:, None, :]
            x.copy_(normalized.reshape_as(x))

    def gems():
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, weight, weight, cos_sin, positions, heads, kv_heads, rotary, 1e-6
        )

    baseline_ms = triton.testing.do_bench(reference)
    gems_ms = triton.testing.do_bench(gems)
    speedup = baseline_ms / gems_ms
    print(
        f"tokens={tokens} baseline_ms={baseline_ms:.4f} "
        f"gems_ms={gems_ms:.4f} speedup={speedup:.3f}"
    )
