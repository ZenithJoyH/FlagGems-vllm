# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE

moe = importlib.import_module("flaggems_vllm.ops.fused_moe")
pytestmark = [
    pytest.mark.fused_moe,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def literal_oai(x):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(0, 384), (1, 127), (7, 384), (32, 768), (64, 6144)])
def test_quant_and_oai_quant(dtype, shape, monkeypatch):
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
    torch.manual_seed(43)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 8
    q, s = moe._int8_quantize_per_token_triton(x)
    refq, refs = literal_quant(x)
    assert torch.equal(q, refq)
    torch.testing.assert_close(s, refs, rtol=0, atol=0)
    packed = torch.randn((shape[0], 2 * shape[1]), device="cuda", dtype=dtype) * 8
    activated = torch.empty(shape, device="cuda", dtype=dtype)
    q, s = moe._swigluoai_quantize_per_token_triton(packed, activated, 1.702, 1.0, 7.0)
    expected = torch.empty_like(activated)
    if shape[0]:
        moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            expected,
            packed,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
    torch.testing.assert_close(activated, expected, rtol=0, atol=0)
    refq, refs = literal_quant(expected)
    assert torch.equal(q, refq)
    torch.testing.assert_close(s, refs, rtol=0, atol=0)


def test_quant_nan_halfway_graph(monkeypatch):
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
    x = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [127.0, 0.5, 1.5, 2.5], [float("nan"), 1.0, -2.0, 3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, s = moe._int8_quantize_per_token_triton(x)
    rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0, equal_nan=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q, s = moe._int8_quantize_per_token_triton(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0)


def test_oai_parameter_validation():
    x = torch.randn((2, 768), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 384), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        moe.apply_moe_activation(moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE, out, x)
    with pytest.raises(ValueError):
        moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=-1.0,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,i", [(1, 384), (7, 768), (32, 1536)])
@pytest.mark.parametrize("mode", ["off", "quant_oai"])
def test_w8a8_oai_experts(dtype, m, i, mode, monkeypatch):
    monkeypatch.setattr(moe, "_W8A8_QUANT_FUSION_MODE", mode)
    monkeypatch.setattr(moe, "_int8_quant_dispatch_policy", lambda: False)
    torch.manual_seed(47)
    e, h, k = 4, 128, 2
    x = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.randint(-64, 64, (e, 2 * i, h), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-64, 64, (e, h, i), device="cuda", dtype=torch.int8)
    s1 = torch.rand((e, 2 * i), device="cuda") * 0.001 + 0.001
    s2 = torch.rand((e, h), device="cuda") * 0.001 + 0.001
    ids = (torch.arange(m * k, device="cuda").view(m, k) % e).int()
    weights = torch.rand((m, k), device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    actual = moe.fused_experts_impl(
        x,
        w1,
        w2,
        weights,
        ids,
        activation="swigluoai_uninterleave",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=s1,
        w2_scale=s2,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    if m == 1 and mode == "off":
        kwargs = dict(
            activation="swigluoai_uninterleave",
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=s1,
            w2_scale=s2,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        out = moe.outplace_fused_experts(x, w1, w2, weights, ids, **kwargs)
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
        in_place = x.clone()
        assert (
            moe.inplace_fused_experts(in_place, w1, w2, weights, ids, **kwargs) is None
        )
        torch.testing.assert_close(in_place, actual, rtol=0, atol=0)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qx, sx = literal_quant(x)
        contributions = []
        for t in range(m):
            row = []
            for rank in range(k):
                expert = int(ids[t, rank])
                up = (
                    (qx[t : t + 1].float() @ w1[expert].float().t())
                    * sx[t]
                    * s1[expert]
                )
                activated = literal_oai(up.to(dtype))
                qa, sa = literal_quant(activated)
                down = (qa.float() @ w2[expert].float().t()) * sa * s2[expert]
                row.append((down * weights[t, rank]).to(dtype))
            contributions.append(torch.stack(row).float().sum(0).to(dtype))
        expected = torch.cat(contributions)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.0005 if dtype == torch.float16 else 0.004,
        rtol=0.003 if dtype == torch.float16 else 0.016,
    )


def test_quant_full_and_mixed_dispatch():
    import flag_gems

    x = torch.tensor(
        [[52.0, 26.0, 13.0, 0.0], [float("nan"), 1.0, 2.0, 3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    with flag_gems.use_gems(include=["abs", "amax", "clamp", "true_divide", "round"]):
        assert moe._int8_quant_dispatch_policy() is True
        q, s = moe._int8_quantize_per_token_triton(x)
        rq, rs = literal_quant(x)
    assert torch.equal(q, rq)
    torch.testing.assert_close(s, rs, rtol=0, atol=0, equal_nan=True)
    with flag_gems.use_gems(include=["abs"]):
        assert moe._int8_quant_dispatch_policy() is None
    assert moe._int8_quant_dispatch_policy() is False


@pytest.mark.parametrize(
    "shape",
    (
        [
            (0, 4, 128, 384, 2),
            (1, 4, 128, 384, 2),
            (64, 128, 6144, 384, 4),
            (1024, 128, 6144, 384, 4),
        ]
        if QUICK_MODE
        else [
            (0, 4, 128, 384, 2),
            (1, 4, 128, 384, 2),
            (8, 4, 128, 384, 2),
            (32, 4, 128, 384, 2),
            (32, 128, 6144, 384, 4),
            (64, 128, 6144, 384, 4),
            (65, 128, 6144, 384, 4),
            (1024, 128, 6144, 384, 4),
            (8192, 128, 6144, 384, 4),
        ]
    ),
)
@pytest.mark.parametrize("scale_rank", [2] if QUICK_MODE else [2, 3])
@pytest.mark.parametrize(
    "route_dtype", [torch.float32] if QUICK_MODE else [torch.float32, torch.bfloat16]
)
def test_w8a8_experts_graph(shape, scale_rank, route_dtype):
    """Exercise the complete API on every backend with the same integer oracle.

    One nonzero per weight row permits an independent CPU reference for large
    shapes without copying the dense expert weights or using device GEMM.
    """
    tokens, experts, hidden_size, intermediate_size, top_k = shape
    device = flaggems_vllm.device
    generator = torch.Generator().manual_seed(81)
    hidden_cpu = (
        torch.randn(tokens, hidden_size, generator=generator).abs() * 0.25 + 0.125
    ).to(torch.bfloat16)
    routes_cpu = torch.randint(
        0, experts, (tokens, top_k), generator=generator, dtype=torch.int32
    )
    router_cpu = torch.rand(tokens, top_k, generator=generator)
    router_cpu /= router_cpu.sum(-1, keepdim=True)
    router_cpu = router_cpu.to(route_dtype)
    e = torch.arange(experts)[:, None]
    channels1 = torch.arange(2 * intermediate_size)[None, :]
    channels2 = torch.arange(hidden_size)[None, :]
    index1 = (channels1 * 7 + e * 13) % hidden_size
    index2 = (channels2 * 3 + e * 11) % intermediate_size
    coeff1 = ((channels1 + e) % 3 + 1).to(torch.int8)
    coeff2 = ((channels2 + e * 2) % 3 + 1).to(torch.int8)
    scale1 = (1 + ((channels1 + e) % 4)).float() / 64
    scale2 = (1 + ((channels2 + e) % 4)).float() / 32
    w1 = torch.zeros(
        (experts, 2 * intermediate_size, hidden_size), dtype=torch.int8, device=device
    )
    w2 = torch.zeros(
        (experts, hidden_size, intermediate_size), dtype=torch.int8, device=device
    )
    w1[e.to(device), channels1.to(device), index1.to(device)] = coeff1.to(device)
    w2[e.to(device), channels2.to(device), index2.to(device)] = coeff2.to(device)
    w1_scale, w2_scale = scale1.to(device), scale2.to(device)
    if scale_rank == 3:
        w1_scale, w2_scale = w1_scale.unsqueeze(-1), w2_scale.unsqueeze(-1)
    hidden = hidden_cpu.to(device)
    routes, router = routes_cpu.to(device), router_cpu.to(device)

    def run():
        return flaggems_vllm.fused_experts_impl(
            hidden,
            w1,
            w2,
            router,
            routes,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    eager = run().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert out.shape == hidden.shape and out.dtype == hidden.dtype
    torch.testing.assert_close(out, eager, rtol=0, atol=0)

    for begin in range(0, tokens, 128):
        end = min(begin + 128, tokens)
        choice = routes_cpu[begin:end].long()
        q1, a1_scale = literal_quant(hidden_cpu[begin:end])
        dots1 = q1[:, None, :].expand(-1, top_k, -1).gather(2, index1[choice])
        dots1 = (dots1.int() * coeff1[choice].int()).float()
        up = (dots1 * a1_scale[:, None, :] * scale1[choice]).to(torch.bfloat16)
        gate, value = up.float().chunk(2, dim=-1)
        activated = (gate * torch.sigmoid(gate) * value).to(torch.bfloat16)
        q2, a2_scale = literal_quant(activated)
        dots2 = (q2.gather(2, index2[choice]).int() * coeff2[choice].int()).float()
        down = dots2 * a2_scale * scale2[choice]
        down = (down * router_cpu[begin:end, :, None].float()).to(torch.bfloat16)
        expected = down.float().sum(1).to(torch.bfloat16)
        # Existing full-pipeline CPU reference tolerance; no vendor branching.
        torch.testing.assert_close(out[begin:end].cpu(), expected, rtol=0.03, atol=1e-7)
