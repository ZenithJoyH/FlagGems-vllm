# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from triton import language as tl

import flaggems_vllm

moe = importlib.import_module("flaggems_vllm.ops.fused_moe")
pytestmark = pytest.mark.fused_moe
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def literal_oai(x):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@requires_cuda
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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
@pytest.mark.parametrize(
    "shape",
    [
        (0, 4, 128, 384, 2),
        (1, 4, 128, 384, 2),
        (8, 4, 128, 384, 2),
        (32, 4, 128, 384, 2),
        (32, 128, 6144, 384, 4),
        (64, 128, 6144, 384, 4),
        (65, 128, 6144, 384, 4),
        (1024, 128, 6144, 384, 4),
        (8192, 128, 6144, 384, 4),
    ],
)
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
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


class _TensorMetadata:
    """CPU policy fixture, not a substitute for device kernel validation."""

    def __init__(self, shape, dtype, device=None, contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.contiguous = contiguous
        self.ndim = len(shape)

    def size(self, dim):
        return self.shape[dim]

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dim):
        result = 1
        for size in self.shape[dim + 1 :]:
            result *= size
        return result


def _policy_inputs(tokens):
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import _EXPECTED_CONFIGS

    return dict(
        A=_TensorMetadata((tokens, 6144), torch.int8),
        B=_TensorMetadata((128, 768, 6144), torch.int8),
        C=_TensorMetadata((tokens, 4, 768), torch.bfloat16),
        A_scale=_TensorMetadata((tokens, 1), torch.float32),
        B_scale=_TensorMetadata((128, 768, 1), torch.float32),
        top_k=4,
        config=dict(_EXPECTED_CONFIGS.get(tokens, _EXPECTED_CONFIGS[64])),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
        B_bias=None,
        FUSE_SILU=False,
        direct_sum=False,
        mul_routed_weight=False,
    )


@pytest.mark.fused_moe
@pytest.mark.parametrize("tokens", [1, 32, 64, 65, 1024, 8192])
def test_thead_gemm1_policy(tokens, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _policy_inputs(tokens)
    original = dict(args["config"])
    selected = vendor._gemm1_config(**args)
    assert selected["INT8_ACCUMULATOR"] is True
    assert selected["BLOCK_SIZE_K"] == (
        256 if tokens in (64, 1024, 8192) else original["BLOCK_SIZE_K"]
    )
    assert args["config"] == original


@pytest.mark.fused_moe
@pytest.mark.parametrize(
    "override",
    [
        "capability",
        "empty",
        "large",
        "layout",
        "device",
        "dtype",
        "shape",
        "scale",
        "top_k",
        "fp8",
        "int8_disabled",
        "w8a16",
        "w4a16",
        "per_channel",
        "block",
        "bias",
        "fuse",
        "sum",
        "route",
        "swap",
        "explicit_true",
        "explicit_false",
    ],
)
def test_thead_gemm1_policy_fallback(override, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(
        vendor, "_supports_gemm1_tile", lambda _: override != "capability"
    )
    args = _policy_inputs(64)
    if override in ("empty", "large"):
        args = _policy_inputs(0 if override == "empty" else 8193)
    elif override == "layout":
        args["A"].contiguous = False
    elif override == "device":
        args["B"].device = torch.device("cuda:1")
    elif override == "dtype":
        args["A"].dtype = torch.float16
    elif override == "shape":
        args["B"].shape = (128, 769, 6144)
    elif override == "scale":
        args["B_scale"].dtype = torch.bfloat16
    elif override == "top_k":
        args["top_k"] = 2
    elif override in ("fp8", "w8a16", "w4a16"):
        names = {
            "fp8": "use_fp8_w8a8",
            "w8a16": "use_int8_w8a16",
            "w4a16": "use_int4_w4a16",
        }
        args[names[override]] = True
    elif override in ("int8_disabled", "per_channel"):
        args[
            "use_int8_w8a8" if override == "int8_disabled" else "per_channel_quant"
        ] = False
    elif override in ("block", "bias"):
        args["block_shape" if override == "block" else "B_bias"] = [128, 128]
    elif override in ("fuse", "sum", "route"):
        args[
            {"fuse": "FUSE_SILU", "sum": "direct_sum", "route": "mul_routed_weight"}[
                override
            ]
        ] = True
    elif override == "swap":
        args["config"]["SWAP_AB"] = True
    elif override.startswith("explicit"):
        args["config"]["INT8_ACCUMULATOR"] = override == "explicit_true"
    original = dict(args["config"])
    assert vendor._gemm1_config(**args) is args["config"]
    assert args["config"] == original


@pytest.mark.fused_moe
def test_thead_preserves_custom_tile(monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _policy_inputs(64)
    args["config"]["BLOCK_SIZE_K"] = 64
    selected = vendor._gemm1_config(**args)
    assert selected["INT8_ACCUMULATOR"] is True
    assert selected["BLOCK_SIZE_K"] == 64


@pytest.mark.fused_moe
@pytest.mark.parametrize("mismatch", ["name", "major", "minor", "warp_size"])
def test_thead_capability_guard(mismatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    props = dict(name="PPU-ZW810E", major=8, minor=0, warp_size=32)
    props[mismatch] = "other" if mismatch == "name" else 7
    vendor._supports_gemm1_tile.cache_clear()
    with patch.object(
        torch.cuda, "get_device_properties", return_value=SimpleNamespace(**props)
    ):
        assert vendor._supports_gemm1_tile(0) is False
    vendor._supports_gemm1_tile.cache_clear()


@pytest.mark.fused_moe
def test_int8_accumulator_overflow_bound():
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )

    a = torch.empty((1, 131072), dtype=torch.int8, device="meta")
    b = torch.empty((1, 1, 131072), dtype=torch.int8, device="meta")
    out = torch.empty((1, 1, 1), dtype=torch.bfloat16, device="meta")
    a_scale = torch.empty((1, 1), device="meta")
    b_scale = torch.empty((1, 1, 1), device="meta")
    config = dict(
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3,
        INT8_ACCUMULATOR=True,
    )
    with pytest.raises(ValueError, match="INT32_MAX"):
        invoke_fused_moe_triton_kernel(
            a,
            b,
            out,
            a_scale,
            b_scale,
            None,
            None,
            None,
            None,
            False,
            1,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )


@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize(
    "flag", [None, "SWAP_AB", "USE_INT32_OFFSETS", "PAIR_GATE_UP_DOT"]
)
def test_thead_native_tile_defaults(scale_rank, flag, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _policy_inputs(64)
    if scale_rank == 2:
        args["B_scale"].shape = (128, 768)
    if flag is not None:
        args["config"][flag] = False
    args["config"].pop("SPLIT_K")
    original = dict(args["config"])
    selected = vendor._gemm1_config(**args)
    assert selected["INT8_ACCUMULATOR"] is True
    assert selected["BLOCK_SIZE_K"] == 256
    assert args["config"] == original


@pytest.mark.parametrize(
    "override", ["BLOCK_SIZE_K", "num_stages", "USE_INT32_OFFSETS", "unknown"]
)
def test_thead_native_tile_preserves_nondefault_settings(override, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _policy_inputs(64)
    args["config"][override] = {
        "BLOCK_SIZE_K": 64,
        "num_stages": 4,
        "USE_INT32_OFFSETS": True,
        "unknown": 1,
    }[override]
    original = dict(args["config"])
    selected = vendor._gemm1_config(**args)
    assert selected["INT8_ACCUMULATOR"] is True
    assert selected["BLOCK_SIZE_K"] == original["BLOCK_SIZE_K"]
    assert args["config"] == original


@pytest.mark.parametrize("selected", [True, False])
def test_thead_internal_dispatch_selector(selected, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    calls = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def record(*args, **kwargs):
                calls.append(kwargs)

            return record

    monkeypatch.setattr(vendor, "fused_moe_kernel", RecordingKernel())
    args = _policy_inputs(64)
    if not selected:
        args["A"].shape = (64, 6143)
        args["B"].shape = (128, 768, 6143)
    vendor.dispatch_fused_moe_kernel(
        **args,
        B_zp=None,
        topk_weights=None,
        sorted_token_ids=None,
        expert_ids=None,
        num_tokens_post_padded=None,
    )
    assert len(calls) == 1
    assert calls[0].get("INT8_ACCUMULATOR", False) is selected
    assert calls[0]["BLOCK_SIZE_K"] == (256 if selected else 128)


def _gemm2_policy_inputs(tokens=1024):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    return dict(
        A=_TensorMetadata((tokens * 4, 384), torch.int8),
        B=_TensorMetadata((128, 6144, 384), torch.int8),
        C=_TensorMetadata((tokens, 4, 6144), torch.bfloat16),
        A_scale=_TensorMetadata((tokens * 4, 1), torch.float32),
        B_scale=_TensorMetadata((128, 6144, 1), torch.float32),
        topk_weights=_TensorMetadata((tokens, 4), torch.float32),
        top_k=1,
        config=vendor.get_default_config(
            tokens, 128, 6144, 384, 4, "bf16", gemm_stage="gemm2"
        ),
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
        B_bias=None,
        FUSE_SILU=False,
        direct_sum=False,
        mul_routed_weight=True,
    )


@pytest.mark.parametrize("tokens", [1024, 8192])
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
def test_thead_internal_gemm2_policy(tokens, scale_rank, route_dtype, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _gemm2_policy_inputs(tokens)
    if scale_rank == 2:
        args["B_scale"] = _TensorMetadata((128, 6144), torch.float32)
    args["topk_weights"].dtype = route_dtype
    original = dict(args["config"])
    assert vendor._use_gemm2_int32(**args) is True
    calls = []
    monkeypatch.setattr(
        moe, "invoke_fused_moe_triton_kernel", lambda *a, **k: calls.append((a, k))
    )
    vendor.invoke_fused_moe_triton_kernel(
        **args,
        sorted_token_ids=None,
        expert_ids=None,
        num_tokens_post_padded=None,
    )
    assert len(calls) == 1
    forwarded = calls[0][0][11]
    assert "USE_INT32_OFFSETS" not in forwarded
    assert forwarded == {
        key: value for key, value in original.items() if key != "USE_INT32_OFFSETS"
    }
    assert args["config"] == original


@pytest.mark.parametrize(
    "override",
    [
        "small",
        "other_m",
        "capability",
        "shape",
        "layout",
        "device",
        "dtype",
        "scale_dtype",
        "scale_shape",
        "route_dtype",
        "route_shape",
        "no_route",
        "top_k",
        "fp8",
        "w8a16",
        "w4a16",
        "int8_disabled",
        "per_channel",
        "block",
        "bias",
        "fuse",
        "sum",
        "route_disabled",
        "swap",
        "explicit_false",
        "explicit_true",
        "unknown_config",
    ],
)
def test_thead_internal_gemm2_policy_fallback(override, monkeypatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    monkeypatch.setattr(
        vendor, "_supports_gemm1_tile", lambda _: override != "capability"
    )
    args = _gemm2_policy_inputs(
        64 if override == "small" else 1025 if override == "other_m" else 1024
    )
    if override == "shape":
        args["B"].shape = (128, 6144, 385)
    elif override == "layout":
        args["A"].contiguous = False
    elif override == "device":
        args["topk_weights"].device = torch.device("cuda:1")
    elif override == "dtype":
        args["C"].dtype = torch.float16
    elif override == "scale_dtype":
        args["A_scale"].dtype = torch.bfloat16
    elif override == "scale_shape":
        args["B_scale"].shape = (128, 6143, 1)
    elif override == "route_dtype":
        args["topk_weights"].dtype = torch.float16
    elif override == "route_shape":
        args["topk_weights"].shape = (1024, 3)
    elif override == "no_route":
        args["topk_weights"] = None
    elif override == "top_k":
        args["top_k"] = 4
    elif override in ("fp8", "w8a16", "w4a16"):
        args[
            {
                "fp8": "use_fp8_w8a8",
                "w8a16": "use_int8_w8a16",
                "w4a16": "use_int4_w4a16",
            }[override]
        ] = True
    elif override in ("int8_disabled", "per_channel", "route_disabled"):
        args[
            {
                "int8_disabled": "use_int8_w8a8",
                "per_channel": "per_channel_quant",
                "route_disabled": "mul_routed_weight",
            }[override]
        ] = False
    elif override in ("block", "bias"):
        args["block_shape" if override == "block" else "B_bias"] = [128, 128]
    elif override in ("fuse", "sum"):
        args["FUSE_SILU" if override == "fuse" else "direct_sum"] = True
    elif override == "swap":
        args["config"]["SWAP_AB"] = True
    elif override.startswith("explicit"):
        args["config"]["INT8_ACCUMULATOR"] = override == "explicit_true"
    elif override == "unknown_config":
        args["config"]["unknown"] = 1
    original = dict(args["config"])
    assert vendor._use_gemm2_int32(**args) is False
    assert args["config"] == original
