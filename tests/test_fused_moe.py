# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton
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
@pytest.mark.fused_moe
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU-specific policy")
@pytest.mark.parametrize("tokens", [1, 8, 32, 64, 65, 1024, 8192])
@pytest.mark.parametrize("cancel", [False, True])
def test_int8_gemm1_exact_accumulation_graph(tokens, cancel):
    props = torch.cuda.get_device_properties(0)
    if props.name != "PPU-ZW810E":
        pytest.skip("Requires the measured 810E capability")
    assert inspect.getfile(flaggems_vllm.invoke_fused_moe_triton_kernel).endswith(
        "_thead/ops/fused_moe.py"
    )
    a = torch.full((tokens, 6144), 127, dtype=torch.int8, device="cuda")
    b = torch.full((128, 768, 6144), 127, dtype=torch.int8, device="cuda")
    expected_integer = 6144 * 127 * 127
    if cancel:
        b[:, :, 3072:] = -127
        b[:, :, -1] = -126
        expected_integer = 127
    a_scale = torch.full((tokens, 1), 0.03125, device="cuda")
    b_scale = torch.full((128, 768, 1), 0.015625, device="cuda")
    out = torch.empty((tokens, 4, 768), dtype=torch.bfloat16, device="cuda")
    ids = (
        torch.arange(4, device="cuda", dtype=torch.int32).expand(tokens, 4).contiguous()
    )
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import _EXPECTED_CONFIGS

    config = dict(_EXPECTED_CONFIGS.get(tokens, _EXPECTED_CONFIGS[64]))
    bm = config["BLOCK_SIZE_M"]
    size = min(ids.numel() * bm, ids.numel() + 128 * (bm - 1))
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, bm), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")

    def run():
        flaggems_vllm.moe_align_block_size_triton(
            ids, 128, bm, sorted_ids, expert_ids, padded
        )
        flaggems_vllm.invoke_fused_moe_triton_kernel(
            a,
            b,
            out,
            a_scale,
            b_scale,
            None,
            sorted_ids,
            expert_ids,
            padded,
            False,
            4,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.full_like(out, expected_integer * 0.03125 * 0.015625)
    assert torch.equal(out, expected)


@requires_cuda
@pytest.mark.fused_moe
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU-specific policy")
def test_empty_moe_alignment_graph():
    ids = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    sorted_ids = torch.empty(0, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(0, dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")

    def run():
        flaggems_vllm.moe_align_block_size_triton(
            ids, 128, 32, sorted_ids, expert_ids, padded
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert padded.item() == 0


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


@requires_cuda
@pytest.mark.fused_moe
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU kernel regression"
)
@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
def test_existing_int32_down_path_graph(tokens, route_dtype):
    """The upstream INT32_DOWN schedule remains independent of the new opt-in."""
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        invoke_fused_moe_triton_kernel,
        moe_align_block_size_triton,
    )

    a = torch.full((tokens * 4, 384), 127, dtype=torch.int8, device="cuda")
    b = torch.full((128, 6144, 384), 127, dtype=torch.int8, device="cuda")
    b[:, :, 192:] = -127
    b[:, :, -1] = -126
    a_scale = torch.full((tokens * 4, 1), 0.03125, device="cuda")
    b_scale = torch.full((128, 6144, 1), 0.015625, device="cuda")
    weights = torch.tensor([0.31, 0.23, 0.17, 0.29], dtype=route_dtype, device="cuda")
    weights = weights.expand(tokens, 4).contiguous()
    out = torch.empty((tokens, 4, 6144), dtype=torch.bfloat16, device="cuda")
    ids = (
        torch.arange(4, device="cuda", dtype=torch.int32).expand(tokens, 4).contiguous()
    )
    size = min(ids.numel() * 32, ids.numel() + 128 * 31)
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, 32), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")
    config = dict(
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3,
    )

    def run():
        moe_align_block_size_triton(ids, 128, 32, sorted_ids, expert_ids, padded)
        invoke_fused_moe_triton_kernel(
            a,
            b,
            out,
            a_scale,
            b_scale,
            weights,
            sorted_ids,
            expert_ids,
            padded,
            True,
            1,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    expected = ((127 * 0.03125 * 0.015625) * weights.cpu().float()).to(torch.bfloat16)
    torch.testing.assert_close(
        out.cpu(), expected[:, :, None].expand_as(out.cpu()), rtol=0, atol=0
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


@requires_cuda
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU kernel regression"
)
@pytest.mark.parametrize("tokens", [8, 64, 65])
@pytest.mark.parametrize("scale_rank", [2, 3])
def test_int8_gemm1_varied_routes_scales_graph(tokens, scale_rank):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    if not vendor._supports_gemm1_tile(torch.cuda.current_device()):
        pytest.skip("Requires 810E")
    generator = torch.Generator().manual_seed(81)
    positions = torch.tensor([0, 31, 127, 255, 256, 1023, 2047, 3071, 4095, 6143])
    a_cpu = torch.randint(
        -128, 128, (tokens, 6144), dtype=torch.int8, generator=generator
    )
    coeff = torch.randint(
        -128, 128, (128, 768, positions.numel()), dtype=torch.int8, generator=generator
    )
    a_cpu[0, 0] = -128
    coeff[:, :, 0] = -128
    ids_cpu = torch.randint(0, 128, (tokens, 4), dtype=torch.int32, generator=generator)
    # Ensure the last expert and distinct routes are covered, including padding.
    ids_cpu[0] = torch.tensor([0, 5, 63, 127], dtype=torch.int32)
    a_scale_cpu = torch.randint(1, 6, (tokens, 1), generator=generator).float() / 128
    b_scale_cpu = torch.randint(1, 6, (128, 768), generator=generator).float() / 256
    a = a_cpu.cuda()
    b = torch.zeros((128, 768, 6144), dtype=torch.int8, device="cuda")
    b[:, :, positions.cuda()] = coeff.cuda()
    ids = ids_cpu.cuda()
    a_scale = a_scale_cpu.cuda()
    b_scale = b_scale_cpu.cuda()
    if scale_rank == 3:
        b_scale = b_scale.unsqueeze(-1)
    out = torch.empty((tokens, 4, 768), dtype=torch.bfloat16, device="cuda")
    config = dict(vendor._EXPECTED_CONFIGS.get(tokens, vendor._EXPECTED_CONFIGS[64]))
    bm = config["BLOCK_SIZE_M"]
    size = min(ids.numel() * bm, ids.numel() + 128 * (bm - 1))
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, bm), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")

    def reference(routes):
        products = a_cpu[:, None, None, positions].to(torch.int64) * coeff[
            routes.long()
        ].to(torch.int64)
        dots = products.sum(-1).float()
        return (dots * a_scale_cpu[:, None, :] * b_scale_cpu[routes.long()]).to(
            torch.bfloat16
        )

    def run():
        flaggems_vllm.moe_align_block_size_triton(
            ids, 128, bm, sorted_ids, expert_ids, padded
        )
        flaggems_vllm.invoke_fused_moe_triton_kernel(
            a,
            b,
            out,
            a_scale,
            b_scale,
            None,
            sorted_ids,
            expert_ids,
            padded,
            False,
            4,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference(ids_cpu), rtol=0, atol=0)
    # Replay with new routes so stale scratch/output cannot satisfy the assertion.
    ids_cpu = (ids_cpu + 17) % 128
    ids.copy_(ids_cpu)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference(ids_cpu), rtol=0, atol=0)


@requires_cuda
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU full-experts regression"
)
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("tokens", [64, 1024, 8192])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
def test_thead_full_experts_int8_selection_graph(
    scale_rank, tokens, route_dtype, monkeypatch
):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    if not vendor._supports_gemm1_tile(torch.cuda.current_device()):
        pytest.skip("Requires 810E")
    generator = torch.Generator().manual_seed(81)
    hidden_cpu = (torch.randn(tokens, 6144, generator=generator) * 0.25).to(
        torch.bfloat16
    )
    routes_cpu = torch.randint(
        0, 128, (tokens, 4), generator=generator, dtype=torch.int32
    )
    router_cpu = torch.rand(tokens, 4, generator=generator)
    router_cpu /= router_cpu.sum(-1, keepdim=True)
    router_cpu = router_cpu.to(route_dtype)
    channels1, channels2 = torch.arange(768), torch.arange(6144)
    experts = torch.arange(128)[:, None]
    index1 = (channels1[None, :] * 7 + experts * 13) % 6144
    index2 = (channels2[None, :] * 3 + experts * 11) % 384
    coeff1 = ((channels1[None, :] + experts) % 7 - 3).to(torch.int8)
    coeff2 = ((channels2[None, :] + experts * 2) % 7 - 3).to(torch.int8)
    scale1 = (1 + ((channels1[None, :] + experts) % 4)).float() / 64
    scale2 = (1 + ((channels2[None, :] + experts) % 4)).float() / 32
    w1 = torch.zeros((128, 768, 6144), dtype=torch.int8, device="cuda")
    w2 = torch.zeros((128, 6144, 384), dtype=torch.int8, device="cuda")
    w1[experts.cuda(), channels1.cuda()[None, :], index1.cuda()] = coeff1.cuda()
    w2[experts.cuda(), channels2.cuda()[None, :], index2.cuda()] = coeff2.cuda()
    w1_scale, w2_scale = scale1.cuda(), scale2.cuda()
    if scale_rank == 3:
        w1_scale, w2_scale = w1_scale.unsqueeze(-1), w2_scale.unsqueeze(-1)
    hidden, routes, router = hidden_cpu.cuda(), routes_cpu.cuda(), router_cpu.cuda()

    def quant(x):
        scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
        return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale

    expected_chunks = []
    for begin in range(0, tokens if tokens == 64 else 0, 128):
        end = min(begin + 128, tokens)
        choice = routes_cpu[begin:end].long()
        q1, a1_scale = quant(hidden_cpu[begin:end])
        ref1 = q1[:, None, :].expand(-1, 4, -1).gather(2, index1[choice]).int()
        ref1 = (ref1 * coeff1[choice].int()).float()
        ref1 = (ref1 * a1_scale[:, None, :] * scale1[choice]).to(torch.bfloat16)
        gate, up = ref1.float().chunk(2, dim=-1)
        activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        q2, a2_scale = quant(activated)
        ref2 = q2.gather(2, index2[choice]).int()
        ref2 = (ref2 * coeff2[choice].int()).float()
        ref2 = ref2 * a2_scale * scale2[choice] * router_cpu[begin:end, :, None].float()
        expected_chunks.append(
            ref2.to(torch.bfloat16).float().sum(1).to(torch.bfloat16)
        )
    expected = torch.cat(expected_chunks) if expected_chunks else None

    records = []

    class RecordingKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                records.append(kwargs)
                return self.kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(
        vendor, "fused_moe_kernel", RecordingKernel(vendor.fused_moe_kernel)
    )
    monkeypatch.setattr(moe, "fused_moe_kernel", RecordingKernel(moe.fused_moe_kernel))

    quantizer = vendor.moe_kernel_quantize_input
    stage_inputs = []
    capture_stage = False

    def record_quantizer(*args, **kwargs):
        quantized, scale = quantizer(*args, **kwargs)
        if capture_stage and quantized.size(-1) == 384:
            stage_inputs.append((quantized.cpu(), scale.cpu()))
        return quantized, scale

    monkeypatch.setattr(vendor, "moe_kernel_quantize_input", record_quantizer)

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

    selector = vendor._use_gemm2_int32
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(vendor, "_use_gemm2_int32", lambda *a, **k: False)
        baseline = run().clone()
    records.clear()
    capture_stage = True
    run()
    capture_stage = False
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    gemm1 = [record for record in records if record["top_k"] == 4]
    assert gemm1 and all(
        record["INT8_ACCUMULATOR"] and not record["FUSE_SILU"] for record in gemm1
    )
    # The complete API's existing custom tile stays intact; K256 applies only
    # when its configuration matches one of the measured native configurations.
    assert all(
        record["BLOCK_SIZE_K"] == (32 if tokens == 8192 else 64) for record in gemm1
    )
    gemm2 = [record for record in records if record["top_k"] == 1]
    assert gemm2 and all(
        record.get("INT32_DOWN", False) == (tokens in (1024, 8192)) for record in gemm2
    )
    assert vendor._use_gemm2_int32 is selector
    torch.testing.assert_close(out, baseline, rtol=0, atol=0)
    if tokens == 64:
        torch.testing.assert_close(out.cpu(), expected, rtol=0.03, atol=1e-7)
    else:
        # Large-batch native SiLU/quantization can cross BF16 rounding ties
        # differently from a CPU float reference, even on the unchanged path.
        # Preserve exact whole-call equivalence above, and independently verify
        # GEMM2 plus reduction against an integer oracle for its actual inputs.
        assert len(stage_inputs) == 1
        actual_q2, actual_scale = stage_inputs[0]
        actual_q2 = actual_q2.reshape(tokens, 4, 384)
        actual_scale = actual_scale.reshape(tokens, 4, 1)
        for begin in range(0, tokens, 128):
            end = min(begin + 128, tokens)
            choice = routes_cpu[begin:end].long()
            dots = actual_q2[begin:end].gather(2, index2[choice]).int()
            dots = (dots * coeff2[choice].int()).float()
            down = dots * actual_scale[begin:end] * scale2[choice]
            down = (down * router_cpu[begin:end, :, None].float()).to(torch.bfloat16)
            reference = down.float().sum(1).to(torch.bfloat16)
            torch.testing.assert_close(out[begin:end].cpu(), reference, rtol=0, atol=0)


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
    vendor._invoke_fused_moe_triton_kernel(
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


@requires_cuda
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU internal GEMM2 regression"
)
@pytest.mark.parametrize("tokens", [1024, 8192])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale_rank", [2, 3])
def test_thead_internal_gemm2_integer_oracle_graph(
    tokens, route_dtype, scale_rank, monkeypatch
):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe as vendor

    if not vendor._supports_gemm1_tile(torch.cuda.current_device()):
        pytest.skip("Requires 810E")
    generator = torch.Generator().manual_seed(81)
    values = torch.randint(
        -128, 128, (tokens * 4, 1), dtype=torch.int8, generator=generator
    )
    values[0, 0] = -128
    a = values.expand(-1, 384).contiguous().cuda()
    last = torch.randint(-128, 128, (128, 6144), dtype=torch.int8, generator=generator)
    b = torch.full((128, 6144, 384), 127, dtype=torch.int8, device="cuda")
    b[:, :, 192:] = -127
    b[:, :, -1] = last.cuda()
    a_scale_cpu = torch.rand((tokens * 4, 1), generator=generator) * 0.03 + 0.01
    b_scale_cpu = torch.rand((128, 6144), generator=generator) * 0.02 + 0.01
    weights_cpu = torch.rand((tokens, 4), generator=generator).to(route_dtype)
    ids_cpu = torch.randint(0, 128, (tokens, 4), dtype=torch.int32, generator=generator)
    a_scale, b_scale = a_scale_cpu.cuda(), b_scale_cpu.cuda()
    if scale_rank == 3:
        b_scale = b_scale.unsqueeze(-1)
    weights, ids = weights_cpu.cuda(), ids_cpu.cuda()
    out = torch.empty((tokens, 4, 6144), dtype=torch.bfloat16, device="cuda")
    config = vendor.get_default_config(
        tokens, 128, 6144, 384, 4, "bf16", gemm_stage="gemm2"
    )
    bm = config["BLOCK_SIZE_M"]
    size = ids.numel() + 128 * (bm - 1)
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, bm), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")
    kernel = moe.fused_moe_kernel
    records = []

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                records.append(kwargs)
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(moe, "fused_moe_kernel", RecordingKernel())

    def run():
        vendor.moe_align_block_size_triton(ids, 128, bm, sorted_ids, expert_ids, padded)
        vendor._invoke_fused_moe_triton_kernel(
            a,
            b,
            out,
            a_scale,
            b_scale,
            weights,
            sorted_ids,
            expert_ids,
            padded,
            True,
            1,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert records and all(record["INT32_DOWN"] for record in records)
    for begin in range(0, tokens, 128):
        end = min(begin + 128, tokens)
        choice = ids_cpu[begin:end].long()
        dots = values[begin * 4 : end * 4].reshape(-1, 4, 1).to(torch.int64) * (
            last[choice].to(torch.int64) + 127
        )
        expected = (
            dots.float()
            * a_scale_cpu.reshape(tokens, 4, 1)[begin:end]
            * b_scale_cpu[choice]
        )
        expected = (expected * weights_cpu[begin:end, :, None].float()).to(
            torch.bfloat16
        )
        torch.testing.assert_close(out[begin:end].cpu(), expected, rtol=0, atol=0)
