# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton
from triton import language as tl

import flaggems_vllm


@pytest.mark.fused_moe
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU-specific policy")
@pytest.mark.parametrize("tokens", [1, 8, 32, 64, 65, 1024, 8192])
@pytest.mark.parametrize("cancel", [False, True])
def test_int8_gemm1_exact_accumulation_graph(tokens, cancel):
    props = torch.cuda.get_device_properties(0)
    if props.name != "PPU-ZW810E":
        pytest.skip("Requires the measured 810E capability")
    assert inspect.getfile(flaggems_vllm.invoke_fused_moe_triton_kernel).endswith(
        "fused_moe_int8.py"
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
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe_int8 import (
        _EXPECTED_CONFIGS,
    )

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

    def size(self, dim):
        return self.shape[dim]

    def is_contiguous(self):
        return self.contiguous


def _policy_inputs(tokens):
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe_int8 import (
        _EXPECTED_CONFIGS,
    )

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
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe_int8 as vendor

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
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe_int8 as vendor

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
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe_int8 as vendor

    monkeypatch.setattr(vendor, "_supports_gemm1_tile", lambda _: True)
    args = _policy_inputs(64)
    args["config"]["BLOCK_SIZE_K"] = 64
    selected = vendor._gemm1_config(**args)
    assert selected["INT8_ACCUMULATOR"] is True
    assert selected["BLOCK_SIZE_K"] == 64


@pytest.mark.fused_moe
@pytest.mark.parametrize("mismatch", ["name", "major", "minor", "warp_size"])
def test_thead_capability_guard(mismatch):
    from flaggems_vllm.runtime.backend._thead.ops import fused_moe_int8 as vendor

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
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel

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


@pytest.mark.fused_moe
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU kernel regression"
)
@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize("route_dtype", [torch.float32, torch.bfloat16])
def test_existing_int32_down_path_graph(tokens, route_dtype):
    """The upstream INT32_DOWN schedule remains independent of the new opt-in."""
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel
    from flaggems_vllm.runtime.backend._thead.ops.moe_align_block_size import (
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
