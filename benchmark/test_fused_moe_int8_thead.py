# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired GEMM1-only graph timing against the original FP32 schedule.

The public API is a fused MoE launcher, not a Torch primitive. The baseline is
the same generic Triton launcher with the new policy explicitly disabled.
Alignment is prepared once outside timing; quantization/activation/GEMM2/sum
are excluded. Functional tests independently check integer reference results.
"""

import json
import statistics

import pytest
import torch
import triton
from triton import language as tl

import flaggems_vllm


@pytest.mark.fused_moe
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="810E benchmark")
@pytest.mark.parametrize("tokens", [32, 64, 65, 1024, 8192])
def test_fused_moe_int8_thead_gemm1(tokens):
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel as baseline
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe_int8 import (
        _EXPECTED_CONFIGS,
        _supports_gemm1_tile,
    )
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe_int8 import (
        invoke_fused_moe_triton_kernel as candidate,
    )
    from flaggems_vllm.runtime.backend._thead.ops.moe_align_block_size import (
        moe_align_block_size_triton,
    )

    if not _supports_gemm1_tile(torch.cuda.current_device()):
        pytest.skip("Requires the measured 810E capability")
    torch.manual_seed(42)
    a = torch.randint(-128, 128, (tokens, 6144), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 128, (128, 768, 6144), dtype=torch.int8, device="cuda")
    a_scale = torch.full((tokens, 1), 0.03125, device="cuda")
    b_scale = torch.full((128, 768, 1), 0.015625, device="cuda")
    out = torch.empty((tokens, 4, 768), dtype=torch.bfloat16, device="cuda")
    ids = (torch.arange(tokens * 4, dtype=torch.int32, device="cuda") % 128).reshape(
        tokens, 4
    )
    config = dict(_EXPECTED_CONFIGS.get(tokens, _EXPECTED_CONFIGS[64]))
    bm = config["BLOCK_SIZE_M"]
    size = min(ids.numel() * bm, ids.numel() + 128 * (bm - 1))
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, bm), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")
    moe_align_block_size_triton(ids, 128, bm, sorted_ids, expert_ids, padded)

    def launch(op, cfg):
        op(
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
            cfg,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    baseline_config = dict(config, INT8_ACCUMULATOR=False)
    launch(baseline, baseline_config)
    launch(candidate, config)
    torch.cuda.synchronize()
    timings = {"baseline": [], "candidate": []}
    for trial in range(5):
        order = (
            ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        )
        for name in order:
            op, cfg = (
                (baseline, baseline_config)
                if name == "baseline"
                else (candidate, config)
            )
            timings[name].append(
                triton.testing.do_bench_cudagraph(lambda: launch(op, cfg), rep=100)
            )
    before = statistics.median(timings["baseline"])
    after = statistics.median(timings["candidate"])
    print(
        json.dumps(
            {
                "tokens": tokens,
                "scope": "GEMM1 only",
                "unit": "ms",
                "baseline_ms": before,
                "candidate_ms": after,
                "speedup": before / after,
                "paired_timings": timings,
            }
        )
    )
