# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Preliminary packed SwiGLU-OAI benchmark (device required)."""

import pytest
import torch
import triton.testing

import flaggems_vllm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-compatible GPU needed")
@pytest.mark.parametrize("shape", [(1, 3072), (32, 3072), (256, 3072)])
def test_swigluoai_uninterleave_benchmark(shape):
    device = flaggems_vllm.runtime.device.name
    x = torch.randn(shape[0], shape[1] * 2, device=device, dtype=torch.bfloat16)
    out = torch.empty(shape, device=device, dtype=torch.bfloat16)

    def reference():
        gate, up = x.float().chunk(2, -1)
        gate = gate.clamp(max=7.0)
        out.copy_(
            (gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7.0, 7.0) + 1.0))
            .to(out.dtype)
        )

    def gems():
        flaggems_vllm.swigluoai_uninterleave(x, 7.0, 1.702, 1.0, out=out)

    baseline_ms = triton.testing.do_bench(reference)
    gems_ms = triton.testing.do_bench(gems)
    speedup = baseline_ms / gems_ms
    print(
        f"shape={shape} baseline_ms={baseline_ms:.4f} "
        f"gems_ms={gems_ms:.4f} speedup={speedup:.3f}"
    )
