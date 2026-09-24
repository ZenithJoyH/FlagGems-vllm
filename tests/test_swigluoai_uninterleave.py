# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The packed MiniMax-M3 activation is shared by dense and MoE callers."""

import pytest
import torch

import flaggems_vllm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 128), (7, 3072), (0, 128)])
def test_swigluoai_uninterleave(dtype, shape):
    device = flaggems_vllm.runtime.device.name
    x = torch.randn(shape[0], 2 * shape[1], device=device, dtype=dtype)
    gate, up = x.float().chunk(2, dim=-1)
    limit, alpha, beta = 7.0, 1.702, 1.0
    clamped_gate = gate.clamp(max=limit)
    expected = (
        clamped_gate
        * torch.sigmoid(alpha * clamped_gate)
        * (up.clamp(min=-limit, max=limit) + beta)
    ).to(dtype)
    out = torch.empty(shape, device=device, dtype=dtype)
    result = flaggems_vllm.swigluoai_uninterleave(
        x, limit, alpha, beta, out=out
    )
    assert result is out
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    allocated = flaggems_vllm.swigluoai_uninterleave(x, limit, alpha, beta)
    torch.testing.assert_close(allocated, expected, rtol=1e-2, atol=1e-2)


def test_swigluoai_uninterleave_rejects_alias():
    device = flaggems_vllm.runtime.device.name
    x = torch.empty(2, 256, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="alias"):
        flaggems_vllm.swigluoai_uninterleave(
            x, 7.0, 1.702, 1.0, out=x[:, :128]
        )
