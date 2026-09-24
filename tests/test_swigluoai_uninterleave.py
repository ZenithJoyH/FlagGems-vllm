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


def test_swigluoai_uninterleave_rejects_noncontiguous_nonalias_out():
    device = flaggems_vllm.runtime.device.name
    x = torch.empty(2, 256, device=device, dtype=torch.bfloat16)
    storage = torch.empty(2, 256, device=device, dtype=torch.bfloat16)
    out = storage[:, ::2]
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.swigluoai_uninterleave(x, 7.0, 1.702, 1.0, out=out)


def test_swigluoai_uninterleave_allows_empty_view():
    device = flaggems_vllm.runtime.device.name
    x = torch.empty(0, 256, device=device, dtype=torch.bfloat16)
    out = x[:, :128]
    result = flaggems_vllm.swigluoai_uninterleave(
        x, 7.0, 1.702, 1.0, out=out
    )
    assert result is out
    assert result.shape == (0, 128)


def test_swigluoai_uninterleave_alias_metadata_with_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flaggems_vllm.ops.swigluoai_uninterleave import _storage_overlaps

    with FakeTensorMode():
        x = torch.empty(2, 256)
        independent_out = torch.empty(2, 128)
        alias_out = x[:, :128]
        workspace = torch.empty(1024)
        workspace_input = workspace[:512].view(2, 256)
        disjoint_out = workspace[512:768].view(2, 128)
        assert not _storage_overlaps(x, independent_out)
        assert _storage_overlaps(x, alias_out)
        assert not _storage_overlaps(workspace_input, disjoint_out)


def test_swigluoai_uninterleave_allows_disjoint_shared_workspace():
    device = flaggems_vllm.runtime.device.name
    workspace = torch.empty(768, device=device, dtype=torch.bfloat16)
    x = workspace[:512].view(2, 256)
    out = workspace[512:].view(2, 128)
    x.copy_(torch.randn_like(x))
    expected = flaggems_vllm.swigluoai_uninterleave(x, 7.0, 1.702, 1.0)
    result = flaggems_vllm.swigluoai_uninterleave(
        x, 7.0, 1.702, 1.0, out=out
    )
    assert result is out
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
