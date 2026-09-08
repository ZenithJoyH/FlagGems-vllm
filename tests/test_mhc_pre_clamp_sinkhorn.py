# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm


def _reference(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    clamp_min,
    clamp_max,
    hc_post_mult_value,
    sinkhorn_repeat,
):
    dtype = residual.dtype
    flat = residual.flatten(start_dim=-2).float()
    normalized = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(normalized.to(dtype), fn.to(dtype)).float()
    pre_mix, post_mix, comb_mix = mixes.split((4, 4, 16), dim=-1)
    pre_mix = torch.sigmoid(pre_mix * hc_scale[0] + hc_base[:4])
    post_mix = (
        torch.sigmoid(post_mix * hc_scale[1] + hc_base[4:8])
        * hc_post_mult_value
    )
    comb_mix = torch.clamp(
        comb_mix * hc_scale[2] + hc_base[8:], clamp_min, clamp_max
    ).view(-1, 4, 4)
    comb_mix = torch.exp(comb_mix - comb_mix.max(dim=-1, keepdim=True).values)
    for _ in range(sinkhorn_repeat):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + rms_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + rms_eps)
    layer_input = (pre_mix.unsqueeze(-1).to(dtype) * residual).sum(dim=-2)
    outer_shape = residual.shape[:-2]
    return (
        post_mix.to(dtype).view(*outer_shape, 4, 1),
        comb_mix.to(dtype).view(*outer_shape, 4, 4),
        layer_input.to(dtype).view(*outer_shape, residual.shape[-1]),
    )


def _make_inputs(num_tokens, hidden_size):
    torch.manual_seed(42)
    device = flaggems_vllm.device
    residual = torch.randn(
        (num_tokens, 4, hidden_size), dtype=torch.bfloat16, device=device
    )
    fn = torch.randn((24, 4 * hidden_size), dtype=torch.float32, device=device) * 1e-3
    hc_scale = torch.tensor((0.3, -0.2, 2.0), dtype=torch.float32, device=device)
    hc_base = torch.linspace(-60, 60, 24, dtype=torch.float32, device=device)
    return residual, fn, hc_scale, hc_base


@pytest.mark.mhc_pre
@pytest.mark.parametrize("num_tokens,hidden_size", [(1, 1280), (7, 1280), (16, 2560)])
def test_mhc_pre_clamp_sinkhorn_matches_reference(num_tokens, hidden_size):
    args = _make_inputs(num_tokens, hidden_size)
    kwargs = dict(
        rms_eps=1e-6,
        clamp_min=-30.0,
        clamp_max=30.0,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    actual = flaggems_vllm.mhc_pre_clamp_sinkhorn(*args, **kwargs)
    expected = _reference(*args, **kwargs)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2
        )


@pytest.mark.mhc_pre
def test_mhc_pre_clamp_sinkhorn_empty():
    args = _make_inputs(0, 1280)
    post, comb, layer_input = flaggems_vllm.mhc_pre_clamp_sinkhorn(
        *args, 1e-6, -30.0, 30.0, 2.0, 20
    )
    assert post.shape == (0, 4, 1)
    assert comb.shape == (0, 4, 4)
    assert layer_input.shape == (0, 1280)


@pytest.mark.mhc_pre
def test_mhc_pre_clamp_sinkhorn_rejects_unsupported_layout():
    residual, fn, hc_scale, hc_base = _make_inputs(2, 1280)
    residual = residual.transpose(0, 1).contiguous().transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        flaggems_vllm.mhc_pre_clamp_sinkhorn(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            -30.0,
            30.0,
            2.0,
            20,
        )


@pytest.mark.mhc_pre
def test_mhc_pre_clamp_sinkhorn_accepts_inference_parameters():
    residual, fn, hc_scale, hc_base = _make_inputs(1, 1280)
    fn.requires_grad_(True)
    with torch.no_grad():
        actual = flaggems_vllm.mhc_pre_clamp_sinkhorn(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            -30.0,
            30.0,
            2.0,
            20,
        )
    expected = _reference(
        residual, fn, hc_scale, hc_base, 1e-6, -30.0, 30.0, 2.0, 20
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2
        )


@pytest.mark.mhc_pre
def test_mhc_pre_clamp_sinkhorn_cuda_graph_replay():
    if flaggems_vllm.device != "cuda":
        pytest.skip("CUDA graph validation requires the NVIDIA backend")
    args = _make_inputs(7, 1280)
    kwargs = dict(
        rms_eps=1e-6,
        clamp_min=-30.0,
        clamp_max=30.0,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    flaggems_vllm.mhc_pre_clamp_sinkhorn(*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.mhc_pre_clamp_sinkhorn(*args, **kwargs)
    graph.replay()
    expected = _reference(*args, **kwargs)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2
        )
