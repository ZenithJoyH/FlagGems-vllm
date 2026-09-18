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

import flaggems_vllm
from flag_gems.fused.flashmla_sparse import (
    flash_mla_sparse_fwd as generic_flash_mla_sparse_fwd,
)
from flaggems_vllm.runtime.backend._thead.fused.flashmla_sparse import (
    _can_use_thead_hq4_prefill,
    _can_use_thead_splitk,
)
from flaggems_vllm.runtime.backend._thead.ops import fused_moe as thead_fused_moe

from . import accuracy_utils as utils


pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="T-Head only"
)


def _sparse_inputs(sq: int):
    torch.manual_seed(20260918 + sq)
    q = (torch.randn(sq, 4, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(2048, 1, 576, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    indices = torch.arange(2048, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 2048).expand(sq, 1, 2048).contiguous()
    lengths = torch.linspace(1, 2048, sq, device="cuda").to(torch.int32)
    sinks = torch.randn(4, device="cuda", dtype=torch.float32)
    return q, kv, indices, lengths, sinks


def _quantize_reference(x: torch.Tensor):
    x_flat = x.reshape(-1, x.shape[-1])
    scale = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10).float() / 127
    quantized = (x_flat.float() / scale).round().clamp(-128, 127)
    return quantized.reshape(x.shape), scale.reshape(x.shape[:-1] + (1,))


def test_hy4_thead_backend_dispatch():
    assert flaggems_vllm.fused_experts_impl.__module__.endswith(
        "_thead.ops.fused_moe"
    )
    assert flaggems_vllm.moe_sum.__module__.endswith("_thead.fused.moe_sum")
    assert flaggems_vllm.flash_mla_sparse_fwd.__module__.endswith(
        "_thead.fused.flashmla_sparse"
    )


def test_hy4_w8a8_config_is_not_plain_bf16():
    dtype = thead_fused_moe._get_config_dtype_str(
        dtype=torch.bfloat16, use_int8_w8a8=True
    )
    assert dtype == "int8_w8a8"
    config = thead_fused_moe.get_default_config(
        2048, 256, 256, 6144, 8, dtype
    )
    assert config == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
        "USE_INT32_OFFSETS": False,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 256, 6144])
def test_hy4_dynamic_per_token_int8_quant_is_exact(dtype, hidden_size):
    generator = torch.Generator(device="cpu").manual_seed(20260918 + hidden_size)
    x = torch.randn(7, hidden_size, generator=generator, dtype=dtype)
    x[0].zero_()
    x = x.to("cuda")
    expected_q, expected_scale = _quantize_reference(x)
    actual_q, actual_scale = thead_fused_moe._int8_quantize(
        x, A_scale=None, per_act_token=True
    )
    assert torch.equal(actual_q, expected_q.to(torch.int8))
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-3, atol=1e-12)


@pytest.mark.parametrize("m", [1, 256, 513, 2048])
def test_hy4_moe_sum_specialization(m):
    inp = torch.randn((m, 8, 6144), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((m, 6144), dtype=inp.dtype, device=inp.device)
    reference = torch.sum(utils.to_reference(inp), dim=1)
    flaggems_vllm.moe_sum(inp, output)
    utils.gems_assert_close(output, reference, inp.dtype)


@pytest.mark.parametrize("sq", [1, 64, 128, 2048])
def test_hy4_sparse_mla_specialization(sq):
    q, kv, indices, lengths, sinks = _sparse_inputs(sq)
    if sq <= 64:
        assert _can_use_thead_splitk(q, kv, indices, 512, sinks, lengths)
    else:
        assert _can_use_thead_hq4_prefill(q, kv, indices, 512, sinks, lengths)
    scale = 576**-0.5
    expected = generic_flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    actual = flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2
        )


def test_hy4_sparse_mla_graph_replay():
    q, kv, indices, lengths, sinks = _sparse_inputs(128)
    scale = 576**-0.5
    flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.flash_mla_sparse_fwd(
            q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
        )
    graph.replay()
    graph.replay()
    q.add_(0.01)
    graph.replay()
    torch.cuda.synchronize()
    expected = generic_flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2
        )
