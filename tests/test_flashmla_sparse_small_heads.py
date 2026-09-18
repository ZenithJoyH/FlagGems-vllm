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


def _inputs(heads: int):
    torch.manual_seed(20260907)
    q = (torch.randn(2, heads, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(256, 1, 576, device="cuda") * 0.05).to(torch.bfloat16)
    indices = torch.arange(128, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 128).expand(2, 1, 128).contiguous()
    lengths = torch.tensor([96, 127], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    return q, kv, indices, lengths, sinks


def _reference(q, kv, indices, lengths, sinks, scale):
    outputs, maxima, lses = [], [], []
    for token in range(q.shape[0]):
        ids = indices[token, 0, : lengths[token]].long()
        keys = kv[ids, 0].float()
        scores = torch.einsum("hd,kd->hk", q[token].float(), keys) * scale
        maximum = scores.max(dim=-1).values
        lse = torch.logsumexp(scores, dim=-1)
        weights = torch.exp(scores - maximum[:, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks - maximum)
        outputs.append(
            (torch.einsum("hk,kd->hd", weights, keys[:, :512]) / denominator[:, None])
            .to(torch.bfloat16)
        )
        maxima.append(maximum)
        lses.append(lse)
    return torch.stack(outputs), torch.stack(maxima), torch.stack(lses)


@pytest.mark.parametrize("heads", [4, 64])
def test_flash_mla_sparse_supports_small_head_counts(heads):
    q, kv, indices, lengths, sinks = _inputs(heads)
    scale = 576**-0.5
    expected = _reference(q, kv, indices, lengths, sinks, scale)
    actual = flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


def test_flash_mla_sparse_small_heads_graph_replay_changed_input():
    q, kv, indices, lengths, sinks = _inputs(4)
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
    q.add_(0.01)
    expected = _reference(q, kv, indices, lengths, sinks, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)
