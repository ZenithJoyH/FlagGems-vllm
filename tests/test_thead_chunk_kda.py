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

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from flaggems_vllm.runtime.backend._thead.ops import chunk_kda


def make_inputs(tokens: int = 512):
    q = torch.empty((1, tokens, 4, 128), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    raw_g = torch.empty_like(q)
    beta = torch.empty((1, tokens, 4), dtype=torch.float32)
    A_log = torch.empty((4,), dtype=torch.float32)
    g_bias = torch.empty((4 * 128,), dtype=torch.float32)
    initial_state = torch.empty((1, 4, 128, 128), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.long)
    return q, k, v, raw_g, beta, A_log, g_bias, initial_state, cu_seqlens


def test_long_prefill_routes_to_flaggems_vllm_candidate(monkeypatch):
    inputs = make_inputs()
    q, k, v, raw_g, beta, A_log, g_bias, initial_state, cu_seqlens = inputs
    chunk_indices = torch.tensor([[0, 0]], dtype=torch.long)
    metadata = SimpleNamespace(
        fl_kda_prefill_max_query_len=512,
        num_spec_decodes=0,
        fl_kda_total_tokens=512,
        fl_kda_num_sequences=1,
        fl_kda_cu_seqlens_long=cu_seqlens,
        fl_kda_chunk_indices_16=chunk_indices,
    )
    candidate_fn = Mock(return_value=(torch.empty_like(q), initial_state))
    fallback_fn = Mock()
    monkeypatch.setattr(chunk_kda, "_get_prefill_metadata", lambda: metadata)
    monkeypatch.setattr(
        chunk_kda,
        "_candidate_module",
        lambda: SimpleNamespace(chunk_kda_fwd_infer_triton=candidate_fn),
    )
    monkeypatch.setattr(
        chunk_kda,
        "_fallback_module",
        lambda: SimpleNamespace(chunk_kda_with_safe_gate=fallback_fn),
    )

    chunk_kda.chunk_kda_with_safe_gate(
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        g_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    candidate_fn.assert_called_once()
    fallback_fn.assert_not_called()
    assert candidate_fn.call_args.kwargs["cu_seqlens"] is cu_seqlens
    assert candidate_fn.call_args.kwargs["chunk_indices"] is chunk_indices
    assert candidate_fn.call_args.kwargs["use_beta_sigmoid_in_kernel"] is False


def test_unsupported_prefill_uses_flaggems_fallback(monkeypatch):
    inputs = make_inputs(tokens=32)
    q, k, v, raw_g, beta, A_log, g_bias, initial_state, cu_seqlens = inputs
    candidate_fn = Mock()
    fallback_fn = Mock(return_value=(torch.empty_like(q), initial_state))
    monkeypatch.setattr(chunk_kda, "_get_prefill_metadata", lambda: None)
    monkeypatch.setattr(
        chunk_kda,
        "_candidate_module",
        lambda: SimpleNamespace(chunk_kda_fwd_infer_triton=candidate_fn),
    )
    monkeypatch.setattr(
        chunk_kda,
        "_fallback_module",
        lambda: SimpleNamespace(chunk_kda_with_safe_gate=fallback_fn),
    )

    chunk_kda.chunk_kda_with_safe_gate(
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        g_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    candidate_fn.assert_not_called()
    fallback_fn.assert_called_once()
