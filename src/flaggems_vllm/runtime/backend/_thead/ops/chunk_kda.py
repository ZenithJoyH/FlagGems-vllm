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

"""Shape-gated long-prefill KDA routing for T-Head PPU."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from vllm.forward_context import get_forward_context


_MIN_MAX_QUERY_LEN = 512


def _candidate_module():
    return import_module("flaggems_vllm.ops.FLA.chunk_kda")


def _fallback_module():
    return import_module(
        "flaggems_vllm.runtime.backend._thead.ops.fused_recurrent_kda"
    )


def is_available() -> bool:
    try:
        candidate = _candidate_module()
        fallback = _fallback_module()
        return callable(candidate.chunk_kda_fwd_infer_triton) and callable(
            fallback.chunk_kda_with_safe_gate
        )
    except (AttributeError, ImportError, OSError):
        return False


def _get_prefill_metadata() -> Any | None:
    try:
        attn_metadata = get_forward_context().attn_metadata
    except AssertionError:
        return None
    if not isinstance(attn_metadata, dict):
        return None
    for metadata in attn_metadata.values():
        if (
            getattr(metadata, "fl_kda_prefill_max_query_len", 0)
            >= _MIN_MAX_QUERY_LEN
            and getattr(metadata, "num_spec_decodes", 0) == 0
            and getattr(metadata, "fl_kda_cu_seqlens_long", None) is not None
            and getattr(metadata, "fl_kda_chunk_indices_16", None) is not None
        ):
            return metadata
    return None


def _supported_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    return (
        q.ndim == 4
        and q.shape[0] == 1
        and q.shape[2:] == (4, 128)
        and k.shape == q.shape
        and v.shape == q.shape
        and raw_g.shape == q.shape
        and beta.shape == q.shape[:-1]
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and raw_g.dtype == torch.bfloat16
        and beta.dtype == torch.float32
        and A_log.dtype == torch.float32
        and g_bias is not None
        and g_bias.dtype == torch.float32
        and initial_state is not None
        and initial_state.dtype == torch.float32
        and initial_state.ndim == 4
        and initial_state.shape[1:] == (4, 128, 128)
        and output_final_state
        and use_qk_l2norm_in_kernel
        and cu_seqlens is not None
    )


def chunk_kda_with_safe_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float = -5.0,
):
    metadata = _get_prefill_metadata()
    if (
        metadata is not None
        and q.shape[1] == getattr(metadata, "fl_kda_total_tokens", -1)
        and initial_state is not None
        and initial_state.shape[0]
        == getattr(metadata, "fl_kda_num_sequences", -1)
        and _supported_shape(
            q,
            k,
            v,
            raw_g,
            beta,
            A_log,
            g_bias,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
        )
    ):
        return _candidate_module().chunk_kda_fwd_infer_triton(
            q=q,
            k=k,
            v=v,
            g=raw_g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=False,
            cu_seqlens=metadata.fl_kda_cu_seqlens_long,
            chunk_indices=metadata.fl_kda_chunk_indices_16,
            chunk_size=16,
            safe_gate=True,
            lower_bound=lower_bound,
            A_log=A_log.reshape(-1),
            dt_bias=g_bias,
        )
    return _fallback_module().chunk_kda_with_safe_gate(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        A_log=A_log,
        g_bias=g_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )


__all__ = ["chunk_kda_with_safe_gate", "is_available"]
