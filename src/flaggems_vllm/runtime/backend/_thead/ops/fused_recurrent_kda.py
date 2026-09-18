# SPDX-License-Identifier: Apache-2.0
"""Graph-safe vector-gated recurrent KDA decode kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_kda_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    state_indices_ptr,
    output_ptr,
    scale: tl.constexpr,
    num_states: tl.constexpr,
    heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_g_k: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_state_n: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_index_t: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_v: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    value_block = tl.program_id(2)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offsets_k < key_dim
    mask_v = offsets_v < value_dim

    state_id = tl.load(state_indices_ptr + token * stride_index_t)
    valid_state = (state_id >= 0) & (state_id < num_states)

    q_offsets = token * stride_q_t + head * stride_q_h + offsets_k * stride_q_k
    k_offsets = token * stride_k_t + head * stride_k_h + offsets_k * stride_k_k
    q = tl.load(q_ptr + q_offsets, mask=mask_k, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    gate_offsets = token * stride_g_t + head * stride_g_h + offsets_k * stride_g_k
    decay = tl.exp(tl.load(g_ptr + gate_offsets, mask=mask_k, other=0.0).to(tl.float32))
    state_offsets = (
        state_id * stride_state_n
        + head * stride_state_h
        + offsets_k[:, None] * stride_state_k
        + offsets_v[None, :] * stride_state_v
    )
    state_mask = valid_state & mask_k[:, None] & mask_v[None, :]
    state = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )
    state *= decay[:, None]

    v_offsets = token * stride_v_t + head * stride_v_h + offsets_v * stride_v_v
    value = tl.load(v_ptr + v_offsets, mask=mask_v, other=0.0).to(tl.float32)
    residual = value - tl.sum(k[:, None] * state, axis=0)
    beta = tl.load(beta_ptr + token * stride_beta_t + head * stride_beta_h).to(
        tl.float32
    )
    state += (beta * k)[:, None] * residual[None, :]

    output = tl.sum((q * scale)[:, None] * state, axis=0)
    output_offsets = (
        token * stride_output_t + head * stride_output_h + offsets_v * stride_output_v
    )
    tl.store(output_ptr + output_offsets, output, mask=valid_state & mask_v)
    tl.store(output_ptr + output_offsets, 0.0, mask=(~valid_state) & mask_v)
    tl.store(state_ptr + state_offsets, state, mask=state_mask)


@triton.jit
def _chunk_kda_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    beta_ptr,
    a_log_ptr,
    g_bias_ptr,
    cu_seqlens_ptr,
    state_ptr,
    output_ptr,
    scale: tl.constexpr,
    lower_bound: tl.constexpr,
    heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_k_t: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_v_t: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_v: tl.constexpr,
    stride_g_t: tl.constexpr,
    stride_g_h: tl.constexpr,
    stride_g_k: tl.constexpr,
    stride_beta_t: tl.constexpr,
    stride_beta_h: tl.constexpr,
    stride_state_n: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_v: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE_QK: tl.constexpr,
):
    sequence = tl.program_id(0)
    head = tl.program_id(1)
    value_block = tl.program_id(2)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offsets_k < key_dim
    mask_v = offsets_v < value_dim
    state_offsets = (
        sequence * stride_state_n
        + head * stride_state_h
        + offsets_k[:, None] * stride_state_k
        + offsets_v[None, :] * stride_state_v
    )
    state_mask = mask_k[:, None] & mask_v[None, :]
    state = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )
    a = tl.exp(tl.load(a_log_ptr + head).to(tl.float32))
    if HAS_BIAS:
        gate_bias = tl.load(
            g_bias_ptr + head * key_dim + offsets_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
    else:
        gate_bias = tl.zeros((BLOCK_K,), dtype=tl.float32)

    token = tl.load(cu_seqlens_ptr + sequence)
    end = tl.load(cu_seqlens_ptr + sequence + 1)
    while token < end:
        q_offsets = token * stride_q_t + head * stride_q_h + offsets_k * stride_q_k
        k_offsets = token * stride_k_t + head * stride_k_h + offsets_k * stride_k_k
        q = tl.load(q_ptr + q_offsets, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + k_offsets, mask=mask_k, other=0.0).to(tl.float32)
        if NORMALIZE_QK:
            q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
            k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

        gate_offsets = (
            token * stride_g_t + head * stride_g_h + offsets_k * stride_g_k
        )
        raw_gate = tl.load(raw_g_ptr + gate_offsets, mask=mask_k, other=0.0).to(
            tl.float32
        )
        gate = lower_bound * tl.sigmoid(a * (raw_gate + gate_bias))
        state *= tl.exp(gate)[:, None]

        v_offsets = token * stride_v_t + head * stride_v_h + offsets_v * stride_v_v
        value = tl.load(v_ptr + v_offsets, mask=mask_v, other=0.0).to(tl.float32)
        residual = value - tl.sum(k[:, None] * state, axis=0)
        beta = tl.load(beta_ptr + token * stride_beta_t + head * stride_beta_h).to(
            tl.float32
        )
        state += (beta * k)[:, None] * residual[None, :]
        output = tl.sum((q * scale)[:, None] * state, axis=0)
        output_offsets = (
            token * stride_output_t + head * stride_output_h + offsets_v * stride_output_v
        )
        tl.store(output_ptr + output_offsets, output, mask=mask_v)
        token += 1

    tl.store(state_ptr + state_offsets, state, mask=state_mask)


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute regular decode with device-side request-to-state mapping.

    Every packed token must represent one regular decode request. This covers
    vLLM's one-token decode batch, including capture size one and eager fallback
    for multiple concurrent requests. Prefill remains on the separate chunk KDA
    path owned by the model integration.
    """

    del kwargs
    if beta is None or initial_state is None:
        raise ValueError("recurrent KDA requires beta and initial_state")
    if cu_seqlens is None or ssm_state_indices is None:
        raise ValueError(
            "recurrent KDA requires device sequence and state metadata"
        )
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(
            f"Expected packed q [1, tokens, heads, key_dim], got {tuple(q.shape)}"
        )
    if k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "q/k/v must have matching packed batch, token, and head dimensions"
        )
    if g.shape != q.shape:
        raise ValueError(
            f"Vector KDA gate must match q shape; got g={tuple(g.shape)}, q={tuple(q.shape)}"
        )
    if beta.ndim != 3 or beta.shape != q.shape[:3]:
        raise ValueError(
            f"KDA beta must have shape {tuple(q.shape[:3])}, got {tuple(beta.shape)}"
        )
    if initial_state.ndim != 4:
        raise ValueError(
            "KDA state must have shape [states, heads, key_dim, value_dim]"
        )
    tokens, heads, key_dim = q.shape[1:]
    value_dim = v.shape[-1]
    if (heads, key_dim, value_dim) != (4, 128, 128):
        raise ValueError(
            "recurrent KDA currently requires H=4, K=128, V=128; "
            f"got H={heads}, K={key_dim}, V={value_dim}"
        )
    if initial_state.shape[1:] != (heads, key_dim, value_dim):
        raise ValueError("KDA state dimensions do not match q/v")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() - 1 != tokens:
        raise NotImplementedError(
            "recurrent KDA supports one regular decode token per sequence"
        )
    if ssm_state_indices.ndim == 1:
        if ssm_state_indices.shape[0] < tokens:
            raise ValueError("ssm_state_indices must provide one state per token")
        stride_index_t = ssm_state_indices.stride(0)
    elif ssm_state_indices.ndim == 2:
        if ssm_state_indices.shape[0] < tokens:
            raise ValueError("ssm_state_indices must provide one state row per token")
        stride_index_t = ssm_state_indices.stride(0)
    else:
        raise ValueError("ssm_state_indices must have rank 1 or 2")
    if not inplace_final_state:
        raise NotImplementedError("recurrent KDA requires in-place final state")
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError("recurrent KDA requires Q/K L2 normalization")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("recurrent KDA requires BF16 q/k/v")
    if g.dtype != torch.float32 or initial_state.dtype != torch.float32:
        raise ValueError(
            "recurrent KDA requires FP32 vector gate and recurrent state"
        )

    if scale is None:
        scale = key_dim**-0.5
    output = torch.empty_like(v)
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32
    grid = (tokens, heads, triton.cdiv(value_dim, block_v))
    _recurrent_kda_decode_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        ssm_state_indices,
        output,
        scale=float(scale),
        num_states=initial_state.shape[0],
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_k=q.stride(3),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        stride_v_t=v.stride(1),
        stride_v_h=v.stride(2),
        stride_v_v=v.stride(3),
        stride_g_t=g.stride(1),
        stride_g_h=g.stride(2),
        stride_g_k=g.stride(3),
        stride_beta_t=beta.stride(1),
        stride_beta_h=beta.stride(2),
        stride_state_n=initial_state.stride(0),
        stride_state_h=initial_state.stride(1),
        stride_state_k=initial_state.stride(2),
        stride_state_v=initial_state.stride(3),
        stride_index_t=stride_index_t,
        stride_output_t=output.stride(1),
        stride_output_h=output.stride(2),
        stride_output_v=output.stride(3),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=1,
    )
    return output, initial_state


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
    """Execute variable-length vector-gated KDA with a fused bounded gate."""

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(
            f"Expected packed q [1, tokens, heads, key_dim], got {tuple(q.shape)}"
        )
    if k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "q/k/v must have matching packed batch, token, and head dimensions"
        )
    if raw_g.shape != q.shape:
        raise ValueError(
            f"raw_g must match q shape; got raw_g={tuple(raw_g.shape)}, q={tuple(q.shape)}"
        )
    if beta.ndim != 3 or beta.shape != q.shape[:3]:
        raise ValueError(
            f"KDA beta must have shape {tuple(q.shape[:3])}, got {tuple(beta.shape)}"
        )
    if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    _, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    sequences = cu_seqlens.numel() - 1
    if (heads, key_dim, value_dim) != (4, 128, 128):
        raise ValueError(
            "chunk KDA currently requires H=4, K=128, V=128; "
            f"got H={heads}, K={key_dim}, V={value_dim}"
        )
    if A_log.numel() != heads:
        raise ValueError("A_log must provide one value per KDA head")
    if g_bias is not None and g_bias.numel() != heads * key_dim:
        raise ValueError("g_bias must provide one value per KDA gate element")
    if initial_state is None:
        final_state = torch.zeros(
            sequences,
            heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
    else:
        if initial_state.shape != (sequences, heads, key_dim, value_dim):
            raise ValueError(
                "initial_state must have shape "
                f"{(sequences, heads, key_dim, value_dim)}, got {tuple(initial_state.shape)}"
            )
        final_state = initial_state.clone()
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("chunk KDA requires BF16 q/k/v")
    if final_state.dtype != torch.float32:
        raise ValueError("chunk KDA recurrent state must be FP32")
    if scale is None:
        scale = key_dim**-0.5

    output = torch.empty_like(v)
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32
    grid = (sequences, heads, triton.cdiv(value_dim, block_v))
    _chunk_kda_varlen_kernel[grid](
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        g_bias if g_bias is not None else A_log,
        cu_seqlens,
        final_state,
        output,
        scale=float(scale),
        lower_bound=float(lower_bound),
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_q_k=q.stride(3),
        stride_k_t=k.stride(1),
        stride_k_h=k.stride(2),
        stride_k_k=k.stride(3),
        stride_v_t=v.stride(1),
        stride_v_h=v.stride(2),
        stride_v_v=v.stride(3),
        stride_g_t=raw_g.stride(1),
        stride_g_h=raw_g.stride(2),
        stride_g_k=raw_g.stride(3),
        stride_beta_t=beta.stride(1),
        stride_beta_h=beta.stride(2),
        stride_state_n=final_state.stride(0),
        stride_state_h=final_state.stride(1),
        stride_state_k=final_state.stride(2),
        stride_state_v=final_state.stride(3),
        stride_output_t=output.stride(1),
        stride_output_h=output.stride(2),
        stride_output_v=output.stride(3),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        HAS_BIAS=g_bias is not None,
        NORMALIZE_QK=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=1,
    )
    return output, final_state if output_final_state else None


__all__ = ["chunk_kda_with_safe_gate", "fused_recurrent_kda"]
