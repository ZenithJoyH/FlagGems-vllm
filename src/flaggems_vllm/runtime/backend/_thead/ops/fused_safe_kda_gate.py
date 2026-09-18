# SPDX-License-Identifier: Apache-2.0
"""Bounded KDA gate kernel without an inference-framework dependency."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in (32, 64, 128)
        for nw in (4, 8, 16, 32)
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def _safe_gate_kernel(
    g,
    A,
    y,
    g_bias,
    lower_bound: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT
    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=n_d < D, other=0.0).to(tl.float32)
        b_g += b_bias[None, :]
    b_y = lower_bound / (1.0 + tl.exp(-(b_a * b_g)))
    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_safe_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Compute ``lower_bound*sigmoid(exp(A_log)*(g+g_bias))`` in FP32."""
    orig_shape = g.shape[:-1]
    g = g.view(-1, g.shape[-1])
    tokens, hidden = g.shape
    heads = A_log.numel()
    assert heads * head_k_dim == hidden
    out = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(tokens, meta["BT"]), heads)

    _safe_gate_kernel[grid](
        g,
        A_log,
        out,
        g_bias,
        lower_bound,
        tokens,
        heads,
        head_k_dim,
        BD=triton.next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )
    return out.view(*orig_shape, heads, head_k_dim)


__all__ = ["fused_safe_kda_gate"]
