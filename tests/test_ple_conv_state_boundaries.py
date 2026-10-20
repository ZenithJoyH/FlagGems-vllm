# SPDX-License-Identifier: Apache-2.0
"""PLE in-place state updates must remain exact across warps and graph replays."""

import pytest
import torch
from test_ple_conv import inputs

from flaggems_vllm.ops.qwen4.ple_conv import ple_short_conv_

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
MODES = ["decode", "prefill", "spec"]
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton accelerator required"
)


def oracle(x, initial, weight, ids, kw):
    output, updated = torch.zeros_like(x), initial.clone()
    width, dilation = weight.shape[1], kw["dilation"]
    history = (width - 1) * dilation
    starts = kw.get("query_start_loc")
    starts = starts.tolist() if starts is not None else list(range(len(ids) + 1))
    flags, accepted = kw.get("has_initial_state"), kw.get("accepted_tokens")
    for req, row in enumerate(ids):
        begin, end = starts[req : req + 2]
        if (
            row == kw.get("null_block_id", -1)
            or not 0 <= row < initial.shape[0]
            or begin == end
        ):
            continue
        rollback = 0 if accepted is None else int(accepted[req]) - 1
        has = (flags is None or bool(flags[req])) and (
            accepted is None or 0 <= rollback < kw["max_query_len"]
        )
        for t in range(end - begin):
            acc = torch.zeros(x.shape[1], dtype=torch.float32)
            for tap in range(width):
                pos = t - (width - 1 - tap) * dilation
                value = torch.zeros_like(acc)
                if pos >= 0:
                    value = x[begin + pos].float()
                elif has and rollback + history + pos >= 0:
                    value = (
                        initial[row, :, rollback + history + pos].to(x.dtype).float()
                    )
                acc += value * weight[:, tap].float()
            rounded = acc.to(x.dtype).float()
            output[begin + t] = (rounded * torch.sigmoid(rounded)).to(x.dtype)
        if not history:
            continue
        count = history if accepted is None else history + end - begin - 1
        for s in range(count):
            pos = s + end - begin - history if accepted is None else s + 1 - history
            value = torch.zeros(x.shape[1], dtype=x.dtype)
            if 0 <= pos < end - begin:
                value = x[begin + pos]
            elif pos < 0 and has and rollback + history + pos >= 0:
                value = initial[row, :, rollback + history + pos].to(x.dtype)
            updated[row, :, s] = value.to(initial.dtype)
    return output, updated


def state_on_device(initial, layout):
    rows, channels, capacity = initial.shape
    if layout == "channel-contiguous":
        state = torch.empty(
            (rows, capacity, channels), device="cuda", dtype=initial.dtype
        ).transpose(1, 2)
    else:
        state = torch.empty_like(
            initial, device="cuda", memory_format=torch.contiguous_format
        )
    state.copy_(initial)
    return state


def execute(x_cpu, initial_cpu, weight_cpu, ids_cpu, kw_cpu, layout, eager_repeats):
    x, weight = x_cpu.cuda(), weight_cpu.cuda()
    state = state_on_device(initial_cpu, layout)
    pristine = state.clone()
    indices = torch.tensor(
        ids_cpu, device="cuda", dtype=kw_cpu.pop("index_dtype", torch.int32)
    )
    kw = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in kw_cpu.items()}
    output = torch.empty_like(x)
    tol = {torch.bfloat16: 2e-2, torch.float16: 3e-3, torch.float32: 2e-5}[x.dtype]

    def reset(phase):
        state.copy_(pristine)
        x.copy_(x_cpu * ([1, 0.5, -0.5, 1][phase]))
        valid = sorted({r for r in ids_cpu if r > 0 and r < initial_cpu.shape[0]})
        mapping = {r: valid[(i + phase) % len(valid)] for i, r in enumerate(valid)}
        ids = [mapping.get(r, r) for r in ids_cpu]
        indices.copy_(torch.tensor(ids, device="cuda", dtype=indices.dtype))
        flags = kw_cpu.get("has_initial_state")
        if flags is not None:
            kw["has_initial_state"].copy_(~flags if phase % 2 else flags)
        output.fill_(float("nan"))
        return ids

    def invoke():
        ple_short_conv_(x, output, state, weight, indices, **kw)

    def verify(ids):
        torch.cuda.synchronize()
        current_kw = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()
        }
        expected_output, expected_state = oracle(
            x.cpu(), initial_cpu, weight_cpu, ids, current_kw
        )
        torch.testing.assert_close(output.cpu(), expected_output, atol=tol, rtol=tol)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)

    for _ in range(eager_repeats):
        ids = reset(0)
        invoke()
        verify(ids)
    reset(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for phase in (1, 2, 3):
        ids = reset(phase)
        graph.replay()
        verify(ids)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("dilation", [1, 3])
def test_all_widths_dilations_dtype_modes(dtype, mode, width, dilation):
    channels = [1, 33, 129][width % 3]
    history = (width - 1) * dilation
    capacity = max(1, history + (3 if mode == "spec" else 0))
    lengths = (
        [1] * 8
        if mode == "decode"
        else ([1, 3, 0, 2, 1, 3, 1, 2] if mode == "spec" else [0, 1, 9, 2, 3, 0, 1, 4])
    )
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    tokens = int(starts[-1]) + (0 if mode == "decode" else 2)
    x = (
        ((torch.arange(tokens * channels) % 17 - 8) / 16)
        .reshape(tokens, channels)
        .to(dtype)
    )
    weight = torch.full((channels, width), 0.25, dtype=dtype)
    state = (
        ((torch.arange(8 * channels * capacity) % 13 - 6) / 8)
        .reshape(8, channels, capacity)
        .to(dtype)
    )
    kw = {
        "dilation": dilation,
        "null_block_id": 0,
        "has_initial_state": torch.tensor(
            [True, False, True, True, False, True, True, False]
        ),
    }
    if mode != "decode":
        kw["query_start_loc"] = starts
    if mode == "spec":
        kw.update(
            accepted_tokens=torch.tensor([1, 2, 0, 4, 5, -1, 4, 1], dtype=torch.int32),
            max_query_len=4,
        )
    execute(
        x,
        state,
        weight,
        [1, 2, 3, 0, -1, 8, 4, 5],
        kw,
        "channel-contiguous" if width % 2 else "contiguous",
        1,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["contiguous", "channel-contiguous"])
def test_production_channels_history20_and_graph(dtype, mode, index_dtype, layout):
    # Reuse the maintained synthetic fixture; no saved model inputs are needed.
    torch.manual_seed(73049)
    args, kw = inputs(mode, torch.bfloat16, channels=10240)
    x, _, old, weight, indices = args
    old = old.cpu().to(dtype)
    state = torch.full((old.shape[0], 10240, 20), -17, dtype=dtype)
    state[:, :, : old.shape[2]] = old
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
    kwargs["index_dtype"] = index_dtype
    execute(
        x.cpu().to(dtype),
        state,
        weight.cpu().to(dtype),
        indices.cpu().tolist(),
        kwargs,
        layout,
        3,
    )
