# SPDX-License-Identifier: Apache-2.0
"""PLE cache row arithmetic must stay 64-bit across graph index changes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flaggems_vllm.ops.qwen4.ple_conv import ple_short_conv_  # noqa: E402


def _reference(x, weight, histories, indices, lengths, initial, accepted, rows):
    channels, width = weight.shape
    state_len = (width - 1) * 3
    out = torch.zeros_like(x)
    expected = {}
    begin = 0
    for q, (row, length) in enumerate(zip(indices, lengths)):
        if row <= 0 or row >= rows or length == 0:
            begin += length
            continue
        history = histories[row]
        rollback = 0 if accepted is None else accepted[q] - 1
        has_history = initial[q] and (accepted is None or 0 <= rollback < 3)
        for j in range(length):
            acc = torch.zeros(channels)
            for tap in range(width):
                pos = j - (width - 1 - tap) * 3
                value = torch.zeros(channels)
                if pos >= 0:
                    value = x[begin + pos].float()
                elif has_history and rollback + state_len + pos >= 0:
                    value = history[:, rollback + state_len + pos].float()
                acc += value * weight[:, tap].float()
            rounded = acc.to(x.dtype).float()
            out[begin + j] = (rounded * torch.sigmoid(rounded)).to(x.dtype)
        updated = history.clone()
        count = state_len if accepted is None else state_len + length - 1
        for s in range(count):
            pos = s + length - state_len if accepted is None else s + 1 - state_len
            value = torch.zeros(channels, dtype=x.dtype)
            if 0 <= pos < length:
                value = x[begin + pos]
            elif pos < 0 and has_history and rollback + state_len + pos >= 0:
                value = history[:, rollback + state_len + pos]
            updated[:, s] = value
        expected[row] = updated
        begin += length
    return out, expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_large_state_rows_and_changed_index_graph(dtype, mode):
    if not torch.cuda.is_available():
        pytest.skip("requires a Triton accelerator")
    rows, channels, stride, guard = 12003, 129, 204800, 128
    capacity = 11 if mode == "spec" else 9
    n = (rows - 1) * stride + channels * capacity + 2 * guard
    required = n * torch.empty((), dtype=dtype).element_size()
    if torch.cuda.mem_get_info()[0] < required + 1024**3:
        pytest.skip("large-row regression needs space for its real strided allocation")
    backing = torch.full((n,), -17.0, device="cuda", dtype=dtype)
    state = torch.as_strided(
        backing, (rows, channels, capacity), (stride, 1, channels), guard
    )
    lengths = [1] * 8 if mode == "decode" else [1, 3, 0, 2, 1, 3, 1, 1]
    starts_cpu = [0]
    for length in lengths:
        starts_cpu.append(starts_cpu[-1] + length)
    x_cpu = (
        ((torch.arange(starts_cpu[-1] * channels) % 17 - 8) / 16)
        .reshape(-1, channels)
        .to(dtype)
    )
    weight_cpu = torch.full((channels, 4), 0.25, dtype=dtype)
    x, weight = x_cpu.cuda(), weight_cpu.cuda()
    out = torch.empty_like(x)
    initial_cpu = [True, True, True, True, True, True, False, True]
    initial = torch.tensor(initial_cpu, device="cuda", dtype=torch.bool)
    accepted_cpu = [1, 2, 1, 1, 1, 1, 3, 2] if mode == "spec" else None
    accepted = (
        torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
        if accepted_cpu is not None
        else None
    )
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    # int32 is intentional: it is the production index dtype, not pointer width.
    indices = torch.empty(8, device="cuda", dtype=torch.int32)
    batches = [
        [10485, 10486, 11242, 0, -1, rows, 12000, 7],
        [12000, 11242, 10486, 0, -1, rows, 10485, 8],
        [11242, 12000, 10485, 0, -1, rows, 10486, 9],
    ]
    histories = {
        row: ((torch.arange(channels * capacity) % 11 - 5) / 8)
        .reshape(channels, capacity)
        .to(dtype)
        for row in {r for batch in batches for r in batch if 0 < r < rows}
    }

    def reset(batch):
        backing.fill_(-17)
        for row, value in histories.items():
            state[row].copy_(value)
        indices.copy_(torch.tensor(batch, device="cuda", dtype=torch.int32))

    def invoke():
        ple_short_conv_(
            x,
            out,
            state,
            weight,
            indices,
            dilation=3,
            query_start_loc=None if mode == "decode" else starts,
            has_initial_state=initial,
            accepted_tokens=accepted,
            max_query_len=3 if mode == "spec" else 1,
            null_block_id=0,
        )

    def verify(batch):
        expected_out, updates = _reference(
            x_cpu,
            weight_cpu,
            histories,
            batch,
            lengths,
            initial_cpu,
            accepted_cpu,
            rows,
        )
        torch.testing.assert_close(out.cpu(), expected_out, atol=0.004, rtol=0.004)
        for row, old in histories.items():
            torch.testing.assert_close(
                state[row].cpu(), updates.get(row, old), atol=0, rtol=0
            )
        # NULL row and padding are untouched, including regions adjacent to writes.
        null_row = state[0].cpu()
        torch.testing.assert_close(
            null_row, torch.full_like(null_row, -17), atol=0, rtol=0
        )
        for row in histories:
            end = guard + row * stride + channels * capacity
            assert bool((backing[end : end + guard] == -17).all())
        assert bool((backing[:guard] == -17).all())
        assert bool((backing[-guard:] == -17).all())

    reset(batches[0])
    invoke()
    torch.cuda.synchronize()
    verify(batches[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for batch in batches:
        reset(batch)
        graph.replay()
        torch.cuda.synchronize()
        verify(batch)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_ple_empty_input_does_not_write_state(index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a Triton accelerator")
    state = torch.full((4, 17, 9), 3, device="cuda", dtype=torch.bfloat16)
    x = torch.empty((0, 17), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((17, 4), device="cuda", dtype=torch.bfloat16)
    ple_short_conv_(
        x,
        torch.empty_like(x),
        state,
        weight,
        torch.empty(0, device="cuda", dtype=index_dtype),
        dilation=3,
    )
    assert bool((state == 3).all())
