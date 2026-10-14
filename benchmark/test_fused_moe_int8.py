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

import inspect
import json
import statistics

import pytest
import torch
import triton
from triton import language as tl

import flaggems_vllm

from . import base

# Per-vendor native compute precision: (fp8, int8), per vendor documentation.
# Vendors left unlisted are skipped rather than assumed to work.
_OFFICIAL_PRECISION = {
    "nvidia": (True, True),
    "mthreads": (True, True),
    "hygon": (False, True),
    "metax": (False, True),
    "thead": (False, True),
    "ascend": (False, True),
}
_SUPPORTS_FP8, _SUPPORTS_INT8 = _OFFICIAL_PRECISION.get(
    flaggems_vllm.vendor_name, (False, False)
)

# Vendor has native INT8, but the kernels this benchmark reaches do not run
# there yet (a kernel-side gap, not a hardware limitation).
_INT8_UNSUPPORTED = {
    "mthreads",  # MUSA Triton has no tl.extra.cuda namespace
}

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


def _supports_keyword(op, keyword):
    try:
        parameters = inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


# vLLM versions differ: newer ones require ``inplace`` (no default), older ones
# do not accept the keyword at all.
VLLM_FUSED_MOE_SUPPORTS_INPLACE = HAS_VLLM_FUSED_MOE and _supports_keyword(
    vllm_fused_experts_impl, "inplace"
)


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


class FusedMoEINT8Benchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl with INT8 W8A8 quantization.

    Weights are pre-quantized to INT8 with per-channel (per output-dim) scales.
    Activations are dynamically quantized per-token inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int8_input_fn(config, cur_dtype)

    def _int8_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights one expert at a time to avoid OOM on large E.
        w1_int8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int8[e] = to_int8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )
            w2_int8[e] = to_int8(
                torch.randn(
                    hidden_size,
                    intermediate_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )

        # Synthetic per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts,
                intermediate_size * 2,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int8,
            w2_int8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call vllm fused_experts_impl with INT8."""
    kwargs = {"inplace": False} if VLLM_FUSED_MOE_SUPPORTS_INPLACE else {}
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        **kwargs,
    )


def _gems_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT8."""
    return flaggems_vllm.ops_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    not (HAS_VLLM_FUSED_MOE and _SUPPORTS_INT8)
    or flaggems_vllm.vendor_name in _INT8_UNSUPPORTED,
    reason="vLLM not installed, or no native INT8 support / known unsupported "
    "on this vendor (per vendor documentation)",
)
def test_fused_experts_impl_int8():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with INT8 W8A8 quantization.
    """
    bench = FusedMoEINT8Benchmark(
        op_name="fused_experts_impl",
        torch_op=_vllm_fused_moe_int8_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int8_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="810E benchmark")
@pytest.mark.parametrize("tokens", [32, 64, 65, 1024, 8192])
def test_fused_moe_int8_thead_gemm1(tokens):
    from flaggems_vllm.ops.fused_moe import invoke_fused_moe_triton_kernel as baseline
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        _EXPECTED_CONFIGS,
        _supports_gemm1_tile,
    )
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        invoke_fused_moe_triton_kernel as candidate,
    )
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        moe_align_block_size_triton,
    )

    if not _supports_gemm1_tile(torch.cuda.current_device()):
        pytest.skip("Requires the measured 810E capability")
    torch.manual_seed(42)
    a = torch.randint(-128, 128, (tokens, 6144), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 128, (128, 768, 6144), dtype=torch.int8, device="cuda")
    a_scale = torch.full((tokens, 1), 0.03125, device="cuda")
    b_scale = torch.full((128, 768, 1), 0.015625, device="cuda")
    out = torch.empty((tokens, 4, 768), dtype=torch.bfloat16, device="cuda")
    ids = (torch.arange(tokens * 4, dtype=torch.int32, device="cuda") % 128).reshape(
        tokens, 4
    )
    config = dict(_EXPECTED_CONFIGS.get(tokens, _EXPECTED_CONFIGS[64]))
    bm = config["BLOCK_SIZE_M"]
    size = min(ids.numel() * bm, ids.numel() + 128 * (bm - 1))
    sorted_ids = torch.empty(size, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(size, bm), dtype=torch.int32, device="cuda")
    padded = torch.empty(1, dtype=torch.int32, device="cuda")
    moe_align_block_size_triton(ids, 128, bm, sorted_ids, expert_ids, padded)

    def launch(op, cfg):
        op(
            a,
            b,
            out,
            a_scale,
            b_scale,
            None,
            sorted_ids,
            expert_ids,
            padded,
            False,
            4,
            cfg,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    baseline_config = dict(config)
    launch(baseline, baseline_config)
    launch(candidate, config)
    torch.cuda.synchronize()
    timings = {"baseline": [], "candidate": []}
    for trial in range(5):
        order = (
            ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        )
        for name in order:
            op, cfg = (
                (baseline, baseline_config)
                if name == "baseline"
                else (candidate, config)
            )
            timings[name].append(
                triton.testing.do_bench_cudagraph(lambda: launch(op, cfg), rep=100)
            )
    before = statistics.median(timings["baseline"])
    after = statistics.median(timings["candidate"])
    print(
        json.dumps(
            {
                "tokens": tokens,
                "scope": "GEMM1 only",
                "unit": "ms",
                "baseline_ms": before,
                "candidate_ms": after,
                "speedup": before / after,
                "paired_timings": timings,
            }
        )
    )
