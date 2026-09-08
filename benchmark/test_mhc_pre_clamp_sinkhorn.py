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

from flaggems_vllm.ops.mhc.mhc_pre_clamp_sinkhorn import (
    mhc_pre_clamp_sinkhorn,
)
from . import base


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
    return post_mix.to(dtype).unsqueeze(-1), comb_mix.to(dtype), layer_input.to(dtype)


class MHCPreClampSinkhornBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, hidden_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1280),
            (7, 1280),
            (256, 1280),
            (512, 2560),
            (1024, 3584),
            (4096, 3584),
        ]

    def get_input_iter(self, dtype):
        for num_tokens, hidden_size in self.shapes:
            torch.manual_seed(42)
            residual = torch.randn(
                (num_tokens, 4, hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            fn = (
                torch.randn(
                    (24, 4 * hidden_size), dtype=torch.float32, device=self.device
                )
                * 1e-3
            )
            hc_scale = torch.randn((3,), dtype=torch.float32, device=self.device)
            hc_base = torch.randn((24,), dtype=torch.float32, device=self.device)
            yield residual, fn, hc_scale, hc_base, 1e-6, -30.0, 30.0, 2.0, 20


@pytest.mark.mhc_pre
def test_mhc_pre_clamp_sinkhorn_benchmark():
    bench = MHCPreClampSinkhornBenchmark(
        op_name="mhc_pre_clamp_sinkhorn",
        torch_op=_reference,
        gems_op=mhc_pre_clamp_sinkhorn,
        dtypes=[torch.bfloat16],
    )
    bench.run()
