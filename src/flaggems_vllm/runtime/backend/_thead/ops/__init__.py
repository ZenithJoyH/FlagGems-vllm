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

from flaggems_vllm.runtime.backend._thead.ops.bf16_paged_mqa_logits_graph_safe import (
    bf16_paged_mqa_logits_graph_safe,
)
from flaggems_vllm.runtime.backend._thead.ops.causal_conv1d import (
    causal_conv1d_fn,
)
from flaggems_vllm.runtime.backend._thead.ops.causal_conv1d_update import (
    causal_conv1d_update,
)
from flaggems_vllm.runtime.backend._thead.ops.chunk_kda import (
    chunk_kda_with_safe_gate,
)
from flaggems_vllm.runtime.backend._thead.ops.cp_gather_indexer_k_bf16_cache import (
    cp_gather_indexer_k_bf16_cache,
)
from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
    fused_experts_impl,
    inplace_fused_experts,
    invoke_fused_moe_triton_kernel,
    outplace_fused_experts,
)
from flaggems_vllm.runtime.backend._thead.ops.fused_recurrent_kda import (
    fused_recurrent_kda,
)
from flaggems_vllm.runtime.backend._thead.ops.fused_safe_kda_gate import (
    fused_safe_kda_gate,
)
from flaggems_vllm.runtime.backend._thead.ops.indexer_pool import (
    append_tail_to_topk,
    expand_pools_to_tokens,
)
from flaggems_vllm.runtime.backend._thead.ops.kpool_compress import (
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from flaggems_vllm.runtime.backend._thead.ops.persistent_topk import persistent_topk
from flaggems_vllm.runtime.backend._thead.ops.prefill_tail import persist_prefill_tail

__all__ = [
    "append_tail_to_topk",
    "bf16_paged_mqa_logits_graph_safe",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_kda_with_safe_gate",
    "cp_gather_indexer_k_bf16_cache",
    "expand_pools_to_tokens",
    "fused_experts_impl",
    "fused_recurrent_kda",
    "fused_safe_kda_gate",
    "inplace_fused_experts",
    "invoke_fused_moe_triton_kernel",
    "kpool_compress_and_write_cache",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "outplace_fused_experts",
    "persist_prefill_tail",
    "persistent_topk",
]
