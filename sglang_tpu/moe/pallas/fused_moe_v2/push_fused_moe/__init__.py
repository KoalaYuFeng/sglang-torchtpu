# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/push_fused_moe/__init__.py
# Copyright 2026 Google LLC
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
"""Push dispatch for the fused EP MoE v2: `fused_ep_moe_v2(dispatch="push")`.

The fused EP MoE v2 kernel with the activation all-gather replaced by
point-to-point pushes of each shard's routed rows: a SparseCore kernel pushes
the first visit positions before the TensorCore kernel starts, and the
TensorCore kernel pushes the rest with in-kernel remote DMAs. See layer.py.
Callers go through `fused_ep_moe_v2`; `fused_ep_moe_push` is exported for the
package's own tests.
"""

from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe.layer import fused_ep_moe_push

# The keyword arguments `fused_ep_moe_push` takes beyond v2's operands, which
# `fused_ep_moe_v2(dispatch="push", push_options=...)` forwards. Everything
# else in its signature is a v2 parameter forwarded by name.
PUSH_OPTIONS = (
    "sc_prepush",
    "tc_exchange",
    "height_rungs_override",
    "tail_barrier",
    "onehot_slots",
    "hbm_staging",
    "dispatch_ahead",
    "matmul_scale_repeat",
    "exchange_row_gather",
    "bf16_sort_counts",
    "slim_prepush_barrier",
    "fused_dispatch_meta",
    "table_kernel",
    "exchange_tables",
    "kernel_plan_tables",
    "light_prepush",
    "single_barrier",
    "early_scale_scatter",
    "fuse_metadata",
    # Step 7 K2 (off by default, except the K32Z push path's per-bucket
    # defaults for defer_push, fold_select, x_tables and x_dedup when left
    # unset: layer.K32Z_PUSH_DEFAULTS; ret_last_off and commit_only are
    # benchmark only and give invalid output).
    "visit_order",
    "two_hop",
    "defer_push",
    "fold_select",
    "ret_last_off",
    "commit_only",
    # Step 7 K2 dispatch dedup (off by default; x_gather_off is benchmark
    # only and gives invalid output). See push_fused_moe.dedup.
    "x_tables",
    "x_dedup",
    "x_dedup_local",
    "x_owner_visit",
    "x_gather_off",
    "x_src_cap",
    # Step 7 O (off by default; see push_fused_moe.dedup.receiver_tables).
    "x_flat_owner",
    # Step 11 E1 (off by default): the return's runs ship from inside their
    # expert's tile loop on the two-hop path (push kernel `_es`).
    "early_ship",
    # Step 12 P2: the plan's per-pair slab rows from the exchange-table kernel
    # (`_sr`); unset takes host.K32Z_PUSH_SLAB_ROWS (K32Z at 8-256 rows: on).
    "exchange_slab_rows",
)

__all__ = ["PUSH_OPTIONS", "fused_ep_moe_push"]
