# SPDX-License-Identifier: Apache-2.0
"""CPU checks for Step 12 P2: the plan's per-pair slab rows computed in the
exchange-table kernel (push option exchange_slab_rows, off by default).

* plan_kernel.slab_rows_from_pairs (the kernel's body, run here as plain JAX
  on CPU) equals host.routing_pairs_from_tables' slab_row bit for bit, on
  every rank, for the real-text replay sets ($P2_REPLAY: "{bucket}" in the
  path, f32 [sets, ep * bucket, E] crafted logits, as the layer benchmark
  replays them) and on random routings with unrouted rows; the run bases
  come from the table kernel (Pallas TPU interpreter).
* Lowering (TPU7x target on CPU): the switch off lowers to the default text;
  on, the exchange-table kernel carries `_sr`; on the x_tables path it is
  ignored.

    PYTHONPATH=<tree>/python:<tree>:<this dir> python -m pytest -q <this file>
"""

import os
import sys
from functools import partial

if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=8"
        ).strip()

import numpy as np
import pytest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
pytest.importorskip("ml_dtypes")
pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402
from test_dsv41_push_k2_knobs import (  # noqa: E402,F401
    _lower_push,
    _names,
    _tuned_push_options,
    v2,
)

EP, G, K = 8, 48, 6
E_TOTAL = EP * G
LANES = 128


def _replay_ids(path, index):
    """The layer benchmark's replay selection (host_select, zero bias):
    [ep * bucket, K] ids, E_TOTAL on padding rows."""
    gating = np.array(np.load(path, mmap_mode="r")[index], np.float64)
    pad = ~np.isfinite(gating).any(axis=1)
    gating[pad] = 0.0
    scores = np.sqrt(np.logaddexp(0.0, gating))
    ids = np.argpartition(-scores, K, axis=1)[:, :K].astype(np.int64)
    ids[pad] = E_TOTAL
    return ids


def _random_ids(seed, t_local, unrouted=2):
    rng = np.random.default_rng(seed)
    w = rng.zipf(1.3, E_TOTAL).astype(np.float64)
    keys = rng.exponential(size=(EP * t_local, E_TOTAL)) / w[None, :]
    ids = np.argpartition(keys, K - 1, axis=1)[:, :K].astype(np.int64)
    ids[t_local - unrouted : t_local] = E_TOTAL  # rank 0's last rows
    return ids


def _check_rank_tables(v2, ids, t_local, me):
    host, plan = v2.host, v2.plan
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import host as ph

    T = EP * t_local
    n = t_local * K
    block = host.routing_block(t_local, K)
    stride = host.ragged_stride_bound(T, K, E_TOTAL, 256)
    pairs = ids.reshape(EP, n).astype(np.int32)
    counts = np.stack(
        [np.bincount(p[p < E_TOTAL], minlength=E_TOTAL) for p in pairs]
    ).astype(np.int32)
    seg = np.cumsum(counts[me]) - counts[me]
    out = plan.build_push_tables(
        jnp.asarray(counts),
        jnp.asarray(seg, jnp.int32),
        me,
        ep=EP,
        n_pre=2,
        tile_m=256,
        interpret=True,
        light_prepush=2,
    )
    recv_base_me, mirror_base, run_base = (jnp.asarray(a) for a in out[8])
    expert_blocks = jnp.asarray(pairs.reshape(T * K // block, block))
    nbl = n // block
    local_blocks = expert_blocks[me * nbl : (me + 1) * nbl]
    gathered = host.GatheredPairs(
        expert_blocks=expert_blocks,
        local_blocks=local_blocks,
        block_hist=host.pair_block_hist(local_blocks, E_TOTAL),
        rows_by_dest=jnp.asarray(counts),
        n_tokens=T,
        topk=K,
    )
    want = ph.routing_pairs_from_tables(
        jnp.int32(me),
        gathered,
        recv_base_me,
        mirror_base,
        run_base,
        e_total=E_TOTAL,
        ep=EP,
        t_local=t_local,
        block=block,
        shard_stride=stride,
    )[2]
    n_pr = -(-n // LANES)
    words = np.zeros((EP, n_pr * LANES), np.int32)
    words[:, :n] = pairs
    rb = np.zeros((EP, LANES), np.int32)
    rb[:, :G] = np.asarray(run_base)
    rows = plan.slab_rows_from_pairs(
        jnp.asarray(words.reshape(EP, n_pr, LANES)),
        jnp.asarray(rb),
        jnp.int32(me),
        g=G,
        n_pairs=n,
        off_base=host.off_shard_slab_row(stride, EP),
        stride=stride,
    )
    got = np.asarray(jnp.concatenate(rows, 0)).reshape(EP, n_pr * LANES)[:, :n]
    want = np.asarray(want).reshape(EP, n)
    np.testing.assert_array_equal(got, want)
    return int((want != want.max()).sum())


_REPLAY = os.environ.get("P2_REPLAY", "")
_BUCKETS = [int(b) for b in os.environ.get("P2_BUCKETS", "8,64").split(",")]


@pytest.mark.skipif(not _REPLAY, reason="P2_REPLAY (replay set path) not set")
@pytest.mark.parametrize("t_local", _BUCKETS)
def test_slab_rows_match_the_plan_on_replay_sets(v2, t_local):
    path = _REPLAY.replace("{bucket}", str(t_local))
    n_sets = np.load(path, mmap_mode="r").shape[0]
    for index in range(n_sets):
        ids = _replay_ids(path, index)
        on = [_check_rank_tables(v2, ids, t_local, me) for me in range(EP)]
        print(f"P2 replay {path}[{index}] rows {t_local}: 8/8 ranks equal, on-shard pairs {on}")


@pytest.mark.parametrize("t_local", [8, 16, 64])
@pytest.mark.parametrize("seed", [3, 11])
def test_slab_rows_match_the_plan_on_random_routing(v2, t_local, seed):
    ids = _random_ids(seed, t_local)
    for me in (0, 3, EP - 1):
        _check_rank_tables(v2, ids, t_local, me)


# --------------------------------------------------------------------------- #
# Lowering.
# --------------------------------------------------------------------------- #
def test_switch_off_and_x_path_lower_to_the_default_program(v2, monkeypatch):
    k32z = v2.host.WeightFormat.K32Z
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=16)
    # Step 12 I: on by default for a K32Z build at 8-256 rows per rank.
    assert v2.host.K32Z_PUSH_SLAB_ROWS
    base = lower(push_options=_tuned_push_options(8))
    on = lower(push_options={**_tuned_push_options(8), "exchange_slab_rows": True})
    assert on == base
    off = lower(push_options={**_tuned_push_options(8), "exchange_slab_rows": False})
    assert off != base
    names = _names(on, "moe_tc_routing_exchange")
    assert names and all("_sr" in n for n in names), names
    assert not any("_sr" in n for n in _names(off, "moe_tc_routing_exchange"))
    monkeypatch.setattr(v2.host, "K32Z_PUSH_SLAB_ROWS", 0)
    assert lower(push_options=_tuned_push_options(8)) == off
    monkeypatch.undo()
    x = {**_tuned_push_options(8), "x_tables": True}
    lower256 = partial(_lower_push, v2, weight_format=k32z, t_local=256)
    assert lower256(push_options={**x, "exchange_slab_rows": True}) == lower256(
        push_options=x
    )
