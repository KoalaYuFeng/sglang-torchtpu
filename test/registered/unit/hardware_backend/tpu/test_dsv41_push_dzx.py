# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Step 7 K2d per-rank dispatch dedup (DZX, off by default).

* The NumPy model (benchmark/tpu/dsv41_dzx_model.py, written from the design)
  replays a layer's data movement on random routings -- uniform, skewed with
  empty experts, hot; the lowest-id owner, the visit-order owner, own-rank
  dedup, an SRC cap small enough to overflow, tiles of several heights --
  and asserts: every tile copy partition covers the tile once, every true
  tile row holds the token the default layout puts there, every duplicate
  is copied from a landed owner row visited earlier (or a SparseCore
  position), each pair's run position indexes its own token, and whole runs
  wherever the dedup is off (SparseCore positions, overflowed ranks).
* Device pieces against the model, bitwise: the sender tables and the
  owned-first counting sort (dedup.sender_tables + host.local_sort_tokens,
  count and argsort paths), the table kernel with owned counts (Pallas TPU
  interpreter), the receiver's slab rows and SRC, the XLA visit order of the
  fallback owner rule.
* x_tables alone gives the default plan's arrival rows, mirror positions and
  slab rows bit for bit (routing_pairs_from_tables), and owned counts equal
  to the pair counts leave every default table unchanged.
* Lowering (TPU7x target on CPU): every x switch at its default lowers to
  the bundle's text; outside 256..1024 rows per rank the switches leave the
  program as it is; each switch set lowers to its own text with its name
  tokens; refusals; the breakdown harness's DZX tokens.

    PYTHONPATH=<tree>/python:<tree>:<this dir> python -m pytest -q <this file>
"""

import os
import sys
from functools import partial
from pathlib import Path

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


register_tpu_ci(est_time=300, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
pytest.importorskip("ml_dtypes")
pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from test_dsv41_push_k2_knobs import (  # noqa: E402,F401
    _lower_push,
    _names,
    _push_build_args,
    _tuned_push_options,
    bd,
    v2,
)

TREE = Path(__file__).resolve().parents[5]
EP, G, K = 8, 48, 6
E_TOTAL = EP * G

X_DEFAULTS = dict(
    x_tables=False,
    x_dedup=False,
    x_dedup_local=False,
    x_owner_visit=False,
    x_gather_off=False,
    x_src_cap=0,
)


@pytest.fixture(scope="module")
def model():
    path = str(TREE / "benchmark" / "tpu")
    if path not in sys.path:
        sys.path.insert(0, path)
    import dsv41_dzx_model

    return dsv41_dzx_model


@pytest.fixture(scope="module")
def dd(v2):
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import dedup

    assert dedup.X_IDX_SHIFT == 16 and v2.plan.X_SHIFT == 16
    assert v2.push_kernel.X_SHIFT == 16
    return dedup


@pytest.fixture(scope="module")
def ph(v2):
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import host

    return host


def _case(model, kind, t_local, *, seed=11, tile_m=64, **kw):
    empty = {"uniform": 0, "skew": 5, "hot": 3}[kind]
    ids = model.routing(kind, seed, ep=EP, g=G, t_local=t_local, empty=empty)
    return model.build(ids, g=G, n_pre=2, tile_m=tile_m, **kw)


# --------------------------------------------------------------------------- #
# The model: replay invariants.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["uniform", "skew", "hot"])
@pytest.mark.parametrize(
    "mode,tile_m",
    [("x", 64), ("x", 8), ("x_local", 64), ("x_visit", 64), ("x_cap", 64), ("D", 64)],
)
def test_model_replay(model, kind, mode, tile_m):
    case = _case(
        model,
        kind,
        64,
        tile_m=tile_m,
        dedup=mode != "D",
        local=mode == "x_local",
        owner_visit=mode == "x_visit",
        cap=24 if mode == "x_cap" else None,
    )
    stats = model.replay(case)
    r3, landed = case["r3"], case["landed"]
    assert (landed <= r3).all() and (landed >= 0).all()
    if mode == "D":
        assert (landed == r3).all() and stats["dup_copies"] == 0
        return
    # Whole runs where the dedup is off: SparseCore positions, overflowed
    # destinations, own-rank runs unless x_dedup_local.
    for d in range(EP):
        whole = case["xsc"][d] | case["overflow"][d]
        assert (landed[:, d][:, whole] == r3[:, d][:, whole]).all()
        if mode != "x_local":
            assert (landed[d, d] == r3[d, d]).all()
    gathered = sum(rk["n_dup"] for rk in case["ranks"])
    assert stats["dup_copies"] == gathered
    for d in range(EP):
        if case["overflow"][d]:
            assert case["ranks"][d]["n_dup"] == 0
        else:
            assert case["ranks"][d]["n_dup"] <= case["cap"]
    if mode == "x_cap":
        assert case["overflow"].any()
    else:
        assert gathered > 0
    # Every dup copy plus the landed rows account for every pair.
    assert landed.sum() + gathered == r3.sum()


def test_model_dedup_link_ratio_uniform_512(model):
    """Uniform top-6 of 384 over 8 ranks at 512 rows per rank: a token
    reaches a rank with probability 0.5537, so the dispatch moves ~0.74-0.75
    of D's rows per rank pair (whole runs at the two SparseCore positions
    add ~1 %), and ~0.17 of the arriving token rows are duplicate copies."""
    case = _case(model, "uniform", 512, tile_m=256)
    lc = model.link_counts(case)
    assert 0.72 < lc["ratio"] < 0.78, lc
    dma = model.dma_counts(case)
    assert dma["overflow_ranks"] == 0
    assert 500 < dma["dup_rows_mean"] < 900, dma


# --------------------------------------------------------------------------- #
# Device pieces against the model.
# --------------------------------------------------------------------------- #
def _vp(case, s):
    ids = case["ids"][s]
    flat = case["rank"].reshape(-1)
    return np.where(ids < E_TOTAL, flat[np.minimum(ids, E_TOTAL - 1)], 0)


@pytest.mark.parametrize("t_local", [64, 600], ids=["count_sort", "argsort"])
@pytest.mark.parametrize(
    "dedup,local,visit",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
    ids=["tables", "dedup", "dedup_local", "dedup_visit"],
)
def test_sender_tables_match_model(v2, dd, ph, model, t_local, dedup, local, visit):
    block = v2.host.routing_block(t_local, K)
    ids = model.routing("skew", 3, ep=EP, g=G, t_local=t_local, empty=5)
    # Unrouted rows (all slots e_total), as the layer marks them.
    ids[:, -3:] = E_TOTAL
    rank = None
    if visit:
        rank, _ = model.visit_ranks(model.pair_counts(ids, EP, G), 2, light_prepush=2)
    for s in sorted({0, EP - 1}):
        vp = None
        if visit:
            flat = rank.reshape(-1)
            vp = np.where(ids[s] < E_TOTAL, flat[np.minimum(ids[s], E_TOTAL - 1)], 0)
        want = model.sender(
            ids[s], s, g=G, e_total=E_TOTAL, block=block, dedup=dedup, local=local,
            vp=vp,
        )  # fmt: skip
        idx, words, key, cnt = dd.sender_tables(
            jnp.asarray(ids[s], jnp.int32),
            jnp.int32(s),
            g_local=G,
            e_total=E_TOTAL,
            block=block,
            dedup=dedup,
            local=local,
            vp=None if vp is None else jnp.asarray(vp, jnp.int32),
        )
        np.testing.assert_array_equal(np.asarray(idx), want["idx"])
        np.testing.assert_array_equal(np.asarray(words), want["count_words"])
        np.testing.assert_array_equal(np.asarray(key), want["key"])
        np.testing.assert_array_equal(np.asarray(cnt), want["sort_counts"])
        flat = jnp.asarray(ids[s], jnp.int32).reshape(-1)
        words_p = dd.pack_pairs(flat, idx.reshape(-1))
        np.testing.assert_array_equal(np.asarray(words_p), want["pair_words"])
        nb = (2 if dedup else 1) * E_TOTAL
        tok, seg = ph.local_sort_tokens(key, cnt, e_total=nb, topk=K, bf16_counts=True)
        np.testing.assert_array_equal(np.asarray(tok), want["tok"])
        seg = np.asarray(seg)
        np.testing.assert_array_equal(seg[0::2] if dedup else seg, want["seg"])


def _x_table_call(v2, *, n_pre, tile_m, light_prepush, visit_order, x=None):
    """plan_kernel._tables (with owned counts where x = (x_npre, x_cap)),
    in the Pallas TPU interpreter."""
    pk = v2.plan
    n_out = pk.N_TABLE_OUTS + (pk.N_X_OUTS if x else 0)

    def kernel(me_sm, r3_ref, o3_ref, seg3_ref, *outs):
        kw = {} if x is None else dict(o3=o3_ref[...], x_npre=x[0], x_cap=x[1])
        t = pk._tables(
            r3_ref[...],
            seg3_ref[...],
            me_sm[0],
            ep=EP,
            g=G,
            n_pre=n_pre,
            tile_m=tile_m,
            light_prepush=light_prepush,
            visit_order=visit_order,
            **kw,
        )
        pk._store_tables(t, outs, ep=EP, g=G)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem] * 3,
            out_specs=[vmem] * n_out,
            grid=(),
        ),
        out_shape=pk._table_out_shapes(EP, G, x=x is not None),
        interpret=pltpu.InterpretParams(),
    )

    def pad(a):
        a = jnp.asarray(a, jnp.int32)
        return jnp.pad(a, ((0, 0), (0, 0), (0, pk.LANES - G)))

    def run(r3, o3, seg_start_local, me):
        outs = call(
            jnp.asarray([me], jnp.int32),
            pad(r3),
            pad(o3),
            pk._seg3(seg_start_local, EP, G),
        )
        return [
            tuple(map(np.asarray, a)) if isinstance(a, tuple) else np.asarray(a)
            for a in pk._unpack_tables(outs)
        ]

    return run


@pytest.mark.parametrize(
    "kind,mode",
    [
        ("uniform", "x"),
        ("skew", "x"),
        ("hot", "x"),
        ("skew", "x_visit"),
        ("skew", "x_local"),
        ("hot", "x_cap"),
        ("skew", "x_dispoff"),
    ],
)
def test_table_kernel_x_matches_model(v2, model, kind, mode):
    """The table kernel with owned counts (interpreted) gives the model's
    dispatch counts, receive counts, SparseCore counts, x tables and run
    bases; under disp_off (n_pre = g, SparseCore set of 2) as well."""
    visit = mode == "x_visit"
    if mode == "x_dispoff":
        # disp_off: the SparseCore pushes every position (n_pre = g), the
        # dedup keeps the served SparseCore set (x_npre = 2).
        n_pre = G
        ids = model.routing(kind, 11, ep=EP, g=G, t_local=64, empty=5)
        case = model.build(ids, g=G, n_pre=n_pre, tile_m=64, dedup=True, x_npre=2)
    else:
        n_pre = 2
        case = _case(
            model,
            kind,
            64,
            dedup=True,
            local=mode == "x_local",
            owner_visit=visit,
            cap=24 if mode == "x_cap" else None,
        )
    run = _x_table_call(
        v2,
        n_pre=n_pre,
        tile_m=128,
        light_prepush=2,
        visit_order=0 if visit else 1,
        x=(2, case["cap"]),
    )
    for me in sorted({0, 3, EP - 1}):
        rk = case["ranks"][me]
        seg_start = case["senders"][me]["seg"]
        out = run(case["r3"], case["o3"], seg_start, me)
        seg, meta, my_visit, counts_sc, _kt, _er, _sb, _counts, plan, xt = out
        seg = seg.reshape(3, G, EP)
        live = np.arange(G)[:, None] < case["n_visit"][None, :]  # [v, d]
        np.testing.assert_array_equal(np.where(live, seg[0] & 0xFFFF, 0), rk["cnt"])
        vis = case["visit"].T  # [v, d]
        np.testing.assert_array_equal(
            np.where(live, (seg[0] >> 16) & 255, 0), np.where(live, vis, 0)
        )
        np.testing.assert_array_equal(meta, rk["meta"])
        np.testing.assert_array_equal(counts_sc, rk["counts_sc"])
        np.testing.assert_array_equal(plan[2], rk["run_base"])
        np.testing.assert_array_equal(xt[0], rk["x_g"])
        np.testing.assert_array_equal(xt[1], rk["x_k"])
        np.testing.assert_array_equal(my_visit[: case["n_visit"][me]], case["visit"][me, : case["n_visit"][me]])  # fmt: skip
        # Destination rows of the runs this rank sends: the receiver's run base.
        for d in range(EP):
            for v in range(case["n_visit"][d]):
                if rk["cnt"][v, d]:
                    e = case["visit"][d, v]
                    assert seg[2, v, d] == case["ranks"][d]["run_base"][me, e]


@pytest.mark.parametrize("kind", ["uniform", "hot"])
def test_table_kernel_whole_runs_keep_the_default_tables(v2, model, kind):
    """Owned counts equal to the pair counts (nothing deduplicated) give
    every default table unchanged, and x tables with no duplicate."""
    case = _case(model, kind, 64, dedup=False)
    plain = _x_table_call(v2, n_pre=2, tile_m=128, light_prepush=2, visit_order=1)
    x = _x_table_call(
        v2, n_pre=2, tile_m=128, light_prepush=2, visit_order=1, x=(2, 1024)
    )
    for me in (0, EP - 1):
        seg_start = case["senders"][me]["seg"]
        a = plain(case["r3"], case["r3"], seg_start, me)
        b = x(case["r3"], case["r3"], seg_start, me)
        assert len(a) == 9 and len(b) == 10
        for i in (0, 1, 2, 3, 5, 6, 7):
            np.testing.assert_array_equal(a[i], b[i])
        for p, q in zip(a[4] + a[8], b[4] + b[8]):
            np.testing.assert_array_equal(p, q)
        xg = b[9][0]
        np.testing.assert_array_equal(xg & 0xFFFF, case["r3"][:, me, :])
        assert not (xg >> 16).any()


@pytest.mark.parametrize(
    "kind,mode",
    [("uniform", "x"), ("hot", "x"), ("skew", "x_visit"), ("skew", "x_local"),
     ("hot", "x_cap"), ("skew", "tables")],
)  # fmt: skip
@pytest.mark.parametrize("flat_owner", [False, True], ids=["r3", "flat"])
def test_receiver_tables_match_model(dd, model, kind, mode, flat_owner):
    t_local = 64
    case = _case(
        model,
        kind,
        t_local,
        dedup=mode != "tables",
        local=mode == "x_local",
        owner_visit=mode == "x_visit",
        cap=24 if mode == "x_cap" else None,
    )
    stride = 4096
    pairs = np.stack([sd["pair_words"] for sd in case["senders"]])
    for me in sorted({0, 2, EP - 1}):
        rk = case["ranks"][me]
        vpos_me = None
        if mode == "x_visit":
            vpos_me = jnp.asarray(case["rank"][me], jnp.int32)
        slab_row, src = dd.receiver_tables(
            jnp.asarray(pairs, jnp.int32),
            jnp.int32(me),
            jnp.asarray(rk["run_base"], jnp.int32),
            g_local=G,
            t_local=t_local,
            topk=K,
            shard_stride=stride,
            ep=EP,
            x_g=None if mode == "tables" else jnp.asarray(rk["x_g"], jnp.int32),
            cap=case["cap"],
            vpos_me=vpos_me,
            flat_owner=flat_owner,
        )
        slab_row = np.asarray(slab_row).reshape(EP, t_local, K)
        on = rk["slab_row"] >= 0
        np.testing.assert_array_equal(slab_row[on], rk["slab_row"][on])
        sentinel = dd.off_shard_slab_row(stride, EP) - me * stride
        assert (slab_row[~on] == sentinel).all()
        if mode == "tables":
            assert src is None
        else:
            np.testing.assert_array_equal(np.asarray(src), rk["src"])


@pytest.mark.parametrize("k", [1, 2, 6])
def test_group_fold_matches_the_reshaped_reduction(dd, k):
    """Step 7 O x_flat_owner: _group_fold = the [..., n // k, k] reduction
    broadcast back, for min and i32 add (wrapping), over the full i32 range."""
    rng = np.random.default_rng(k)
    v = rng.integers(-(2**31), 2**31 - 1, size=(3, 40 * k), dtype=np.int64)
    v = v.astype(np.int32)
    mask = rng.random(v.shape) < 0.6
    masked = np.where(mask, v, np.int32(2**30))
    got_min = np.asarray(dd._group_fold(jnp.asarray(masked), k, jnp.minimum))
    want_min = np.repeat(masked.reshape(3, -1, k).min(axis=2), k, axis=1)
    np.testing.assert_array_equal(got_min, want_min)
    got_sum = np.asarray(dd._group_fold(jnp.asarray(v), k, jnp.add))
    want_sum = np.repeat(
        v.reshape(3, -1, k).sum(axis=2, dtype=np.int64).astype(np.int32), k, axis=1
    )
    np.testing.assert_array_equal(got_sum, want_sum)


@pytest.mark.parametrize(
    "light_prepush,visit_order", [(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)]
)
def test_visit_positions_match_the_table_order(dd, model, light_prepush, visit_order):
    case = _case(model, "skew", 64, dedup=False)
    for n_pre in (1, 2, 4, G):
        want, n_visit = model.visit_ranks(
            case["r3"], n_pre, light_prepush=light_prepush, visit_order=visit_order
        )
        got = dd.visit_positions(
            jnp.asarray(case["r3"].reshape(EP, E_TOTAL), jnp.int32),
            ep=EP,
            g=G,
            n_pre=n_pre,
            light_prepush=light_prepush,
            visit_order=visit_order,
        )
        np.testing.assert_array_equal(np.asarray(got), want)
        for me in (0, EP - 1):
            visit = np.argsort(want[me])
            mine = np.asarray(
                dd.my_visit_positions(jnp.asarray(visit), jnp.int32(n_visit[me]), G)
            )
            has = np.argsort(want[me])[: n_visit[me]]
            np.testing.assert_array_equal(mine[has], want[me][has])


def test_x_tables_alone_match_the_default_plan(v2, dd, ph, model):
    """x_tables without the dedup (DZt): the combine tables and slab rows
    from run base + idx equal routing_pairs_from_tables' bit for bit."""
    host = v2.host
    t_local = 64
    block = host.routing_block(t_local, K)
    ids = model.routing("skew", 5, ep=EP, g=G, t_local=t_local, empty=3)
    ids[:, -2:] = E_TOTAL  # unrouted rows
    case = model.build(ids, g=G, n_pre=2, tile_m=64, dedup=False, block=block)
    run = _x_table_call(v2, n_pre=2, tile_m=128, light_prepush=2, visit_order=0)
    T = EP * t_local
    nbl = t_local * K // block
    expert_blocks = jnp.asarray(ids.reshape(T * K // block, block), jnp.int32)
    stride = host.ragged_stride_bound(T, K, E_TOTAL, 256)
    words = []
    for s in range(EP):
        idx_s = dd.sender_tables(
            jnp.asarray(ids[s], jnp.int32),
            jnp.int32(s),
            g_local=G,
            e_total=E_TOTAL,
            block=block,
            dedup=False,
        )[0]
        words.append(dd.pack_pairs(expert_blocks[s * nbl : (s + 1) * nbl].reshape(-1), idx_s.reshape(-1)))  # fmt: skip
    words = jnp.stack(words)
    for me in sorted({0, 4, EP - 1}):
        out = run(case["r3"], case["r3"], case["senders"][me]["seg"], me)
        recv_base_me, mirror_base, run_base = (jnp.asarray(a) for a in out[8])
        local_blocks = expert_blocks[me * nbl : (me + 1) * nbl]
        gathered = host.GatheredPairs(
            expert_blocks=expert_blocks,
            local_blocks=local_blocks,
            block_hist=host.pair_block_hist(local_blocks, E_TOTAL),
            rows_by_dest=jnp.asarray(case["r3"].reshape(EP, E_TOTAL), jnp.int32),
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
        )
        ids_me = jnp.asarray(ids[me], jnp.int32)
        idx = dd.sender_tables(
            ids_me, jnp.int32(me), g_local=G, e_total=E_TOTAL, block=block, dedup=False
        )[0]
        pos, mirror = dd.pair_positions(ids_me, idx, recv_base_me, mirror_base)
        slab_row, _ = dd.receiver_tables(
            words, jnp.int32(me), run_base, g_local=G, t_local=t_local, topk=K,
            shard_stride=stride, ep=EP,
        )  # fmt: skip
        np.testing.assert_array_equal(np.asarray(pos), np.asarray(want[0]))
        np.testing.assert_array_equal(np.asarray(mirror), np.asarray(want[1]))
        np.testing.assert_array_equal(np.asarray(slab_row), np.asarray(want[2]))


# --------------------------------------------------------------------------- #
# The served buckets (slow; the K2e pod run sets DZX_LARGE=512,1024).
# --------------------------------------------------------------------------- #
_LARGE = [int(x) for x in os.environ.get("DZX_LARGE", "").split(",") if x]


@pytest.mark.skipif(not _LARGE, reason="DZX_LARGE (rows per rank) not set")
@pytest.mark.parametrize("t_local", _LARGE or [512])
@pytest.mark.parametrize("kind", ["uniform", "skew", "hot"])
def test_served_buckets_match_model(v2, dd, ph, model, t_local, kind):
    """At the served shape (EP8, 48 experts per rank, K32Z tiles of 256 rows):
    the whole device table chain -- sender tables and owned-first sort, the
    table kernel with owned counts (interpreted), receiver slab rows and SRC
    -- equals the model bitwise on every rank, and the model's replay holds."""
    block = v2.host.routing_block(t_local, K)
    case = _case(model, kind, t_local, tile_m=256, block=block, dedup=True)
    model.replay(case)
    run = _x_table_call(
        v2, n_pre=2, tile_m=128, light_prepush=2, visit_order=1, x=(2, case["cap"])
    )
    words = []
    for s in range(EP):
        want = case["senders"][s]
        idx, cw, key, cnt = dd.sender_tables(
            jnp.asarray(case["ids"][s], jnp.int32),
            jnp.int32(s),
            g_local=G,
            e_total=E_TOTAL,
            block=block,
            dedup=True,
        )
        np.testing.assert_array_equal(np.asarray(idx), want["idx"])
        np.testing.assert_array_equal(np.asarray(cw), want["count_words"])
        tok, seg = ph.local_sort_tokens(
            key, cnt, e_total=2 * E_TOTAL, topk=K, bf16_counts=True
        )
        np.testing.assert_array_equal(np.asarray(tok), want["tok"])
        np.testing.assert_array_equal(np.asarray(seg)[0::2], want["seg"])
        flat = jnp.asarray(case["ids"][s], jnp.int32).reshape(-1)
        words.append(dd.pack_pairs(flat, idx.reshape(-1)))
    words = jnp.stack(words)
    for me in range(EP):
        rk = case["ranks"][me]
        out = run(case["r3"], case["o3"], case["senders"][me]["seg"], me)
        np.testing.assert_array_equal(out[1], rk["meta"])
        np.testing.assert_array_equal(out[3], rk["counts_sc"])
        np.testing.assert_array_equal(out[9][0], rk["x_g"])
        np.testing.assert_array_equal(out[9][1], rk["x_k"])
        slab_row, src = dd.receiver_tables(
            words,
            jnp.int32(me),
            jnp.asarray(out[8][2]),
            g_local=G,
            t_local=t_local,
            topk=K,
            shard_stride=16384,
            ep=EP,
            x_g=jnp.asarray(out[9][0]),
            cap=case["cap"],
        )
        on = rk["slab_row"] >= 0
        got = np.asarray(slab_row).reshape(EP, t_local, K)
        np.testing.assert_array_equal(got[on], rk["slab_row"][on])
        np.testing.assert_array_equal(np.asarray(src), rk["src"])


# --------------------------------------------------------------------------- #
# Lowering, names, refusals.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t_local", [16, 256], ids=["decode16", "dzx256"])
def test_x_defaults_lower_to_the_same_program(v2, t_local):
    k32z = v2.host.WeightFormat.K32Z
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=t_local)
    base = lower(push_options=_tuned_push_options(8))
    assert lower(push_options={**_tuned_push_options(8), **X_DEFAULTS}) == base
    for n in _names(base, "moe_tc_routing_exchange") + _names(base, "push_fused_moe_"):
        assert "_xt" not in n and "_xd" not in n and "_xg0" not in n, n


def test_x_switches_leave_other_buckets_alone(v2):
    """At 16 rows per rank (and above 1024) the dedup does not apply: the
    switches build the default program."""
    k32z = v2.host.WeightFormat.K32Z
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=16)
    base = lower(push_options=_tuned_push_options(8))
    on = dict(x_dedup=True, x_dedup_local=True, x_src_cap=64)
    assert lower(push_options={**_tuned_push_options(8), **on}) == base
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import dedup

    assert dedup.x_applies(512) and dedup.x_applies(1024)
    assert not dedup.x_applies(64) and not dedup.x_applies(2048)


# (option overrides, kernel-name prefix, expected token in every such name)
_X_NAMES = [
    (dict(x_tables=True), "moe_tc_routing_exchange_ep2_w", "_xt"),
    (dict(x_dedup=True), "moe_tc_routing_exchange_ep2_w", "_vo1_xd512"),
    (dict(x_dedup=True), "push_fused_moe_", "_xd"),
    (
        dict(x_dedup=True, x_dedup_local=True),
        "moe_tc_routing_exchange_ep2_w",
        "_xd512l",
    ),
    (
        dict(x_dedup=True, x_owner_visit=True),
        "moe_tc_routing_exchange_ep2_w",
        "_xd512v",
    ),
    (dict(x_dedup=True, x_src_cap=64), "moe_tc_routing_exchange_ep2_w", "_xd64"),
    (dict(x_dedup=True, x_gather_off=True), "push_fused_moe_", "_xd_xg0"),
]


def test_x_switches_name_their_programs(v2):
    """At 256 rows per rank (two shards of 8 experts, K32Z), each switch set
    lowers (Mosaic, TPU7x target) to its own text and puts its token in the
    kernel it changes; the visit-order owner adds the counts exchange."""
    k32z = v2.host.WeightFormat.K32Z
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=256)
    texts = {"base": lower(push_options=_tuned_push_options(8))}
    for over, prefix, token in _X_NAMES:
        key = ",".join(f"{k}={v}" for k, v in sorted(over.items()))
        text = texts.get(key) or lower(push_options={**_tuned_push_options(8), **over})
        texts[key] = text
        names = [
            n
            for n in _names(text, prefix)
            if "_tables" in n or prefix.startswith("push")
        ]
        assert names and all(token in n for n in names), (key, names)
    assert len(set(texts.values())) == len(texts), "two switch sets share a program"
    visit = texts["x_dedup=True,x_owner_visit=True"]
    plain = [n for n in _names(visit, "moe_tc_routing_exchange") if "_rows" not in n]
    assert plain, "the visit-order owner's counts exchange is missing"
    assert "_vo1" not in "".join(_names(visit, "moe_tc_routing_exchange"))


@pytest.mark.parametrize(
    "over,what",
    [
        (dict(x_dedup_local=True), "refine x_dedup"),
        (dict(x_gather_off=True), "refine x_dedup"),
        (dict(x_dedup=True, x_src_cap=-1), "x_src_cap"),
        (dict(x_dedup=True, ret_last_off=1), "ret_last_off"),
        (dict(x_tables=True, exchange_tables=False, kernel_plan_tables=False), "exchange-table path"),
        (dict(x_dedup=True, tc_exchange=False, exchange_row_gather=False,
              exchange_tables=False), "exchange-table path"),
    ],
)  # fmt: skip
def test_x_layer_refusals(v2, over, what):
    k32z = v2.host.WeightFormat.K32Z
    with pytest.raises(ValueError, match=what):
        _lower_push(
            v2,
            weight_format=k32z,
            t_local=256,
            push_options={**_tuned_push_options(8), **over},
        )


@pytest.mark.parametrize(
    "kw,what",
    [
        (dict(x_gather_off=True), "x_gather_off"),
        (dict(x_dedup=True, ret_last_off=1), "ret_last_off"),
    ],
)
def test_x_kernel_refusals(v2, kw, what):
    with pytest.raises(ValueError, match=what):
        v2.push_kernel._build_push_moe_kernel(**_push_build_args(v2, **kw))


def test_x_table_builder_refusal(v2):
    with pytest.raises(ValueError, match="x_dedup needs"):
        v2.plan.build_exchange_rows_tables(
            ep=8,
            width=3968,
            n_rows=3072,
            count_at=3584,
            e_total=384,
            n_pre=2,
            tile_m=128,
            x_dedup=True,
            x_npre=0,
            x_cap=0,
        )


def test_harness_x_tokens(bd, v2):
    assert bd.push_knobs_of("DZu1") == dict(x_tables=True)
    assert bd.push_knobs_of("DZv1") == dict(x_dedup=True)
    assert bd.push_knobs_of("DZv1h0z1") == dict(
        x_dedup=True, hbm_staging=False, fold_select=True
    )
    assert bd.push_knobs_of("DZv1w1e1b64") == dict(
        x_dedup=True, x_dedup_local=True, x_owner_visit=True, x_src_cap=64
    )
    assert bd.push_knobs_of("DZv1j1") == dict(x_dedup=True, x_gather_off=True)
    assert bd.config_exact("DZv1") and bd.config_exact("DZu1h0")
    assert not bd.config_exact("DZv1j1")
    assert bd.knobs_of("DZv1w1") == {}
    assert set(o for o, _ in bd.PUSH_KNOBS.values()) <= set(v2.push_options)
    for bad in ("AZv1", "Au1", "AZj1"):
        with pytest.raises(ValueError):
            bd.push_knobs_of(bad)
