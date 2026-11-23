# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Step 7 K2 push switches (DZX groundwork, all off by
default).

* Defaults: the push layer on #2102's tuned bundle lowers to the same text
  with every K2 switch passed at its default as with none passed, for K32Z
  and FP4 at a decode-sized and a combine-sized bucket.
* Names: each switch that is on puts its token in a kernel name -- push
  kernel `_2h`, `_dp`, `_rlo<k>`, `_cmo`; table kernel `_vo1`; combine
  `_sel` -- and every switch set lowers to its own text, so no two programs
  share a name or a compilation-cache entry.
* The select fold: the fused combine with fold_select, in the Pallas TPU
  interpreter, is where(routes, combine, 0) bit for bit (NaN rows included).
* The visit order (visit_order=1, "sc_then_id"): on random routings with
  empty experts, a numpy model of plan_kernel._tables' order puts each
  destination's n_pre lightest experts first (lightest at 0, the set
  light_prepush=2 hands the SparseCore) and the rest in increasing local id;
  every duplicate (a token's non-lowest local expert on a rank) at a
  TensorCore position has its owner (the lowest) visited before it, which the
  default order does not give. The table kernel, run in the Pallas TPU
  interpreter, gives the model's order and, apart from the order, the same
  tables as the default order.
* Refusals, and the breakdown harness's D/DZ token grammar.

Nothing here needs a TPU: the layer lowers for a TPU7x target on two CPU
devices, and the kernels that run, run in the Pallas TPU interpreter.

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import contextlib
import os
import re
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

# CPU JAX with virtual devices for the EP meshes; no effect once JAX is up.
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

import dsv41_fused_ep_reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

TREE = Path(__file__).resolve().parents[5]
ROWBLK = 2  # host.ROWBLK, checked in the fixture


@pytest.fixture(scope="module")
def v2():
    try:
        package = ref.import_fused_moe_v2(isolated=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fused_moe_v2 does not import here: {exc!r}")
    from sglang_tpu.moe.pallas.fused_moe_v2 import host, kernel, push_fused_moe

    assert host.ROWBLK == ROWBLK
    return SimpleNamespace(
        package=package,
        host=host,
        kernel=kernel,
        push_kernel=push_fused_moe.kernel,
        plan=push_fused_moe.plan_kernel,
        push_options=push_fused_moe.PUSH_OPTIONS,
    )


# --------------------------------------------------------------------------- #
# Lowering for TPU on this host (as test_dsv41_k32z_kernel.py does).
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _tpu7x_target():
    from jax._src import tpu_info as core_tpu_info
    from jax._src.pallas.mosaic import tpu_info

    names = ("get_tpu_info", "get_device_kind", "is_tpu_device")
    modules = (tpu_info, core_tpu_info)
    saved_public = pltpu.get_tpu_info
    saved = [{n: getattr(m, n) for n in names} for m in modules]
    kind = "TPU7x"
    info = tpu_info.get_tpu_info_for_chip(
        tpu_info.chip_version_from_device_kind(kind), 1
    )
    pltpu.get_tpu_info = lambda: info
    for m in modules:
        m.get_tpu_info = lambda: info
        m.get_device_kind = lambda: kind
        m.is_tpu_device = lambda: True
    try:
        yield info
    finally:
        pltpu.get_tpu_info = saved_public
        for m, old in zip(modules, saved):
            for n, v in old.items():
                setattr(m, n, v)


def _tuned_push_options(g_local):
    """#2102's tuned push bundle (config D), as the dispatch x routing
    harness copies it from test_fused_ep_moe_push.py."""
    return dict(
        sc_prepush=2,
        tc_exchange=True,
        onehot_slots=True,
        tail_barrier=False,
        hbm_staging=True,
        dispatch_ahead=g_local,
        matmul_scale_repeat=True,
        exchange_row_gather=True,
        bf16_sort_counts=True,
        slim_prepush_barrier=True,
        fused_dispatch_meta=True,
        table_kernel=True,
        exchange_tables=True,
        kernel_plan_tables=True,
        light_prepush=2,
        single_barrier=True,
        early_scale_scatter=2,
    )


K2_DEFAULTS = dict(
    visit_order=0,
    two_hop=False,
    defer_push=False,
    fold_select=False,
    ret_last_off=0,
    commit_only=False,
)


def _lower_push(
    v2,
    *,
    weight_format,
    push_options,
    hidden=1536,
    inter=768,
    ep=2,
    g_local=8,
    t_local=16,
    bits=2,
    logical=None,
    ablation=None,
):
    host = v2.host
    mesh = Mesh(np.asarray(jax.devices()[:ep]), ("d",))
    rows, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    e_total = ep * g_local
    t = ep * t_local
    if weight_format == host.WeightFormat.K32Z:
        geo = host.k32z_table_geometry(hidden, inter, bits)
        s1 = jax.ShapeDtypeStruct(
            (e_total * geo.rows1, geo.cols1), jnp.uint8, sharding=rows
        )
        s2 = jax.ShapeDtypeStruct(
            (e_total * geo.rows2, geo.cols2), jnp.uint8, sharding=rows
        )
        cap = 256
    else:
        r1 = host.fp4_scale_rows(hidden, 512, padded=True)
        r2 = host.fp4_scale_rows(inter, 512, padded=True)
        s1 = jax.ShapeDtypeStruct((e_total * r1, 2 * inter), jnp.float32, sharding=rows)
        s2 = jax.ShapeDtypeStruct((e_total * r2, hidden), jnp.float32, sharding=rows)
        cap = 128
    w1 = jax.ShapeDtypeStruct(
        (e_total, hidden, 2 * inter), jnp.float4_e2m1fn, sharding=rows
    )
    w2 = jax.ShapeDtypeStruct(
        (e_total, inter, hidden), jnp.float4_e2m1fn, sharding=rows
    )
    x = jax.ShapeDtypeStruct((t, hidden), jnp.bfloat16, sharding=rows)
    g = jax.ShapeDtypeStruct((t, e_total), jnp.float32, sharding=rows)
    bias = jax.ShapeDtypeStruct((e_total,), jnp.float32, sharding=repl)
    kw = {} if ablation is None else {"ablation": ablation}

    def layer(x, g, w1, w2, s1, s2, bias):
        return v2.package.fused_ep_moe_v2(
            x,
            w1,
            w2,
            s1,
            s2,
            g,
            topk=6,
            renormalize=True,
            mesh=mesh,
            capacity=cap,
            weight_format=weight_format,
            rhs_qb=512,
            act_fn="silu_and_mul_with_clamp",
            mesh_ep_ranks=tuple(range(ep)),
            sharded_plan=True,
            scoring_fn="sqrtsoftplus",
            score_bias=bias,
            routed_scaling_factor=1.5,
            logical_intermediate_size=logical,
            dispatch="push",
            push_options=dict(push_options),
            **kw,
        )

    with _tpu7x_target():
        return (
            jax.jit(layer)
            .trace(x, g, w1, w2, s1, s2, bias)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )


def _names(text, prefix):
    return sorted(set(re.findall(rf"({prefix}\w*)", text)))


# --------------------------------------------------------------------------- #
# Defaults and names.
# --------------------------------------------------------------------------- #
def test_k2_options_are_push_options(v2):
    assert set(K2_DEFAULTS) <= set(v2.push_options)


@pytest.mark.parametrize("t_local", [16, 256], ids=["decode16", "combine256"])
@pytest.mark.parametrize("fmt", ["k32z", "fp4"])
def test_k2_defaults_lower_to_the_same_program(v2, fmt, t_local):
    """Every K2 switch passed at its default lowers to the very text of the
    bundle without them (so k32z-r2's programs are unchanged)."""
    wf = v2.host.WeightFormat.K32Z if fmt == "k32z" else v2.host.WeightFormat.FP4
    extra = {} if fmt == "k32z" else {"inter": 1024}
    lower = partial(_lower_push, v2, weight_format=wf, t_local=t_local, **extra)
    base = lower(push_options=_tuned_push_options(8))
    assert lower(push_options={**_tuned_push_options(8), **K2_DEFAULTS}) == base
    push = _names(base, "push_fused_moe_")
    assert not any(re.search(r"_(2h|dp|rlo\d+|cmo)(_|$)", n) for n in push), push
    assert not any("_vo" in n for n in _names(base, "moe_tc_routing_exchange"))
    assert not any(n.endswith("_sel") for n in _names(base, "moe_v2_combine"))


# (option overrides, kernel-name prefix, expected token in that kernel's name)
_KNOB_NAMES = [
    (dict(hbm_staging=False), None, None),
    (dict(hbm_staging=False, two_hop=True), "push_fused_moe_", "_2h"),
    (dict(hbm_staging=False, two_hop=True, defer_push=True), "push_fused_moe_", "_dp"),
    (dict(defer_push=True), "push_fused_moe_", "_dp"),
    (dict(ret_last_off=2), "push_fused_moe_", "_rlo2"),
    (dict(hbm_staging=False, ret_last_off=1), "push_fused_moe_", "_rlo1"),
    (dict(commit_only=True), "push_fused_moe_", "_cmo"),
    (dict(hbm_staging=False, commit_only=True), "push_fused_moe_", "_cmo"),
    (dict(visit_order=1), "moe_tc_routing_exchange", "_vo1"),
    (dict(fold_select=True), "moe_v2_combine", "_sel"),
]


@pytest.mark.parametrize("fmt", ["k32z", "fp4"])
def test_k2_switches_name_their_programs(v2, fmt):
    """Each switch set lowers (Mosaic, TPU7x target) to its own text and puts
    its token in the kernel it changes; no two sets share a program."""
    wf = v2.host.WeightFormat.K32Z if fmt == "k32z" else v2.host.WeightFormat.FP4
    extra = {} if fmt == "k32z" else {"inter": 1024}
    # 256 local rows: the fused combine applies, so fold_select has a kernel.
    lower = partial(_lower_push, v2, weight_format=wf, t_local=256, **extra)
    texts = {"base": lower(push_options=_tuned_push_options(8))}
    for over, prefix, token in _KNOB_NAMES:
        key = ",".join(f"{k}={v}" for k, v in sorted(over.items()))
        text = lower(push_options={**_tuned_push_options(8), **over})
        texts[key] = text
        if prefix is not None:
            names = _names(text, prefix)
            assert names and all(token in n for n in names), (key, names)
    assert len(set(texts.values())) == len(texts), "two switch sets share a program"
    staged = _names(texts["base"], "push_fused_moe_")
    assert all("_hbmst" in n for n in staged), staged


def test_two_hop_and_defer_change_the_vmem_ship_build(v2):
    """hbm_staging=False ships tiles from VMEM at this shape; two_hop takes the
    two-hop path back (its own text), defer_push then defers its push."""
    k32z = v2.host.WeightFormat.K32Z
    opts = {**_tuned_push_options(8), "hbm_staging": False}
    lower = partial(_lower_push, v2, weight_format=k32z)
    ship = lower(push_options=opts)
    hop = lower(push_options={**opts, "two_hop": True})
    hop_dp = lower(push_options={**opts, "two_hop": True, "defer_push": True})
    assert len({ship, hop, hop_dp}) == 3
    (name,) = _names(ship, "push_fused_moe_")
    assert _names(hop, "push_fused_moe_") == [name + "_2h"]
    assert _names(hop_dp, "push_fused_moe_") == [name + "_2h_dp"]


@pytest.mark.parametrize("fuse_scales", [True, False], ids=["fused_scales", "coef"])
def test_fold_select_is_the_select_after_the_combine(v2, monkeypatch, fuse_scales):
    """The combine with fold_select (in the Pallas TPU interpreter) gives
    where(routes, combine, 0) bit for bit, NaN arrival rows of unrouted tokens
    included: the push path's fused-scale build and (K2f) the all-gather
    path's coefficient build."""
    t_local, topk, hidden = 256, 6, 256
    lanes = 128
    recv_rows = 2048
    rng = np.random.default_rng(5)
    arrivals = rng.normal(size=(recv_rows, hidden // lanes, lanes)).astype(np.float32)
    arrivals[-64:] = np.nan  # rows nobody wrote
    arrivals = jnp.asarray(arrivals).astype(jnp.float8_e4m3fn)
    routes = rng.random((t_local, 1)) > 0.15
    pos = rng.integers(0, recv_rows - 64, size=(t_local, topk))
    pos = np.where(routes, pos, recv_rows - 1 - rng.integers(0, 64, size=pos.shape))
    sublanes = 16
    scales = jnp.asarray(rng.uniform(0.5, 2.0, size=(sublanes, lanes)), jnp.float32)
    scl = jnp.asarray(rng.integers(0, sublanes * lanes, size=(t_local, topk)))
    weights = np.where(routes, rng.uniform(0.0, 1.0, size=(t_local, topk)), 0.0)
    weights = jnp.asarray(weights, jnp.float32)
    real_call = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", partial(real_call, interpret=pltpu.InterpretParams())
    )
    build = partial(
        v2.kernel._build_combine_kernel,
        t_local=t_local,
        topk=topk,
        hidden=hidden,
        lane_blocks=hidden // lanes,
        wire_dtype=jnp.float8_e4m3fn,
        out_dtype=jnp.bfloat16,
        fuse_scales=fuse_scales,
    )
    # The coefficient build takes scale * weight precomputed (v2's layer).
    coef = scales.reshape(-1)[scl] * weights

    def run(fn, **kw):
        if fuse_scales:
            return fn(arrivals, scales, scl, weights, jnp.asarray(pos), **kw)
        return fn(arrivals, coef, jnp.asarray(pos), **kw)

    try:
        with _tpu7x_target():
            plain = run(build())
            folded = run(build(fold_select=True), routes=routes)
    except NotImplementedError as exc:  # pragma: no cover - interpreter gap
        pytest.skip(f"the TPU interpreter cannot run the combine here: {exc}")
    plain, folded = np.asarray(plain), np.asarray(folded)
    assert np.isnan(plain.astype(np.float32)[~routes[:, 0]]).any()
    want = np.where(routes, plain, np.zeros_like(plain))
    np.testing.assert_array_equal(folded.view(np.uint16), want.view(np.uint16))


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #
def _push_build_args(v2, **kw):
    args = dict(
        g_local=48,
        capacity=256,
        hidden=5120,
        inter=2304,
        ep=8,
        weight_format=v2.host.WeightFormat.K32Z,
        ragged_rows_alloc=4096,
        act_fn="silu_and_mul_with_clamp",
        push_tokens=64,
        sc_prepush=2,
    )
    args.update(kw)
    return args


@pytest.mark.parametrize(
    "kw,what",
    [
        (dict(ret_last_off=49), "ret_last_off"),
        (dict(ret_last_off=-1), "ret_last_off"),
        (dict(ret_last_off=1, commit_only=True), "commit_only"),
        (dict(commit_only=True, ablation=frozenset({"ret_off"})), "ret_off"),
        (dict(ret_last_off=2, ablation=frozenset({"ret_off"})), "ret_off"),
        (dict(ret_last_off=1, g_local=4, ep=8), "unused words"),
    ],
)
def test_push_k2_build_refusals(v2, kw, what):
    with pytest.raises(ValueError, match=what):
        v2.push_kernel._build_push_moe_kernel(**_push_build_args(v2, **kw))


def test_combine_fold_select_takes_the_route_flags(v2):
    """The coefficient combine built with fold_select (K2f) needs the route
    flags, and one built without refuses them."""
    kw = dict(
        t_local=256,
        topk=6,
        hidden=1536,
        lane_blocks=12,
        wire_dtype=jnp.float8_e4m3fn,
        out_dtype=jnp.bfloat16,
    )
    pos = jnp.zeros((256, 6), jnp.int32)
    coef = jnp.zeros((256, 6), jnp.float32)
    arrivals = jnp.zeros((16, 12, 128), jnp.float8_e4m3fn)
    routes = jnp.ones((256, 1), bool)
    with _tpu7x_target():
        folded = v2.kernel._build_combine_kernel(**kw, fold_select=True)
        plain = v2.kernel._build_combine_kernel(**kw)
    with pytest.raises(ValueError, match="fold_select"):
        folded(arrivals, coef, pos)
    with pytest.raises(ValueError, match="fold_select"):
        plain(arrivals, coef, pos, routes=routes)


@pytest.mark.parametrize(
    "over,what",
    [
        (dict(visit_order=2), "visit_order"),
        (
            dict(
                table_kernel=False,
                exchange_tables=False,
                kernel_plan_tables=False,
                light_prepush=0,
                visit_order=1,
            ),
            "table kernel",
        ),
        (dict(ret_last_off=9), "ret_last_off"),
    ],
)
def test_push_k2_layer_refusals(v2, over, what):
    k32z = v2.host.WeightFormat.K32Z
    with pytest.raises(ValueError, match=what):
        _lower_push(
            v2, weight_format=k32z, push_options={**_tuned_push_options(8), **over}
        )


def test_visit_order_refused_by_the_table_builders(v2):
    with pytest.raises(ValueError, match="visit_order"):
        v2.plan.build_exchange_rows_tables(
            ep=8,
            width=3968,
            n_rows=3072,
            count_at=0,
            e_total=384,
            n_pre=2,
            tile_m=128,
            visit_order=3,
        )


# --------------------------------------------------------------------------- #
# The visit order: numpy model, owner invariant, table kernel.
# --------------------------------------------------------------------------- #
def _model_order(r3, n_pre, *, light_prepush=2, visit_order=0):
    """plan_kernel._tables' visit lists [ep (d), g] (positions past n_visit
    hold the experts without rows) and n_visit [ep], in numpy."""
    ep, _, g = r3.shape
    ea = ((r3 + ROWBLK - 1) // ROWBLK * ROWBLK).sum(axis=0)  # [d, e]
    lane = np.arange(g)
    visits, n_visits = [], []
    for d in range(ep):
        # Most rows first, ties by index: v2's expert_visit_list key.
        key = lane - ea[d] * g
        rank = np.array([(key < key[e]).sum() for e in range(g)])
        n_visit = int((ea[d] > 0).sum())
        n_lp = min(n_visit, n_pre)
        m = n_visit - n_lp
        light = (rank >= m) & (rank < n_visit)
        if visit_order == 1:
            tc = rank < m
            by_id = np.cumsum(tc) - tc
            rank = np.where(light, n_visit - 1 - rank, np.where(tc, by_id + n_lp, rank))
        elif light_prepush:
            rest = rank
            if light_prepush == 2:
                half = m >> 1
                rest = np.where(rank & 1, half - 1 - (rank >> 1), half + (rank >> 1))
            rank = np.where(
                light, n_visit - 1 - rank, np.where(rank < m, rest + n_lp, rank)
            )
        assert sorted(rank) == list(range(g))
        visit = np.empty(g, np.int64)
        visit[rank] = lane
        visits.append(visit)
        n_visits.append(n_visit)
    return np.stack(visits), np.array(n_visits)


def _routing(seed, ep, g, t_local, topk=6, empty=0):
    """Every rank's tokens' topk expert ids [ep, t_local, topk] (distinct per
    token) and the pair counts r3 [s, d, e]. `empty` experts per rank get no
    token, and the rest follow a skewed (Zipf-like) popularity."""
    rng = np.random.default_rng(seed)
    e_total = ep * g
    weight = 1.0 / np.arange(1, e_total + 1) ** 0.8
    rng.shuffle(weight)
    for d in range(ep):
        weight[d * g + rng.choice(g, empty, replace=False)] = 0.0
    p = weight / weight.sum()
    ids = np.empty((ep, t_local, topk), np.int64)
    for s in range(ep):
        for t in range(t_local):
            ids[s, t] = rng.choice(e_total, topk, replace=False, p=p)
    r3 = np.zeros((ep, ep, g), np.int64)
    for s in range(ep):
        np.add.at(r3[s], (ids[s] // g, ids[s] % g), 1)
    return ids, r3


def _late_owners(ids, visit, n_pre):
    """Duplicates at a TensorCore position whose owner (the token's lowest
    local expert on that rank) is visited after them."""
    ep, g = visit.shape
    pos = np.empty_like(visit)
    for d in range(ep):
        pos[d, visit[d]] = np.arange(g)
    late = 0
    for tokens in ids:
        for ex in tokens:
            dest, local = ex // g, ex % g
            for d in np.unique(dest):
                locs = np.sort(local[dest == d])
                owner = pos[d, locs[0]]
                for dup in locs[1:]:
                    if pos[d, dup] >= n_pre and owner > pos[d, dup]:
                        late += 1
    return late


@pytest.mark.parametrize("n_pre", [1, 2, 4])
@pytest.mark.parametrize("empty", [0, 5])
def test_sc_then_id_order_and_the_owner_rule(n_pre, empty):
    ep, g = 8, 48
    ids, r3 = _routing(7 + empty, ep, g, t_local=512, empty=empty)
    visit, n_visit = _model_order(r3, n_pre, visit_order=1)
    lp_visit, lp_n = _model_order(r3, n_pre, light_prepush=2)
    np.testing.assert_array_equal(n_visit, lp_n)
    ea = ((r3 + 1) // 2 * 2).sum(axis=0)
    for d in range(ep):
        n_lp = min(n_visit[d], n_pre)
        # The SparseCore positions: light_prepush=2's, the lightest first.
        np.testing.assert_array_equal(visit[d, :n_lp], lp_visit[d, :n_lp])
        rows = ea[d, visit[d, :n_lp]]
        assert list(rows) == sorted(rows)
        if n_visit[d] > n_lp:
            assert rows.max() <= ea[d, visit[d, n_lp : n_visit[d]]].min()
        # Then every other expert with rows, by local id.
        rest = visit[d, n_lp : n_visit[d]]
        assert list(rest) == sorted(rest)
        assert set(visit[d, : n_visit[d]]) == set(np.nonzero(ea[d])[0])
        assert not ea[d, visit[d, n_visit[d] :]].any()
    assert _late_owners(ids, visit, n_pre) == 0
    # The default order (heaviest mid-kernel) breaks the rule here.
    assert _late_owners(ids, lp_visit, n_pre) > 0


def _table_call(v2, ep, g, n_pre, tile_m, light_prepush, visit_order):
    """plan_kernel's table kernel body, in the Pallas TPU interpreter."""
    pk = v2.plan

    def kernel(me_sm, r3_ref, seg3_ref, *outs):
        t = pk._tables(
            r3_ref[...],
            seg3_ref[...],
            me_sm[0],
            ep=ep,
            g=g,
            n_pre=n_pre,
            tile_m=tile_m,
            light_prepush=light_prepush,
            visit_order=visit_order,
        )
        pk._store_tables(t, outs, ep=ep, g=g)

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem],
            out_specs=[vmem] * pk.N_TABLE_OUTS,
            grid=(),
        ),
        out_shape=pk._table_out_shapes(ep, g),
        interpret=pltpu.InterpretParams(),
    )

    def run(r3, seg_start_local, me):
        r3p = jnp.pad(jnp.asarray(r3, jnp.int32), ((0, 0), (0, 0), (0, pk.LANES - g)))
        outs = call(jnp.asarray([me], jnp.int32), r3p, pk._seg3(seg_start_local, ep, g))
        return pk._unpack_tables(outs)

    return run


@pytest.mark.parametrize(
    "ep,g,t_local,empty", [(2, 8, 16, 1), (8, 48, 256, 3)], ids=["small", "served"]
)
def test_table_kernel_visits_in_model_order(v2, ep, g, t_local, empty):
    """The table kernel (interpreted) orders every destination's visits as
    the model does, for the default (light_prepush=2) and sc_then_id orders;
    every table that does not depend on the order is the same under both,
    and the segment table holds the same runs, reordered."""
    n_pre, tile_m = 2, 128
    _, r3 = _routing(3, ep, g, t_local, empty=empty)
    tables = {}
    for vo in (0, 1):
        run = _table_call(v2, ep, g, n_pre, tile_m, 2, vo)
        want, n_visit = _model_order(r3, n_pre, light_prepush=2, visit_order=vo)
        for me in sorted({0, ep - 1}):
            # Segment starts: this shard's local sort, by global expert.
            seg_start = np.concatenate([[0], np.cumsum(r3[me].reshape(-1))[:-1]])
            out = [
                tuple(map(np.asarray, a)) if isinstance(a, tuple) else np.asarray(a)
                for a in run(r3, seg_start, me)
            ]
            seg, meta, my_visit, counts_sc, ktab, erows, sbase, counts, plan = out
            seg_a = seg.reshape(3, g, ep)[0]  # [v, d]
            got = (seg_a.T >> 16) & 255  # [d, v]
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(my_visit, want[me])
            assert counts[v2.host.COUNT_VISITS].sum() == n_visit[me]
            tables[(vo, me)] = out
    for me in sorted({0, ep - 1}):
        a, b = tables[(0, me)], tables[(1, me)]
        # meta, counts_sc, transport tables, expert rows / bases, counts and
        # plan tables do not depend on the order (same SparseCore set).
        for i in (1, 3, 5, 6, 7):
            np.testing.assert_array_equal(a[i], b[i])
        for x, y in zip(a[4] + a[8], b[4] + b[8]):
            np.testing.assert_array_equal(x, y)
        # The segment table: the same (expert, rows, source, destination)
        # runs per destination, in the new order.
        sa, sb = a[0].reshape(3, g, ep), b[0].reshape(3, g, ep)
        for d in range(ep):
            ra = sorted(zip(sa[0, :, d] & 0xFFFFFF, sa[1, :, d], sa[2, :, d]))
            rb = sorted(zip(sb[0, :, d] & 0xFFFFFF, sb[1, :, d], sb[2, :, d]))
            assert ra == rb


# --------------------------------------------------------------------------- #
# The breakdown harness's D/DZ tokens.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bd():
    path = str(TREE / "benchmark" / "tpu")
    if path not in sys.path:
        sys.path.insert(0, path)
    import dsv41_k32z_breakdown

    return dsv41_k32z_breakdown


def test_harness_push_tokens(bd, v2):
    assert bd.push_knobs_of("DZ") == {}
    assert bd.push_knobs_of("DZo1h0t1d1") == dict(
        visit_order=1, hbm_staging=False, two_hop=True, defer_push=True
    )
    assert bd.push_knobs_of("Dn4a16z1") == dict(
        sc_prepush=4, dispatch_ahead=16, fold_select=True
    )
    assert bd.push_knobs_of("DZk2") == dict(ret_last_off=2)
    assert bd.push_knobs_of("DZc1") == dict(commit_only=True)
    # On D/DZ "d" is the push kernel's deferred push, not v2's DEFER_PUSH.
    assert bd.knobs_of("DZd1") == {}
    assert bd.knobs_of("AZd1") == {"DEFER_PUSH": True}
    assert bd.knobs_of("DZg128d1") == {"K32Z_DECODE_ROLL": (2, 128)}
    assert bd.config_exact("DZo1h0") and not bd.config_exact("DZk1")
    assert not bd.config_exact("DZc1")
    assert set(o for o, _ in bd.PUSH_KNOBS.values()) <= set(v2.push_options)
    for bad in ("AZo1", "Ah0", "DZs32", "DZp128", "DZl0", "DZq"):
        with pytest.raises(ValueError):
            bd.push_knobs_of(bad)
