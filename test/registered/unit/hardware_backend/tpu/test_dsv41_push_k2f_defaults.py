# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Step 7 K2f defaults.

* The K32Z push path's per-bucket defaults (push layer K32Z_PUSH_DEFAULTS):
  on #2102's tuned bundle (the exchange-table path) the options left unset
  lower to the very text of the options set explicitly -- 512 rows per rank
  x_tables + fold_select, 1024 x_dedup + defer_push + fold_select -- with
  their name tokens; setting them off lowers to k32z-r2's program (no K2
  token); the K512-form (FP4) push build, the kernel-default bundle, and the
  other buckets keep the all-off program.
* The all-gather build's select fold (host.COMBINE_FOLD_SELECT): forced on
  it puts `_sel` on the combine kernel and nowhere else; left unset (Step 7
  K3) it is on for a K32Z build where the combine kernel applies and off
  (the text of the fold forced off) for the K512 (FP4) build and for K32Z
  below the combine kernel's rows per rank.
* The breakdown harness's K2f tokens: DZu0v0d0z0 (k32z-r2's DZ), AZy1, AZy0.

Nothing here needs a TPU: the layer lowers for a TPU7x target on two CPU
devices (test_dsv41_push_k2_knobs._lower_push's shapes).

    PYTHONPATH=<tree>/python:<tree>:<this dir> python -m pytest -q <this file>
"""

import os
import re
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


register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
pytest.importorskip("ml_dtypes")
pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from test_dsv41_push_k2_knobs import (  # noqa: E402,F401
    _lower_push,
    _names,
    _tpu7x_target,
    _tuned_push_options,
    bd,
    v2,
)

# The options the K2f defaults set, all off: k32z-r2's program.
R2_OFF = dict(x_tables=False, x_dedup=False, defer_push=False, fold_select=False)
K2F = {
    512: dict(x_tables=True, fold_select=True),
    1024: dict(x_dedup=True, defer_push=True, fold_select=True),
}
# Name tokens of the K2 switches (push kernel, table kernel, combine).
K2_TOKEN_RE = re.compile(r"_(xt|xd\d*|dp|vo1|sel)(_|$)")


@pytest.fixture(scope="module")
def push_layer(v2):
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer

    return layer


def _k2_names(text):
    names = (
        _names(text, "push_fused_moe_")
        + _names(text, "moe_tc_routing_exchange")
        + _names(text, "moe_v2_combine")
    )
    return sorted(n for n in names if K2_TOKEN_RE.search(n))


def test_k32z_push_defaults_table(push_layer):
    # Step 7 O adds x_flat_owner where the dedup is (1024).
    want = {512: K2F[512], 1024: {**K2F[1024], "x_flat_owner": True}}
    assert push_layer.K32Z_PUSH_DEFAULTS == want


@pytest.mark.parametrize("t_local", [512, 1024])
def test_k32z_push_defaults_lower_as_set(v2, t_local):
    """On the tuned bundle the unset options lower to the explicit K2f set;
    all off lowers to k32z-r2's program (no K2 name token)."""
    k32z = v2.host.WeightFormat.K32Z
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=t_local)
    tuned = _tuned_push_options(8)
    default = lower(push_options=tuned)
    assert lower(push_options={**tuned, **K2F[t_local]}) == default
    r2 = lower(push_options={**tuned, **R2_OFF})
    assert r2 != default
    assert not _k2_names(r2), _k2_names(r2)
    names = " ".join(_k2_names(default))
    assert "_sel" in names, names
    if t_local == 512:
        assert "_xt" in names and "_xd" not in names and "_dp" not in names, names
    else:
        assert "_xd" in names and "_dp" in names and "_vo1" in names, names


@pytest.mark.parametrize("t_local", [16, 256, 512])
def test_k2f_defaults_leave_other_builds_alone(v2, t_local):
    """The K512-form push build at 512, the kernel-default bundle, and K32Z
    at buckets the table does not list lower to the all-off program."""
    k32z, fp4 = v2.host.WeightFormat.K32Z, v2.host.WeightFormat.FP4
    tuned = _tuned_push_options(8)
    if t_local == 512:
        lower = partial(_lower_push, v2, weight_format=fp4, t_local=512, inter=1024)
        assert lower(push_options=tuned) == lower(push_options={**tuned, **R2_OFF})
        lower = partial(_lower_push, v2, weight_format=k32z, t_local=512)
        plain = lower(push_options={})
        assert plain == lower(push_options=dict(R2_OFF))
        assert not _k2_names(plain), _k2_names(plain)
        return
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=t_local)
    default = lower(push_options=tuned)
    assert default == lower(push_options={**tuned, **R2_OFF})
    assert not _k2_names(default), _k2_names(default)


def test_k2f_ret_last_off_keeps_the_dedup_off(v2):
    """ret_last_off (benchmark only) at 1024 builds without the defaulted
    dedup instead of refusing it."""
    k32z = v2.host.WeightFormat.K32Z
    tuned = _tuned_push_options(8)
    text = _lower_push(
        v2,
        weight_format=k32z,
        t_local=1024,
        push_options={**tuned, "ret_last_off": 1},
    )
    names = " ".join(_names(text, "push_fused_moe_"))
    assert "_rlo1" in names and "_xd" not in names, names


# --------------------------------------------------------------------------- #
# The all-gather build's select fold.
# --------------------------------------------------------------------------- #
def _lower_all_gather(v2, *, t_local, fold=None, fmt="k32z", ep=2, g_local=8, bits=2):
    """The all-gather layer at test shapes -- fmt "k32z": config AZ's build,
    "fp4": the K512 form (config A's; its 512-row scale blocks divide both
    matmuls' contraction sizes, hence its own intermediate size) -- with
    host.COMBINE_FOLD_SELECT set to `fold` (None: left as it is)."""
    host = v2.host
    hidden, inter = 1536, (768 if fmt == "k32z" else 1024)
    mesh = Mesh(np.asarray(jax.devices()[:ep]), ("d",))
    rows, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    e_total = ep * g_local
    t = ep * t_local
    if fmt == "k32z":
        geo = host.k32z_table_geometry(hidden, inter, bits)
        s1 = jax.ShapeDtypeStruct(
            (e_total * geo.rows1, geo.cols1), jnp.uint8, sharding=rows
        )
        s2 = jax.ShapeDtypeStruct(
            (e_total * geo.rows2, geo.cols2), jnp.uint8, sharding=rows
        )
        cap, wf = 256, host.WeightFormat.K32Z
    else:
        r1 = host.fp4_scale_rows(hidden, 512, padded=True)
        r2 = host.fp4_scale_rows(inter, 512, padded=True)
        s1 = jax.ShapeDtypeStruct((e_total * r1, 2 * inter), jnp.float32, sharding=rows)
        s2 = jax.ShapeDtypeStruct((e_total * r2, hidden), jnp.float32, sharding=rows)
        cap, wf = 128, host.WeightFormat.FP4
    w1 = jax.ShapeDtypeStruct(
        (e_total, hidden, 2 * inter), jnp.float4_e2m1fn, sharding=rows
    )
    w2 = jax.ShapeDtypeStruct(
        (e_total, inter, hidden), jnp.float4_e2m1fn, sharding=rows
    )
    x = jax.ShapeDtypeStruct((t, hidden), jnp.bfloat16, sharding=rows)
    g = jax.ShapeDtypeStruct((t, e_total), jnp.float32, sharding=rows)
    bias = jax.ShapeDtypeStruct((e_total,), jnp.float32, sharding=repl)

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
            weight_format=wf,
            rhs_qb=512,
            act_fn="silu_and_mul_with_clamp",
            mesh_ep_ranks=tuple(range(ep)),
            sharded_plan=True,
            scoring_fn="sqrtsoftplus",
            score_bias=bias,
            routed_scaling_factor=1.5,
        )

    saved = host.COMBINE_FOLD_SELECT
    if fold is not None:
        host.COMBINE_FOLD_SELECT = fold
    try:
        with _tpu7x_target():
            return (
                jax.jit(layer)
                .trace(x, g, w1, w2, s1, s2, bias)
                .lower(lowering_platforms=("tpu",))
                .as_text()
            )
    finally:
        host.COMBINE_FOLD_SELECT = saved


def test_all_gather_fold_select(v2):
    """Forced off, no `_sel` (k32z-r2's program); forced on, the combine
    kernel takes the route flags (`_sel`) and the rest of the program's
    kernels keep their names."""
    lower = partial(_lower_all_gather, v2, t_local=256)
    off = lower(fold=False)
    folded = lower(fold=True)
    assert folded != off
    plain = _names(off, "moe_v2_combine")
    sel = _names(folded, "moe_v2_combine")
    assert plain and not any(n.endswith("_sel") for n in plain), plain
    assert sel == [n + "_sel" for n in plain], sel
    assert _names(folded, "fused_ep_moe_v2") == _names(off, "fused_ep_moe_v2")


@pytest.mark.parametrize(
    "fmt, t_local, on", [("k32z", 256, True), ("k32z", 64, False), ("fp4", 256, False)]
)
def test_all_gather_fold_select_default(v2, fmt, t_local, on):
    """Step 7 K3: left unset, the fold is on for a K32Z build where the
    combine kernel applies (the text of the fold forced on) and off for the
    K512 (FP4) build and below the combine kernel's rows per rank (the text
    of the fold forced off: those programs are unchanged)."""
    assert v2.host.COMBINE_FOLD_SELECT is None
    lower = partial(_lower_all_gather, v2, t_local=t_local, fmt=fmt)
    default = lower()
    assert default == lower(fold=on)
    has_sel = any(n.endswith("_sel") for n in _names(default, "moe_v2_combine"))
    assert has_sel == on
    if fmt == "fp4":
        # The K512 build has the combine kernel here too: the fold is only
        # left off, not unavailable.
        assert lower(fold=True) != default


def test_harness_k2f_tokens(bd):
    assert bd.push_knobs_of("DZu0v0d0z0") == R2_OFF
    assert bd.knobs_of("AZy1") == {"COMBINE_FOLD_SELECT": 1}
    assert bd.knobs_of("AZy0") == {"COMBINE_FOLD_SELECT": 0}
    assert bd.knobs_of("Ay0") == {"COMBINE_FOLD_SELECT": 0}
    assert bd.push_knobs_of("AZy1") == {}
    assert bd.config_exact("DZu0v0d0z0") and bd.config_exact("AZy1")
    assert bd.config_exact("AZy0")
    for bad in ("DZy1", "Dy1"):
        with pytest.raises(ValueError):
            bd.push_knobs_of(bad)
