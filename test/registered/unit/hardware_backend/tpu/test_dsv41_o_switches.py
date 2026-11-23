# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Step 7 O switches.

* host.COMBINE_FUSE_SCALES (all-gather builds): forced on, the combine kernel
  is push's fused-scale build (`_p4_fs`, with the select fold's `_sel` where
  that is on) and the rest of the program's kernels keep their names; left
  unset, a K32Z build takes it exactly at host.COMBINE_FUSE_SCALES_TOKENS
  rows per rank (the text of it forced on there, of it forced off elsewhere)
  and the K512 (FP4) build never does.
* The push option x_flat_owner: the receiver's owner tables (dedup) are
  covered against the model in test_dsv41_push_dzx (flat and reshaped); here
  its program differs from the default one only where the dedup applies.
* The breakdown harness's O tokens: AZi<n>, DZF<n>.

Nothing here needs a TPU: the layers lower for a TPU7x target on two CPU
devices (the K2f defaults tests' shapes).

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

from test_dsv41_push_k2_knobs import (  # noqa: E402,F401
    _lower_push,
    _names,
    _tuned_push_options,
    bd,
    v2,
)
from test_dsv41_push_k2f_defaults import _lower_all_gather  # noqa: E402


def _lower_ag(v2, *, fuse=None, **kw):
    """_lower_all_gather with host.COMBINE_FUSE_SCALES set to `fuse` (None:
    left as it is)."""
    host = v2.host
    saved = host.COMBINE_FUSE_SCALES
    if fuse is not None:
        host.COMBINE_FUSE_SCALES = fuse
    try:
        return _lower_all_gather(v2, **kw)
    finally:
        host.COMBINE_FUSE_SCALES = saved


@pytest.mark.parametrize("fmt", ["k32z", "fp4"])
def test_all_gather_fuse_scales_forced(v2, fmt):
    lower = partial(_lower_ag, v2, t_local=256, fmt=fmt)
    off = lower(fuse=False)
    on = lower(fuse=True)
    assert on != off
    plain = _names(off, "moe_v2_combine")
    fused = _names(on, "moe_v2_combine")
    assert plain and not any("_fs" in n for n in plain), plain
    assert fused and all("_p4_fs" in n for n in fused), fused
    # The select fold keeps its own token on the fused build.
    assert [n.endswith("_sel") for n in fused] == [n.endswith("_sel") for n in plain]
    assert _names(on, "fused_ep_moe_v2") == _names(off, "fused_ep_moe_v2")


@pytest.mark.parametrize("fmt, t_local", [("k32z", 256), ("k32z", 64), ("fp4", 256)])
def test_all_gather_fuse_scales_default(v2, fmt, t_local):
    """Left unset: on exactly for K32Z at COMBINE_FUSE_SCALES_TOKENS."""
    host = v2.host
    assert host.COMBINE_FUSE_SCALES is None
    on = fmt == "k32z" and t_local in host.COMBINE_FUSE_SCALES_TOKENS
    assert host.combine_fuse_scales(None, k32z=fmt == "k32z", t_local=t_local) == on
    lower = partial(_lower_ag, v2, t_local=t_local, fmt=fmt)
    assert lower() == lower(fuse=on)


def test_combine_fuse_scales_rule(v2):
    host = v2.host
    for t in (8, 64, 256, 512, 1024, 2048):
        assert host.combine_fuse_scales(True, k32z=False, t_local=t)
        assert not host.combine_fuse_scales(False, k32z=True, t_local=t)
        assert host.combine_fuse_scales(None, k32z=False, t_local=t) is False
        assert host.combine_fuse_scales(None, k32z=True, t_local=t) == (
            t in host.COMBINE_FUSE_SCALES_TOKENS
        )


def test_harness_o_tokens(bd):
    assert bd.knobs_of("AZi1") == {"COMBINE_FUSE_SCALES": 1}
    assert bd.knobs_of("AZi0") == {"COMBINE_FUSE_SCALES": 0}
    assert bd.push_knobs_of("AZi1") == {}
    assert bd.push_knobs_of("DZF1") == {"x_flat_owner": True}
    assert bd.push_knobs_of("DZF0") == {"x_flat_owner": False}
    assert bd.base_config("DZF1") == "DZ" and bd.base_config("DF1") == "D"
    assert bd.config_exact("AZi1") and bd.config_exact("DZF1")
    for bad in ("DZi1", "Di1", "AZF1", "AF1"):
        with pytest.raises(ValueError):
            bd.knobs_of(bad) if bad.startswith("D") else bd.push_knobs_of(bad)


def test_all_gather_quantize_first(v2):
    """host.QUANTIZE_FIRST forced on: the same kernels, another program (the
    barrier); left unset it follows host.quantize_first's per-build rule."""
    host = v2.host
    assert host.QUANTIZE_FIRST is None
    saved = host.QUANTIZE_FIRST
    try:
        host.QUANTIZE_FIRST = True
        on = _lower_all_gather(v2, t_local=256)
        host.QUANTIZE_FIRST = False
        off = _lower_all_gather(v2, t_local=256)
    finally:
        host.QUANTIZE_FIRST = saved
    assert on != off
    for name in ("fused_ep_moe_v2", "moe_v2_combine"):
        assert _names(on, name) == _names(off, name)
    assert _lower_all_gather(v2, t_local=256) == (
        on if host.quantize_first(None, k32z=True, t_local=256) else off
    )
    for t in (8, 256, 2048):
        assert host.quantize_first(None, k32z=False, t_local=t) is False
        assert host.quantize_first(None, k32z=True, t_local=t) == (
            t in host.QUANTIZE_FIRST_TOKENS
        )


def test_harness_quantize_first_token(bd):
    assert bd.knobs_of("AZQ1") == {"QUANTIZE_FIRST": 1}
    assert bd.base_config("AZQ1") == "AZ" and bd.base_config("AQ1") == "A"
    with pytest.raises(ValueError):
        bd.knobs_of("DZQ1")


def test_all_gather_token_ag_core(v2):
    """host.TOKEN_AG_CORE = 1: the token all-gather is a SparseCore
    compute_on call (another program, the same kernels); left unset it
    follows host.token_ag_core's per-build rule."""
    host = v2.host
    assert host.TOKEN_AG_CORE is None
    saved = host.TOKEN_AG_CORE
    try:
        host.TOKEN_AG_CORE = 1
        on = _lower_all_gather(v2, t_local=256)
        host.TOKEN_AG_CORE = 0
        off = _lower_all_gather(v2, t_local=256)
    finally:
        host.TOKEN_AG_CORE = saved
    assert on != off
    for name in ("fused_ep_moe_v2", "moe_v2_combine"):
        assert _names(on, name) == _names(off, name)
    assert _lower_all_gather(v2, t_local=256) == (
        on if host.token_ag_core(None, k32z=True, t_local=256) else off
    )
    for t in (8, 256, 2048):
        assert host.token_ag_core(None, k32z=False, t_local=t) == 0
        assert host.token_ag_core(None, k32z=True, t_local=t) == (
            1 if t in host.TOKEN_AG_CORE_TOKENS else 0
        )
    with pytest.raises(ValueError):
        host.token_ag_core(2, k32z=True, t_local=2048)


def test_harness_token_ag_core_token(bd):
    assert bd.knobs_of("AZC1") == {"TOKEN_AG_CORE": 1}
    assert bd.knobs_of("AZi1C1") == {"COMBINE_FUSE_SCALES": 1, "TOKEN_AG_CORE": 1}
    with pytest.raises(ValueError):
        bd.knobs_of("DZC1")


def test_all_gather_sc_side_core(v2):
    """host.SC_SIDE_CORE 1 (metadata all-gather + slab scatters) and 2 (the
    scatters only): SparseCore compute_on calls (other programs, the same
    kernels); neither with the token all-gather on core 1; left unset it
    follows host.sc_side_core's per-build rule."""
    host = v2.host
    assert host.SC_SIDE_CORE is None
    saved = host.SC_SIDE_CORE, host.TOKEN_AG_CORE
    texts = {}
    try:
        for mode in (0, 1, 2):
            host.SC_SIDE_CORE = mode
            texts[mode] = _lower_all_gather(v2, t_local=256)
        host.SC_SIDE_CORE, host.TOKEN_AG_CORE = 1, 1
        both = _lower_all_gather(v2, t_local=256)
        host.SC_SIDE_CORE = 0
        core1 = _lower_all_gather(v2, t_local=256)
    finally:
        host.SC_SIDE_CORE, host.TOKEN_AG_CORE = saved
    assert len(set(texts.values())) == 3
    assert both == core1
    for mode in (1, 2):
        for name in ("fused_ep_moe_v2", "moe_v2_combine"):
            assert _names(texts[mode], name) == _names(texts[0], name)
    default = host.sc_side_core(None, k32z=True, t_local=256)
    assert _lower_all_gather(v2, t_local=256) == texts[default]
    for t in (8, 256, 2048):
        assert host.sc_side_core(None, k32z=False, t_local=t) == 0
        assert host.sc_side_core(None, k32z=True, t_local=t) == (
            host.SC_SIDE_CORE_DEFAULT if t in host.SC_SIDE_CORE_TOKENS else 0
        )
    with pytest.raises(ValueError):
        host.sc_side_core(3, k32z=True, t_local=2048)


def test_harness_sc_side_token(bd):
    assert bd.knobs_of("AZM1") == {"SC_SIDE_CORE": 1}
    assert bd.knobs_of("AZi1M1Q1") == {
        "COMBINE_FUSE_SCALES": 1,
        "SC_SIDE_CORE": 1,
        "QUANTIZE_FIRST": 1,
    }
    with pytest.raises(ValueError):
        bd.knobs_of("DZM1")


def test_push_flat_owner_default_1024(v2):
    """K32Z push on the tuned bundle at 1024 rows per rank: left unset,
    x_flat_owner is on (the text of it set explicitly); set off, another
    program; at 512 (no dedup) it changes nothing."""
    k32z = v2.host.WeightFormat.K32Z
    tuned = _tuned_push_options(8)
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=1024)
    default = lower(push_options=tuned)
    assert default == lower(push_options={**tuned, "x_flat_owner": True})
    assert default != lower(push_options={**tuned, "x_flat_owner": False})
    lower = partial(_lower_push, v2, weight_format=k32z, t_local=512)
    assert lower(push_options=tuned) == lower(
        push_options={**tuned, "x_flat_owner": True}
    )
