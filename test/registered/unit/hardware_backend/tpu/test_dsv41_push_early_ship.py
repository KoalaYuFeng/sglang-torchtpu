# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Step 11 E1 push option early_ship (by default 3 for a
K32Z build at 1024 rows per rank, 0 everywhere else).

* Default: early_ship=0 lowers to the very text of the bundle without it
  (K32Z, staged, with and without defer_push).
* Name: early_ship=1 (2, 3, 4) on the two-hop path puts `_es` (`_es2`, `_es3`,
  `_es4`) in the push kernel's name and lowers to its own text; where the
  build ships from `out_vm` (hbm_staging off at a one-tile shape) or under
  ret_off it changes nothing.
* The schedule: on random run tables, the tile-loop predicate (a nonempty
  run whose last tile is t - 2, at tile t >= 2) and push_expert's (last
  tile >= n_tiles - 2) ship every nonempty run exactly once, and an
  in-loop ship reads only tiles whose commits the reuse guards have landed.
* early_ship=2's pieces of a run cover its true rows, aligned rows and
  mirror units once each.
* The host default (host.K32Z_PUSH_EARLY_SHIP, on: 1024 only, mode 3):
  per-bucket modes for an unset early_ship, nothing at an unlisted bucket or
  with the knob off, an explicit value wins.
* The breakdown harness's token: DZP1 is {"early_ship": 1}; AZP1 is refused.

Nothing here needs a TPU (lowering for a TPU7x target on CPU devices).

    PYTHONPATH=<tree>/python:<tree>:<this dir> python -m pytest -q <this file>
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from test_dsv41_push_k2_knobs import (  # noqa: F401  (v2 is a fixture)
    _lower_push,
    _names,
    _tuned_push_options,
    v2,
)

TREE = Path(__file__).resolve().parents[5]


def _k32z_lower(v2, **kw):
    return partial(_lower_push, v2, weight_format=v2.host.WeightFormat.K32Z, **kw)


@pytest.mark.parametrize("defer", [False, True], ids=["staged", "staged_dp"])
def test_early_ship_off_is_the_default_program(v2, defer):
    lower = _k32z_lower(v2, t_local=256)
    opts = {**_tuned_push_options(8), "defer_push": defer}
    base = lower(push_options=opts)
    assert lower(push_options={**opts, "early_ship": 0}) == base
    assert not any("_es" in n for n in _names(base, "push_fused_moe_"))


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
@pytest.mark.parametrize("defer", [False, True], ids=["staged", "staged_dp"])
def test_early_ship_names_its_program(v2, defer, mode):
    lower = _k32z_lower(v2, t_local=256)
    opts = {**_tuned_push_options(8), "defer_push": defer}
    base = lower(push_options=opts)
    es = lower(push_options={**opts, "early_ship": mode})
    assert es != base
    (name,) = _names(base, "push_fused_moe_")
    assert "_hbmst" in name
    tag = "_es" if mode == 1 else f"_es{mode}"
    assert _names(es, "push_fused_moe_") == [name + tag]


def test_early_ship_is_off_where_tiles_ship_from_vmem(v2):
    """hbm_staging off at this one-tile shape ships tiles from out_vm (no
    two-hop push to move): early_ship builds the same program."""
    lower = _k32z_lower(v2)
    opts = {**_tuned_push_options(8), "hbm_staging": False}
    ship = lower(push_options=opts)
    for mode in (1, 2, 3, 4):
        assert lower(push_options={**opts, "early_ship": mode}) == ship


def test_early_ship_is_off_under_ret_off(v2):
    lower = _k32z_lower(v2, t_local=256, ablation=frozenset({"ret_off"}))
    opts = _tuned_push_options(8)
    base = lower(push_options=opts)
    for mode in (1, 2, 3, 4):
        assert lower(push_options={**opts, "early_ship": mode}) == base


def test_early_ship_refuses_other_modes(v2):
    lower = _k32z_lower(v2, t_local=256)
    with pytest.raises(ValueError, match="early_ship"):
        lower(push_options={**_tuned_push_options(8), "early_ship": 5})


def _schedule(starts, lens, n_tiles, tile_blocks, lag=2):
    """(the tile each nonempty run ships in, or 'end') as the kernel's two
    predicates give it; raises if a run ships twice."""
    shift = int(np.log2(tile_blocks))
    when = {}
    for t in range(n_tiles):
        if t < lag:
            continue
        for d, (s, n) in enumerate(zip(starts, lens)):
            if n > 0 and ((s + n - 1) >> shift) == t - lag:
                assert d not in when
                when[d] = t
    for d, (s, n) in enumerate(zip(starts, lens)):
        if n > 0 and ((s + n - 1) >> shift) >= n_tiles - lag:
            assert d not in when
            when[d] = "end"
    return when


def test_early_ship_schedule_ships_each_run_once_after_its_commits():
    rng = np.random.default_rng(11)
    tile_blocks = 128  # 256-row tiles of ROWBLK = 2 rows
    for _ in range(2000):
        ep = 8
        lens = rng.integers(0, 6 * tile_blocks, size=ep) * rng.integers(0, 2, size=ep)
        if lens.sum() == 0:
            continue
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        n_tiles = -(-int(lens.sum()) // tile_blocks)
        when = _schedule(starts, lens, n_tiles, tile_blocks)
        assert set(when) == {d for d in range(ep) if lens[d] > 0}
        for d, t in when.items():
            last = (starts[d] + lens[d] - 1) // tile_blocks
            if t == "end":
                assert last >= n_tiles - 2
            else:
                # The guard of tile t has landed every tile <= t - 2.
                assert last <= t - 2 and t < n_tiles


def test_breakdown_token():
    sys.path.insert(0, str(TREE / "benchmark" / "tpu"))
    try:
        import dsv41_k32z_breakdown as bd
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"the breakdown harness does not import here: {exc!r}")
    assert bd.push_knobs_of("DZP1") == {"early_ship": 1}
    assert bd.push_knobs_of("DZ") == {}
    assert bd.config_exact("DZP1")
    assert bd.knobs_of("DZG1") == {"K32Z_PUSH_EARLY_SHIP": 1}
    assert bd.push_knobs_of("DZG1") == {}
    with pytest.raises(ValueError):
        bd.knobs_of("AZG1")
    with pytest.raises(ValueError):
        bd.push_knobs_of("AZP1")


def _pieces(start, length, true_rows, n_tiles, tile_blocks, unit_blocks, lag=2):
    """early_ship=2's pieces of one run (start, length in blocks): the tile
    loop's (tile t - lag at t = lag .. n_tiles - 1) then push_expert's rest,
    as (rows into the run, true rows, aligned rows, first unit, units)."""
    rowblk = 2
    out = []
    bounds = [(b * tile_blocks, (b + 1) * tile_blocks) for b in range(n_tiles - lag)]
    bounds.append(((n_tiles - lag) * tile_blocks, None))
    for blk_lo, blk_hi in bounds:
        lo = max(start, blk_lo)
        hi = start + length if blk_hi is None else min(start + length, blk_hi)
        if hi - lo <= 0:
            continue
        into = (lo - start) * rowblk
        n_true = min(max(true_rows - into, 0), (hi - lo) * rowblk)
        u_lo = lo // unit_blocks
        n_units = (hi - 1) // unit_blocks - u_lo + 1
        out.append((into, n_true, (hi - lo) * rowblk, u_lo, n_units))
    return out


def test_early_ship_pieces_partition_each_run():
    """The pieces of every run cover its true rows, its aligned rows and its
    mirror units once each, in order: the bytes _push_run moves."""
    rng = np.random.default_rng(12)
    tile_blocks, unit_blocks = 128, 64  # 256-row tiles, 128-row mirror units
    for _ in range(3000):
        ep = 8
        true = rng.integers(0, 6 * 256, size=ep) * rng.integers(0, 2, size=ep)
        lens = (true + 1) // 2
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        if lens.sum() == 0:
            continue
        n_tiles = -(-int(lens.sum()) // tile_blocks)
        for d in range(ep):
            ps = _pieces(int(starts[d]), int(lens[d]), int(true[d]), n_tiles,
                         tile_blocks, unit_blocks)
            if lens[d] == 0:
                assert ps == []
                continue
            rows = [r for p in ps for r in range(p[0], p[0] + p[2])]
            assert rows == list(range(2 * int(lens[d])))
            trues = [r for p in ps for r in range(p[0], p[0] + p[1])]
            assert trues == list(range(int(true[d])))
            units = [u for p in ps for u in range(p[3], p[3] + p[4])]
            u0 = int(starts[d]) // unit_blocks
            u1 = (int(starts[d]) + int(lens[d]) - 1) // unit_blocks
            assert units == list(range(u0, u1 + 1))


def test_host_default_is_1024_only_and_per_bucket(v2, monkeypatch):
    """host.K32Z_PUSH_EARLY_SHIP (default on, 1024 only: mode 3) gives an
    unset early_ship the mode its bucket lists (K32Z_PUSH_EARLY_SHIP_MODES);
    off, or at a bucket it does not list, the program is the bundle's; an
    explicit early_ship wins (0 turns it off, 1 still builds 512's mode)."""
    host = v2.host
    assert host.K32Z_PUSH_EARLY_SHIP
    assert host.K32Z_PUSH_EARLY_SHIP_MODES == {1024: 3}
    mode = partial(host.push_early_ship_mode, host.K32Z_PUSH_EARLY_SHIP)
    buckets = (8, 16, 32, 64, 512, 1024, 2048)
    assert [mode(k32z=True, t_local=t) for t in buckets] == [0, 0, 0, 0, 0, 3, 0]
    assert mode(k32z=False, t_local=1024) == 0
    assert host.push_early_ship_mode(0, k32z=True, t_local=1024) == 0
    assert host.push_early_ship_mode(None, k32z=True, t_local=1024) == 0
    lower = _k32z_lower(v2, t_local=256)
    opts = _tuned_push_options(8)
    base = lower(push_options={**opts, "early_ship": 0})
    assert lower(push_options=opts) == base  # 256 is not listed
    monkeypatch.setattr(host, "K32Z_PUSH_EARLY_SHIP_MODES", {256: 3})
    on = lower(push_options=opts)
    assert on == lower(push_options={**opts, "early_ship": 3})
    (name,) = _names(base, "push_fused_moe_")
    assert _names(on, "push_fused_moe_") == [name + "_es3"]
    assert lower(push_options={**opts, "early_ship": 0}) == base
    monkeypatch.setattr(host, "K32Z_PUSH_EARLY_SHIP", 0)
    assert lower(push_options=opts) == base
