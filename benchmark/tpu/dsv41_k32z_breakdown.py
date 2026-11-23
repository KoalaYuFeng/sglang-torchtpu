#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 N3a: layer breakdown of A vs A+Z (one real MoE layer, EP8).

Programs per bucket: config (A = K512, AZ = K32Z; dsv41_k32z_l1.py) x
variant, each the whole jitted layer (routing -> output) built with the
kernel's benchmark-only ``ablation`` switches (Step 7 DESIGN §7):

    full  {}                                    T_full (production)
    comm  {w_off, ffn_off}                      T_comm
    comp  {disp_off, ret_off}                   T_comp
    base  {w_off, ffn_off, disp_off, ret_off}   T_base
    a2w   {ffn_off}    nowt {w_off}    ret {ret_off}    disp {disp_off}

disp_off variants run the layer's compute stage on the route stage's outputs
pre-gathered on every shard (token rows and routing payload), so no
collective runs before the kernel. ``disp`` ({disp_off} alone) is exact and
is checked bitwise against ``full``; every other ablation gives invalid
output by construction (timing only).

Per bucket and routing, ``--trials`` rounds. Each round times every program
in turn (warmups, ``--iters`` synchronous calls, ``--pipelined-iters``
back-to-back calls; wall clock), then runs one profiler session that calls
every program ``--trace-calls`` times round-robin. Each session is digested
per program (dsv41_k32z_trace: span and fused-kernel time max over cores, op
categories, op timeline) and then deleted unless its bucket is in
``--keep-traces`` (first round only). Device numbers pool the rounds' calls.

    PYTHONPATH=$T/python:$T python3 $T/benchmark/tpu/dsv41_k32z_breakdown.py \\
        --weights /weights/DeepSeek-V4.1-Flash --layer 3 --trace-dir prof \\
        --out bd.json
    python3 dsv41_k32z_breakdown.py --table bd.json [more.json ...]

Step 7 K2: D/DZ configs take push-option tokens (CONFIG_RE below; e.g.
DZh0t1d1, DZo1, DZk2), checked bitwise against their base config unless the
option is benchmark-only (ret_last_off, commit_only: rows equal to the base
are counted instead), and ``--ici-counters <K0's ici_k0.py>`` adds one
profiler session per program holding one call, digested into ICI link bytes.
"""

from __future__ import annotations

import os

# Never let `import torch` auto-load torch_tpu (it takes the libtpu lock).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import csv
import functools
import glob
import json
import re
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dsv41_k32z_trace as TR  # noqa: E402

# A config is A, AZ, D or DZ (dsv41_k32z_l1.CONFIGS), optionally followed by
# R1-opt knobs for its K32Z build, timed in the same process as the tree
# default the bare name builds: "r<u>" host.K32Z_FFN_ROLL = u, "m<n>"
# host.K32Z_MIRROR_UNIT = n, "q<h>" adds tile height h to host.K32Z_RUNGS and
# "x<h>" drops it, "f<h>" loops only rungs >= h (K32Z_FFN_ROLL_FROM; e.g.
# AZr2m128q96, AZx64, AZr1f64).
# Step 7 R2 (K1) adds: "s<h>" loops rung h (a set, K32Z_ROLL_RUNGS, with
# "r<u>"), "p<n>" PUSH_FROM_VMEM_ROWS = n (p0: None), "d<n>" DEFER_PUSH,
# "g<h>" K32Z_DECODE_ROLL = (2, h) (g0: None), "l<n>" K32Z_LARGE_PLAN =
# LARGE_PLANS[n] (l0: None); e.g. AZr2s32s64, AZp128d1, AZx64x128, DZg128,
# AZl0.
# Step 7 K2 adds push options for D and DZ (on top of #2102's bundle,
# push_fused_moe.PUSH_OPTIONS): "o<n>" visit_order, "h<n>" hbm_staging,
# "t<n>" two_hop, "d<n>" defer_push (for D/DZ "d" is the push kernel's
# deferred push, as "d" is v2's DEFER_PUSH for AZ), "n<n>" sc_prepush,
# "a<n>" dispatch_ahead, "z<n>" fold_select, and the benchmark-only "k<n>"
# ret_last_off and "c<n>" commit_only (invalid output: not checked against
# the base config); e.g. DZo1, DZh0, DZh0t1d1, DZn4, DZa16, DZz1, DZk2, DZc1.
# Step 7 K2d adds the dispatch dedup (push_fused_moe.dedup; applied at 256 to
# 1024 rows per rank, the default program elsewhere): "u<n>" x_tables (DZt =
# DZu1), "v<n>" x_dedup (DZX = DZv1; implies x_tables and the sc_then_id
# order), "w<n>" x_dedup_local, "e<n>" x_owner_visit (the fallback owner
# rule, with a counts exchange first), "b<n>" x_src_cap (SRC words; 0 = the
# default), and the benchmark-only "j<n>" x_gather_off (invalid output);
# e.g. DZu1, DZv1, DZv1h0, DZv1h0z1, DZv1w1, DZv1e1, DZv1b64, DZv1j1.
# Step 7 K2f: DZ's defaults per bucket (push layer K32Z_PUSH_DEFAULTS: 512
# x_tables + fold_select, 1024 x_dedup + defer_push + fold_select) apply to
# the options a config leaves unset, so the bare DZ is the K2f program and
# DZu0v0d0z0 is k32z-r2's at every bucket (checked bitwise against DZ).
# "y<n>" host.COMBINE_FOLD_SELECT (all-gather builds only: the zeroing
# select folded into the combine kernel; e.g. AZy1). Step 7 K3: left unset
# it is on for K32Z builds where the combine kernel applies, so the bare AZ
# is the folded program there and AZy0 is k32z-r2's (checked bitwise
# against AZ); A is unchanged. Step 7 O: "i<n>" host.COMBINE_FUSE_SCALES
# (all-gather builds only: the arrival-scale lookup inside the combine
# kernel, as push's fused-scale combine; e.g. AZi1, checked bitwise against
# AZ; left unset it takes host.combine_fuse_scales' per-build default).
# Step 7 O: "F<n>" the push option x_flat_owner (D/DZ, under the dedup:
# the receiver's owner tables folded on the flat pair tables; e.g. DZF1,
# checked bitwise against DZ). "Q<n>" host.QUANTIZE_FIRST (all-gather builds
# only: the wire rows quantized ahead of the router; e.g. AZQ1). "C<n>"
# host.TOKEN_AG_CORE (all-gather builds only: the token all-gather on
# SparseCore core n; e.g. AZC1). "E<n>" / "S<n>" the bundle's push options
# early_scale_scatter / slim_prepush_barrier (D/DZ; e.g. DZE0, DZS0).
# "M<n>" host.SC_SIDE_CORE (all-gather builds only: side SparseCore work on
# core 1, M1 the metadata all-gather + slab scatters, M2 the scatters; e.g.
# AZM2).
# Step 8 K8: "X0" (D/DZ) the routing exchange as an XLA all-gather instead of
# the TensorCore remote-DMA kernel (tc_exchange, exchange_row_gather and
# exchange_tables off; the local sort runs while the all-gather is in flight);
# e.g. DZX0, checked bitwise against DZ. Outside 512/1024 rows per rank only
# (the dedup defaults there need the exchange-table path).
# Step 9 K9: "U<n>" host.K32Z_DECODE_STEP, the (gate/up, down) output chunks
# per loop step of a decode build's looped bodies: two digits (U34 = (3, 4),
# U65 = (6, 5)), three for a two-digit down count (U910 = (9, 10)), U0 None;
# e.g. DZU34, AZU34, checked bitwise against the base config.
# Step 10 L4: "H<n>" host.K32Z_L4_SPLIT (all-gather builds at 2048 rows per
# rank only: the hot expert's rows split with the sibling TensorCore, H1 the
# busiest rank's chip pair; H0 None); e.g. AZH1, checked bitwise against AZ.
# Its gain is reported apart from the kernel-only gains (Step 8 rule).
# Step 10 follow-up: "R<n>" host.K32Z_EARLY_SHIP (all-gather builds at 2048 rows per
# rank: each own expert's runs ship two tiles after the tile that completes them;
# kernel-only); e.g. AZR1, AZH4R1, checked bitwise against AZ.
# Step 11 E1: "P<n>" the push option early_ship (D/DZ: each return run ships from
# inside its expert's tile loop two tiles after the tile that completes it, on the
# two-hop path; P2: its per-tile pieces; P3: only in each rank's last two visits; P4: issued after
# the next tile's fetch; kernel-only); e.g. DZP1, checked bitwise against DZ.
# "G<n>" host.K32Z_PUSH_EARLY_SHIP (D/DZ only: the per-bucket early_ship default,
# host.K32Z_PUSH_EARLY_SHIP_MODES, for a push build that leaves early_ship unset);
# on by default since Step 12 (1024: mode 3), so DZ is DZG1 and DZG0 is the old DZ.
# Step 12 P3: "K<n>" host.K32Z_ES_PALLAS (mode 2's pick in one Pallas kernel; P3's
# own tree named it "P", which E1's push option already takes here); e.g. AZR2K1
# (= P3's AZR2P1), checked bitwise against AZ / AZR2. "L<n>" host.K32Z_L4_PALLAS (the
# L4 decision in one Pallas kernel); e.g. AZH4L1, checked bitwise against AZ / AZH4.
# Step 12 P2: "Y<n>" the push option exchange_slab_rows (D/DZ: the plan's per-pair slab
# rows from the exchange-table kernel; exchange-table plan path without x_tables, else
# the default build); e.g. DZY1, checked bitwise against DZ.
TOKENS = "rfmqxspdglohtnazkcuvwebjyiFQCESMXUHRPGKLY"
CONFIG_RE = re.compile(rf"^(A|AZ|D|DZ)((?:[{TOKENS}]\d+)*)$")
TOKEN_RE = re.compile(rf"([{TOKENS}])(\d+)")
KNOBS = {
    "r": "K32Z_FFN_ROLL",
    "f": "K32Z_FFN_ROLL_FROM",
    "m": "K32Z_MIRROR_UNIT",
    "y": "COMBINE_FOLD_SELECT",
    "i": "COMBINE_FUSE_SCALES",
    "Q": "QUANTIZE_FIRST",
    "C": "TOKEN_AG_CORE",
    "M": "SC_SIDE_CORE",
    "H": "K32Z_L4_SPLIT",
    "R": "K32Z_EARLY_SHIP",
    "G": "K32Z_PUSH_EARLY_SHIP",
    "K": "K32Z_ES_PALLAS",
    "L": "K32Z_L4_PALLAS",
}
# K32Z_LARGE_PLAN presets the "l<n>" token names (host.k32z_build_plan).
LARGE_PLANS = {0: None}
# Push-option tokens (D and DZ only): token -> (option, type).
PUSH_KNOBS = {
    "o": ("visit_order", int),
    "h": ("hbm_staging", bool),
    "t": ("two_hop", bool),
    "d": ("defer_push", bool),
    "n": ("sc_prepush", int),
    "a": ("dispatch_ahead", int),
    "z": ("fold_select", bool),
    "k": ("ret_last_off", int),
    "c": ("commit_only", bool),
    "u": ("x_tables", bool),
    "v": ("x_dedup", bool),
    "w": ("x_dedup_local", bool),
    "e": ("x_owner_visit", bool),
    "b": ("x_src_cap", int),
    "j": ("x_gather_off", bool),
    "F": ("x_flat_owner", bool),
    "E": ("early_scale_scatter", int),
    "S": ("slim_prepush_barrier", bool),
    "P": ("early_ship", int),
    "Y": ("exchange_slab_rows", bool),
}
# Push options whose output is invalid by design (benchmark only).
INVALID_PUSH_OPTIONS = ("ret_last_off", "commit_only", "x_gather_off")


def base_config(c):
    m = CONFIG_RE.match(c)
    if not m:
        raise ValueError(f"unknown config {c!r}")
    return m.group(1)


def _tokens(c):
    m = CONFIG_RE.match(c)
    if not m:
        raise ValueError(f"unknown config {c!r}")
    tokens = TOKEN_RE.findall(m.group(2))
    push = m.group(1) in ("D", "DZ")
    if push and any(k in "splyiQCMHRKL" for k, _ in tokens):
        # The push layer reads none of these (all-gather v2 builds only).
        raise ValueError(f"config {c!r}: s/p/l/y/i/Q/C/M/H/R knobs are all-gather only")
    if not push and any(k in PUSH_KNOBS and k != "d" for k, _ in tokens):
        raise ValueError(f"config {c!r}: push-option tokens take D or DZ")
    if not push and any(k == "G" for k, _ in tokens):
        raise ValueError(f"config {c!r}: G (the push early-ship default) takes D or DZ")
    return push, tokens


def push_knobs_of(c):
    """{push option: value} a D/DZ config sets on top of #2102's bundle
    (empty: the bundle itself, and for A/AZ)."""
    push, tokens = _tokens(c)
    if not push:
        return {}
    out = {}
    for k, v in tokens:
        if k in PUSH_KNOBS:
            option, kind = PUSH_KNOBS[k]
            out[option] = kind(int(v))
        elif k == "X" and not int(v):
            out.update(tc_exchange=False, exchange_row_gather=False, exchange_tables=False)
    return out


def config_exact(c):
    """Whether config c's output is valid (no benchmark-only push option)."""
    return not any(push_knobs_of(c).get(o) for o in INVALID_PUSH_OPTIONS)


def knobs_of(c):
    """{host attribute: value} a config forces (empty: the tree defaults)."""
    push, tokens = _tokens(c)
    if push:
        # "d" is the push kernel's deferred push there (push_knobs_of).
        tokens = [(k, v) for k, v in tokens if k not in PUSH_KNOBS]
    out = {KNOBS[k]: int(v) for k, v in tokens if k in KNOBS}
    extra = {int(v) for k, v in tokens if k == "q"}
    drop = {int(v) for k, v in tokens if k == "x"}
    if extra or drop:
        import sglang_tpu.moe.pallas.fused_moe_v2.host as host

        rungs = (set(host.K32Z_DEFAULT_RUNGS) | extra) - drop
        out["K32Z_RUNGS"] = tuple(sorted(rungs))
    rolled = {int(v) for k, v in tokens if k == "s"}
    if rolled:
        if not out.get("K32Z_FFN_ROLL"):
            raise ValueError(f"config {c!r}: s<h> loops rungs only with r<u> > 0")
        out["K32Z_ROLL_RUNGS"] = tuple(sorted(rolled))
    for k, v in tokens:
        v = int(v)
        if k == "p":
            out["PUSH_FROM_VMEM_ROWS"] = v or None
        elif k == "d":
            out["DEFER_PUSH"] = bool(v)
        elif k == "g":
            out["K32Z_DECODE_ROLL"] = (2, v) if v else None
        elif k == "l":
            out["K32Z_LARGE_PLAN"] = LARGE_PLANS[v]
        elif k == "U":
            out["K32Z_DECODE_STEP"] = (
                None if not v else ((v // 10, v % 10) if v < 100 else (v // 100, v % 100))
            )
    return out


def uses_compute_stage(c, v):
    """disp_off on an all-gather config runs the layer's compute stage on
    pre-gathered inputs; the push kernel takes disp_off as its own switch."""
    return "disp_off" in VARIANTS[v] and base_config(c) not in ("D", "DZ")


VARIANTS = {
    "full": (),
    "comm": ("w_off", "ffn_off"),
    "comp": ("disp_off", "ret_off"),
    "base": ("w_off", "ffn_off", "disp_off", "ret_off"),
    "a2w": ("ffn_off",),
    "nowt": ("w_off",),
    "ret": ("ret_off",),
    "disp": ("disp_off",),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table", nargs="+", default=None, help="print tables of results")
    p.add_argument(
        "--split",
        nargs="+",
        default=None,
        help="OLD:NEW[,OLD:NEW...] then result files: where NEW's gain over "
        "OLD comes from (R1-opt source split)",
    )
    p.add_argument(
        "--ops", nargs="+", default=None, help="print A vs AZ op phases of results"
    )
    p.add_argument("--ops-variant", default="full")
    p.add_argument(
        "--drift",
        nargs="+",
        metavar="STEP1_CSV RESULT",
        default=None,
        help="config A vs the Step 1 breakdown (step1_breakdown.csv)",
    )
    p.add_argument("--weights")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--configs", default="A,AZ")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--buckets", default="8,16,32,64,512,1024,2048")
    p.add_argument("--routings", default="uniform,real")
    p.add_argument("--pad-rows", type=int, default=2)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--pipelined-iters", type=int, default=50)
    p.add_argument("--trace-calls", type=int, default=7)
    p.add_argument(
        "--trace-pipelined",
        type=int,
        default=10,
        help="round 0: back-to-back calls per program in their own session",
    )
    p.add_argument(
        "--pipe-trials",
        type=int,
        default=1,
        help="the first n rounds each take the back-to-back session (round 0's "
        "stays pipelined_trace; every round's goes to pipelined_trace_trials)",
    )
    p.add_argument("--trace-dir")
    p.add_argument(
        "--keep-traces", default="", help="buckets whose round-0 trace stays"
    )
    p.add_argument(
        "--min-free-gb", type=float, default=10.0, help="disk guard for the traces"
    )
    p.add_argument("--gate-up-bits", default="auto")
    p.add_argument(
        "--k32z-rungs",
        default="",
        help="benchmark only: the K32Z tile heights (as dsv41_k32z_l1.py)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--out")
    p.add_argument("--hlo-dir", default=None, help="write each compiled HLO here")
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="compile every program and record its sizes; run nothing",
    )
    p.add_argument(
        "--ici-counters",
        default=None,
        metavar="ICI_K0_PY",
        help="path to Step 7 K0's ici_k0.py: per point, one profiler session "
        "per program holding one call, digested by its link_counters into the "
        "program's 'ici' record (Step 7 K2)",
    )
    p.set_defaults(no_reference=True)  # L1.load_weights: no exact-K32 reference
    return p.parse_args(argv)


def memory_info(compiled):
    """The executable's generated code and buffer sizes (bytes), if reported."""
    try:
        m = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    keys = (
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {k: getattr(m, k, None) for k in keys}


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def main(argv=None):
    args = parse_args(argv)
    if args.table:
        return print_tables(args.table)
    if args.split:
        return print_split(args.split[0], args.split[1:])
    if args.ops:
        return print_ops(args.ops, args.ops_variant)
    if args.drift:
        return print_drift(*args.drift)
    if not (args.weights and args.out and args.trace_dir):
        raise SystemExit("--weights, --out and --trace-dir are required")
    import dsv41_k32z_l1 as L1  # sets sys.path for the tree

    D = L1.D
    log = D.log
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    fe = D._fe()
    fe.import_fused_moe_v2(isolated=True)
    import torch  # noqa: F401 - after JAX; must not bring torch_tpu along

    from sglang_tpu import dsv4_mxfp4_k32z as k32z
    from sglang_tpu.moe.pallas import fused_moe_v2 as v2
    from sglang_tpu.moe.pallas.fused_moe_v2 import host

    if args.k32z_rungs:
        rungs = tuple(int(r) for r in args.k32z_rungs.split(","))
        # A ladder without 128 streams up to its tallest rung below it
        # (host.k32z_streamed_rows, Step 7 R2).
        if rungs != tuple(sorted(set(rungs))):
            raise ValueError(f"K32Z rungs {rungs} must be ascending")
        host.k32z_tile_height_rungs = lambda tile_m, _rungs=None: tuple(
            r for r in rungs if r <= tile_m
        )
        log(f"[bd] benchmark-only K32Z rungs {rungs}")

    if "torch_tpu" in sys.modules:
        raise RuntimeError("torch_tpu got imported; it would take the TPU from JAX")
    devices = jax.devices()
    ep = len(devices)
    mesh = Mesh(np.asarray(devices), ("ep",))
    configs = [c for c in args.configs.split(",") if c]
    variants = [v for v in args.variants.split(",") if v]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known {list(VARIANTS)}")
    buckets = [int(b) for b in args.buckets.split(",") if b]
    routings = [r for r in args.routings.split(",") if r]
    keep = {int(b) for b in args.keep_traces.split(",") if b}
    ici = load_ici_counters(args.ici_counters) if args.ici_counters else None
    os.makedirs(args.trace_dir, exist_ok=True)
    keys = [(c, v) for c in configs for v in variants]
    result = {
        "started_utc": D.utc_now(),
        "argv": sys.argv,
        "config": {
            "layer": args.layer,
            "ep": ep,
            "configs": {c: L1.CONFIGS[base_config(c)] for c in configs},
            "push_knobs": {c: push_knobs_of(c) for c in configs if push_knobs_of(c)},
            "variants": {v: list(VARIANTS[v]) for v in variants},
            "buckets_rows_per_rank": buckets,
            "routings": routings,
            "pad_rows_per_rank": args.pad_rows,
            "trials": args.trials,
            "warmup": args.warmup,
            "iters_sync": args.iters,
            "pipelined_iters": args.pipelined_iters,
            "trace_calls_per_round": args.trace_calls,
            "k32z_rungs": args.k32z_rungs or "default",
        },
        "environment": D.environment(jax, mesh),
        "points": [],
    }
    tree_id = L1.TREE / "K32Z_TREE_ID"
    if tree_id.exists():
        result["tree_id"] = tree_id.read_text().strip()
    log(f"[bd] ep={ep} layer={args.layer} configs={configs} variants={variants}")
    wargs = argparse.Namespace(**vars(args))
    wargs.configs = ",".join(sorted({base_config(c) for c in configs}))
    ops, ref_exps, gate_w, bias, norm_w, bits = L1.load_weights(
        wargs, fe, k32z, jax, mesh, result
    )
    del ref_exps
    result["config"]["gate_up_bits"] = bits
    D.write_json(args.out, result)
    repl = NamedSharding(mesh, P())
    rows_sh = NamedSharding(mesh, P("ep"))
    bias_d = jax.device_put(bias, repl)
    zero_bias_d = jax.device_put(np.zeros(L1.NUM_EXPERTS, np.float32), repl)

    def build(c, v):
        """The whole layer as one jitted program named moe_<config>_<variant>."""
        switches = frozenset(VARIANTS[v])
        compute = uses_compute_stage(c, v)

        def prog(x, gating, w1, w2, s1, s2, sbias, *routed):
            kfn = v2.fused_ep_moe_v2
            if base_config(c) in L1.PUSH_CONFIGS:
                options = D.push_combined_options(L1.NUM_EXPERTS // ep)
                options.update(push_knobs_of(c))
                kfn = D.make_kernel_fn(kfn, "push", options)
            if compute:
                kfn = functools.partial(kfn, stage="compute", routed=tuple(routed))
            moe = fe.make_fused_ep_moe(
                mesh,
                logical_intermediate_size=L1.INTER,
                expert_format=L1.CONFIGS[base_config(c)],
                ablation=switches,
                kernel_fn=kfn,
                jit=False,
            )
            return moe(x, gating, w1, w2, s1, s2, sbias)

        prog.__name__ = prog.__qualname__ = f"moe_{c}_{v}"
        return jax.jit(prog)

    def build_route(c):
        def prog(x, gating, w1, w2, s1, s2, sbias):
            moe = fe.make_fused_ep_moe(
                mesh,
                logical_intermediate_size=L1.INTER,
                expert_format=L1.CONFIGS[base_config(c)],
                kernel_fn=functools.partial(v2.fused_ep_moe_v2, stage="route"),
                jit=False,
            )
            return moe(x, gating, w1, w2, s1, s2, sbias)

        prog.__name__ = prog.__qualname__ = f"route_{c}"
        return jax.jit(prog)

    progs = {k: build(*k) for k in keys}
    needs_route = {c for c, v in keys if uses_compute_stage(c, v)}
    routes = {c: build_route(c) for c in needs_route}
    tile = jax.jit(
        lambda a: jnp.tile(a, (ep,) + (1,) * (a.ndim - 1)), out_shardings=rows_sh
    )

    def name(k):
        return f"{k[0]}/{k[1]}"

    for bucket in buckets:
        tokens = ep * bucket
        x = D.make_hidden(
            np.random.default_rng([args.seed, 1, bucket]),
            ep=ep,
            bucket=bucket,
            hidden=L1.HIDDEN,
            norm_w=norm_w,
        )
        x_d = jax.device_put(x, rows_sh)
        abstract = {
            c: (
                jax.ShapeDtypeStruct((tokens, L1.HIDDEN), x_d.dtype, sharding=rows_sh),
                jax.ShapeDtypeStruct(
                    (tokens, L1.NUM_EXPERTS), jnp.float32, sharding=rows_sh
                ),
                *ops[L1.weight_key(base_config(c))],
                jax.ShapeDtypeStruct((L1.NUM_EXPERTS,), jnp.float32, sharding=repl),
            )
            for c in configs
        }
        routed_abs = {}
        for c in needs_route:
            qs, ws, ps = jax.eval_shape(routes[c], *abstract[c])
            routed_abs[c] = (
                jax.ShapeDtypeStruct(
                    (ep * qs.shape[0],) + qs.shape[1:], qs.dtype, sharding=rows_sh
                ),
                jax.ShapeDtypeStruct(ws.shape, ws.dtype, sharding=rows_sh),
                jax.ShapeDtypeStruct(
                    (ep * ps.shape[0],) + ps.shape[1:], ps.dtype, sharding=rows_sh
                ),
            )
        compiled, cinfo = {}, {}
        for k in keys:
            c, v = k
            extra = routed_abs[c] if uses_compute_stage(c, v) else ()
            try:
                t0 = time.perf_counter()
                forced = knobs_of(c)
                saved = {a: getattr(host, a) for a in forced}
                for a, val in forced.items():
                    setattr(host, a, val)
                try:
                    lowered = progs[k].lower(*abstract[c], *extra)
                finally:
                    for a, val in saved.items():
                        setattr(host, a, val)
                t1 = time.perf_counter()
                compiled[k] = lowered.compile()
                t2 = time.perf_counter()
                cinfo[name(k)] = {
                    "lower_s": round(t1 - t0, 2),
                    "compile_s": round(t2 - t1, 2),
                    "kernels": D.kernel_names(lowered, compiled[k]),
                    "memory": memory_info(compiled[k]),
                }
                del lowered
                if args.hlo_dir:
                    os.makedirs(args.hlo_dir, exist_ok=True)
                    Path(args.hlo_dir, f"{c}_{v}_{bucket}.hlo.txt").write_text(
                        compiled[k].as_text()
                    )
                log(
                    f"[b={bucket}] compiled {name(k)}: {cinfo[name(k)]['lower_s']}s + "
                    f"{cinfo[name(k)]['compile_s']}s {cinfo[name(k)]['kernels']}"
                )
            except Exception as exc:  # noqa: BLE001
                cinfo[name(k)] = {
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "traceback": traceback.format_exc()[-4000:],
                }
                log(
                    f"[b={bucket}] compile {name(k)} FAILED: {cinfo[name(k)]['error'][:500]}"
                )
        if args.compile_only:
            result["points"].append(
                {"bucket": bucket, "compile_only": True, "compile": cinfo}
            )
            D.write_json(args.out, result)
            del compiled
            continue
        for routing in routings:
            point = {
                "bucket": bucket,
                "routing": routing,
                "compile": cinfo,
                "programs": {},
            }
            result["points"].append(point)
            try:
                gating, sbias, pad_index, designed, info = D.build_routing(
                    routing,
                    x=x,
                    gate_w=gate_w,
                    bias=bias,
                    ep=ep,
                    bucket=bucket,
                    num_experts=L1.NUM_EXPERTS,
                    topk=L1.TOPK,
                    pad_rows=args.pad_rows,
                    seed=args.seed,
                    jitter=args.jitter,
                    zipf_s=1.2,
                )
                real = np.ones(tokens, bool)
                real[pad_index] = False
                rows = np.nonzero(real)[0]
                sel, _gap = D.host_select(gating[rows], sbias, L1.TOPK)
                st = D.routing_summary(
                    sel, rows, ep=ep, bucket=bucket, num_experts=L1.NUM_EXPERTS
                )
                point["routing_stats"] = {
                    k: st[k]
                    for k in (
                        "recv_pairs_per_rank",
                        "max_pairs_per_expert",
                        "experts_hit_per_rank",
                    )
                }
                g_d = jax.device_put(gating, rows_sh)
                b_d = bias_d if routing == "real" else zero_bias_d
                routed = {}
                for c in needs_route:
                    q_l, tw, packed = routes[c](
                        x_d, g_d, *ops[L1.weight_key(base_config(c))], b_d
                    )
                    routed[c] = (tile(q_l), tw, tile(packed))
                    jax.block_until_ready(routed[c])
            except Exception as exc:  # noqa: BLE001
                point["error"] = f"{type(exc).__name__}: {exc}"[:4000]
                point["traceback"] = traceback.format_exc()[-4000:]
                log(f"[b={bucket} {routing}] setup FAILED: {point['error'][:500]}")
                D.write_json(args.out, result)
                continue

            def operands(k):
                c, v = k
                extra = routed[c] if uses_compute_stage(c, v) else ()
                return (x_d, g_d, *ops[L1.weight_key(base_config(c))], b_d, *extra)

            outs = {}
            ok = []
            for k in keys:
                if k not in compiled:
                    continue
                exact = not VARIANTS[k[1]] and config_exact(k[0])
                pc = point["programs"].setdefault(
                    name(k), {"wall": [], "trace": [], "exact": exact}
                )
                try:
                    out = compiled[k](*operands(k))
                    out.block_until_ready()
                    o = np.asarray(out)
                    of = o.astype(np.float32)
                    pc["nonfinite"] = int((~np.isfinite(of)).sum())
                    pc["pad_rows_exact_zero"] = bool(np.all(of[pad_index] == 0))
                    if k[1] in ("full", "disp"):
                        outs[k] = o
                    ok.append(k)
                except Exception as exc:  # noqa: BLE001
                    pc["error"] = f"{type(exc).__name__}: {exc}"[:4000]
                    pc["traceback"] = traceback.format_exc()[-4000:]
                    log(
                        f"[b={bucket} {routing} {name(k)}] run FAILED: {pc['error'][:500]}"
                    )
            checks = {}
            for c in configs:
                if not config_exact(c):
                    # ret_last_off / commit_only: invalid output by design.
                    continue
                full, disp = outs.get((c, "full")), outs.get((c, "disp"))
                if full is not None and disp is not None:
                    rel, worst, _, mx = D.compare(
                        disp.astype(np.float32), full.astype(np.float32), rows
                    )
                    checks[f"{c}/disp_vs_full"] = {
                        "bitwise_equal": bool(
                            np.array_equal(full.view(np.uint16), disp.view(np.uint16))
                        ),
                        "rel_l2": rel,
                        "max_abs": mx,
                    }
            if ("A", "full") in outs and ("AZ", "full") in outs:
                rel, worst, _, mx = D.compare(
                    outs[("AZ", "full")].astype(np.float32),
                    outs[("A", "full")].astype(np.float32),
                    rows,
                )
                checks["AZ_vs_A_full"] = {"rel_l2": rel, "worst_row_rel": worst}
            for c in configs:
                base = base_config(c)
                if c == base or (c, "full") not in outs or (base, "full") not in outs:
                    continue
                got, ref = outs[(c, "full")], outs[(base, "full")]
                rel, worst, _, mx = D.compare(
                    got.astype(np.float32), ref.astype(np.float32), rows
                )
                if not config_exact(c):
                    # Invalid by design (ret_last_off, commit_only): only the
                    # share of real rows that still equal the base's, bitwise.
                    same = np.all(got.view(np.uint16) == ref.view(np.uint16), axis=1)
                    checks[f"{c}_rows_vs_{base}_full"] = {
                        "rows_equal_frac": float(same[rows].mean()),
                        "rel_l2": rel,
                        "max_abs": mx,
                    }
                    continue
                checks[f"{c}_vs_{base}_full"] = {
                    "bitwise_equal": bool(
                        np.array_equal(got.view(np.uint16), ref.view(np.uint16))
                    ),
                    "rel_l2": rel,
                    "max_abs": mx,
                }
            point["checks"] = checks
            del outs
            if ici is not None:
                ici_sessions(point, ici, ok, compiled, operands, args.trace_dir)
            for trial in range(args.trials):
                for k in ok:
                    _, lat = D.run_point(
                        compiled[k],
                        operands(k),
                        warmup=args.warmup if trial == 0 else 1,
                        iters=args.iters,
                        pipelined_iters=args.pipelined_iters,
                    )
                    point["programs"][name(k)]["wall"].append(lat)
                pdir = os.path.join(args.trace_dir, f"b{bucket}_{routing}_t{trial}")
                if free_gb(args.trace_dir) < args.min_free_gb:
                    log(
                        f"[b={bucket} {routing}] disk guard: < {args.min_free_gb} GiB free, no trace"
                    )
                    point.setdefault("trace_skipped", []).append(trial)
                    continue
                try:
                    with jax.profiler.trace(pdir):
                        for _ in range(args.trace_calls):
                            for k in ok:
                                compiled[k](*operands(k)).block_until_ready()
                    files = sorted(
                        glob.glob(
                            os.path.join(pdir, "**", "*.xplane.pb"), recursive=True
                        )
                    )
                    dev = TR.load(files[-1])
                    point.setdefault("modules_seen", []).append(TR.programs_in(dev))
                    for k in ok:
                        dg = TR.digest(dev, f"jit_moe_{k[0]}_{k[1]}")
                        point["programs"][name(k)]["trace"].append(dg)
                        ovl = TR.overlay(dev, f"jit_moe_{k[0]}_{k[1]}")
                        if ovl and ovl["events"]:
                            point["programs"][name(k)].setdefault("overlay", []).append(
                                ovl
                            )
                    del dev
                except Exception as exc:  # noqa: BLE001
                    point.setdefault("trace_errors", []).append(
                        f"{type(exc).__name__}: {exc}"[:2000]
                    )
                    log(f"[b={bucket} {routing}] trace FAILED: {exc}")
                if not (trial == 0 and bucket in keep):
                    shutil.rmtree(pdir, ignore_errors=True)
                if trial < max(1, args.pipe_trials) and args.trace_pipelined:
                    pipe_trace(
                        point,
                        pdir + "_pipe",
                        ok,
                        compiled,
                        operands,
                        args.trace_pipelined,
                        keep=trial == 0 and bucket in keep,
                        first=trial == 0,
                    )
            summarize_point(point)
            parts = []
            for k in ok:
                s = point["programs"][name(k)].get("summary", {})
                parts.append(
                    f"{name(k)} span {fmt(s.get('span_us'))} kern {fmt(s.get('kernel_us'))}"
                )
            log(f"[b={bucket} {routing}] " + " | ".join(parts) + f" | checks {checks}")
            D.write_json(args.out, result)
        del compiled
    result["finished_utc"] = D.utc_now()
    D.write_json(args.out, result)
    log(f"[bd] done -> {args.out}")
    return 0


def pipe_trace(point, pdir, ok, compiled, operands, calls, keep=False, first=True):
    """One session of back-to-back calls per program: the device period.
    Every session's digest goes to pipelined_trace_trials; the first round's
    is also pipelined_trace (and overlay_pipelined)."""
    import jax

    try:
        with jax.profiler.trace(pdir):
            for k in ok:
                out = None
                for _ in range(calls):
                    out = compiled[k](*operands(k))
                out.block_until_ready()
        files = sorted(
            glob.glob(os.path.join(pdir, "**", "*.xplane.pb"), recursive=True)
        )
        dev = TR.load(files[-1])
        for k in ok:
            pc = point["programs"][f"{k[0]}/{k[1]}"]
            rec = TR.pipelined(dev, f"jit_moe_{k[0]}_{k[1]}")
            pc.setdefault("pipelined_trace_trials", []).append(rec)
            if not first:
                continue
            pc["pipelined_trace"] = rec
            ovl = TR.overlay(dev, f"jit_moe_{k[0]}_{k[1]}")
            if ovl and ovl["events"]:
                pc["overlay_pipelined"] = ovl
    except Exception as exc:  # noqa: BLE001
        point.setdefault("trace_errors", []).append(
            f"pipelined: {type(exc).__name__}: {exc}"[:2000]
        )
    if not keep:
        shutil.rmtree(pdir, ignore_errors=True)


# K0's calibrated reading of the (undocumented) ICI packet counters: 1,024 B
# per egress count from this chip (LOCAL_EGRESS*), 512 B per count forwarded
# from another link (LINK_DRIVEN*) or received (ingress); Step 7 K0 checked
# it against known transfers.
ICI_EGRESS_B = 1024
ICI_HALF_B = 512


def load_ici_counters(path):
    """Step 7 K0's ici_k0.py, loaded from its file for its link_counters."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ici_k0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "link_counters", None)):
        raise SystemExit(f"{path} has no link_counters")
    return mod


def ici_link_bytes(planes):
    """{chip: {link: {"out_local", "out_fwd", "in"}}} bytes from
    link_counters' {chip: {"egress": {"L<k>:<src>": n}, "ingress": ...}}."""
    out = {}
    for chip, rec in planes.items():
        links = {}
        for key, n in rec.get("egress", {}).items():
            link, _, src = key.partition(":")
            d = links.setdefault(link, {"out_local": 0, "out_fwd": 0, "in": 0})
            if src.startswith("LOCAL_EGRESS"):
                d["out_local"] += n * ICI_EGRESS_B
            else:
                d["out_fwd"] += n * ICI_HALF_B
        for link, n in rec.get("ingress", {}).items():
            d = links.setdefault(link, {"out_local": 0, "out_fwd": 0, "in": 0})
            d["in"] += n * ICI_HALF_B
        out[str(chip)] = links
    return out


def ici_sessions(point, ici, ok, compiled, operands, trace_dir):
    """One profiler session per program holding exactly one call (after the
    check call has warmed it): the ICI link counters of that call (K0's
    method), raw and in bytes, into the program's 'ici' record."""
    import jax

    for k in ok:
        pc = point["programs"][f"{k[0]}/{k[1]}"]
        pdir = os.path.join(
            trace_dir, f"ici_b{point['bucket']}_{point['routing']}_{k[0]}_{k[1]}"
        )
        try:
            compiled[k](*operands(k)).block_until_ready()
            with jax.profiler.trace(pdir):
                compiled[k](*operands(k)).block_until_ready()
            planes = ici.link_counters(pdir)
            pc["ici"] = {"planes": planes, "bytes": ici_link_bytes(planes)}
            peak = max(
                (
                    v["out_local"] + v["out_fwd"]
                    for links in pc["ici"]["bytes"].values()
                    for v in links.values()
                ),
                default=0,
            )
            _log(
                f"[ici b={point['bucket']} {point['routing']}] {k[0]}/{k[1]}: "
                f"busiest directed link {peak / 1e6:.2f} MB"
            )
        except Exception as exc:  # noqa: BLE001
            pc["ici_error"] = f"{type(exc).__name__}: {exc}"[:2000]
            _log(f"[ici] {k[0]}/{k[1]} FAILED: {exc}")
        shutil.rmtree(pdir, ignore_errors=True)


def _log(msg):
    print(time.strftime("[%Y-%m-%dT%H:%M:%SZ] ", time.gmtime()) + msg, flush=True)


def fmt(qv):
    return "nan" if not qv else f"{qv['median']:.1f}"


def pooled(traces, key):
    vals = []
    for t in traces:
        if t:
            vals += [v for v in t.get(key, []) if v is not None]
    return TR.q(vals)


def summarize_point(point):
    """Pool the rounds' traced calls per program: median [p10, p90]."""
    for pc in point["programs"].values():
        tr = pc.get("trace", [])
        s = {
            "span_us": pooled(tr, "span_max_us"),
            "kernel_us": pooled(tr, "kernel_max_us"),
            "gap_crit_us": pooled(tr, "gap_crit_us"),
            "pre_kernel_crit_us": pooled(tr, "pre_kernel_crit_us"),
            "post_kernel_crit_us": pooled(tr, "post_kernel_crit_us"),
            "module_us": pooled(tr, "module_max_us"),
            "barrier_crit_us": pooled(tr, "barrier_crit_us"),
            "trial_span_medians": [
                statistics.median([v for v in t["span_max_us"] if v is not None])
                for t in tr
                if t and t.get("span_max_us")
            ],
            "trial_kernel_medians": [
                statistics.median([v for v in t["kernel_max_us"] if v is not None])
                for t in tr
                if t and any(v is not None for v in t.get("kernel_max_us", []))
            ],
        }
        cats = sorted({c for t in tr if t for c in t.get("cat_max_us", {})})
        s["cat_us"] = {}
        for c in cats:
            vals = [
                v for t in tr if t for v in t["cat_max_us"].get(c, []) if v is not None
            ]
            s["cat_us"][c] = TR.q(vals)
        wall = pc.get("wall", [])
        if wall:
            s["wall_pipelined_ms"] = [w["pipelined_mean"] for w in wall]
            s["wall_sync_median_ms"] = [w["median"] for w in wall]
        pc["summary"] = s


# --------------------------------------------------------------------------- #
# Tables (pod CPU): breakdown per config and point, both boundaries.
# --------------------------------------------------------------------------- #
def derive(T):
    """Step 1's derived quantities from medians {variant: T}."""
    F, Cm, Cp, B = (T.get(v) for v in ("full", "comm", "comp", "base"))
    if None in (F, Cm, Cp, B):
        return None
    comm, comp = Cm - B, Cp - B
    ov = Cm + Cp - B - F
    d = {
        "T_full": F,
        "T_comm": Cm,
        "T_comp": Cp,
        "T_base": B,
        "comm_ovh": comm,
        "comp_ovh": comp,
        "overlap": ov,
        "completeness": ov / min(comm, comp) if min(comm, comp) > 0 else None,
        "exp_comm": F - Cp,
        "exp_comp": F - Cm,
    }
    for v, k, ref in (
        ("a2w", "weights_in", "comm"),
        ("nowt", "ffn_in", "comm"),
        ("ret", "ret_exposed", None),
        ("disp", "disp_exposed", None),
    ):
        if T.get(v) is None:
            continue
        d[k] = (T[v] - T[ref]) if ref else (F - T[v])
    return d


COLUMNS = (
    "T_full",
    "T_comm",
    "T_comp",
    "T_base",
    "comm_ovh",
    "comp_ovh",
    "overlap",
    "completeness",
    "exp_comm",
    "exp_comp",
    "weights_in",
    "ffn_in",
    "ret_exposed",
    "disp_exposed",
)


def print_tables(paths):
    pts = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        for pt in r["points"]:
            pt["_layer"] = r["config"]["layer"]
            pts.append(pt)
    pts.sort(key=lambda p: (p["_layer"], p["bucket"], p["routing"] != "uniform"))
    for bnd, key in (("span (whole layer call)", "span_us"), ("kernel", "kernel_us")):
        print(f"\n### Boundary: {bnd}, µs, median of pooled calls\n")
        if key == "kernel_us":
            print("T_comp at the kernel boundary is the {ret_off} kernel.\n")
        print(
            "| layer | rows/rank | routing | cfg | T_full | T_comm | T_comp | T_base "
            "| comm | comp | overlap | compl. | exp comm | exp comp | w_in | ffn_in "
            "| ret exp | disp exp |"
        )
        print("|" + "---|" * 18)
        for pt in pts:
            for cfg in ("A", "AZ"):
                T = {}
                for v in VARIANTS:
                    s = pt["programs"].get(f"{cfg}/{v}", {}).get("summary", {})
                    if s.get(key):
                        T[v] = s[key]["median"]
                if key == "kernel_us" and "ret" in T:
                    # A's dispatch is all before the kernel, so the comp-only
                    # KERNEL is {ret_off} (Step 1's A definition); {disp_off}
                    # leaves the kernel code as is but not its time (N3a).
                    T["comp"] = T["ret"]
                d = derive(T)
                if not d:
                    continue

                cells = [str(pt["_layer"]), str(pt["bucket"]), pt["routing"], cfg]
                for k in COLUMNS:
                    x = d.get(k)
                    prec = 2 if k == "completeness" else 1
                    cells.append("–" if x is None else f"{x:.{prec}f}")
                print("| " + " | ".join(cells) + " |")
    print_attribution(pts)
    print("\n### Checks\n")
    for pt in pts:
        print(
            f"- L{pt['_layer']} {pt['bucket']} {pt['routing']}: {pt.get('checks')} "
            f"errors {pt.get('error')} {pt.get('trace_errors')}"
        )
    return 0


def print_attribution(pts):
    """ΔT_full (AZ - A): span phases on the critical core, then the kernel's
    T_full = T_base + comm + comp - overlap terms and the sub-splits."""

    def med(pt, cfg, v, key):
        s = pt["programs"].get(f"{cfg}/{v}", {}).get("summary", {})
        return (s.get(key) or {}).get("median")

    print("\n### ΔT_full, AZ − A, µs (medians)\n")
    print(
        "| layer | rows/rank | routing | Δspan | Δpre-kernel | Δkernel | Δpost-kernel "
        "| Δgap | kernel: ΔT_base | Δcomm | Δcomp | −Δoverlap | Δw_in | Δffn_in "
        "| Δret exp |"
    )
    print("|" + "---|" * 15)
    for pt in pts:
        cells = [str(pt["_layer"]), str(pt["bucket"]), pt["routing"]]
        for key in PHASES + ("gap_crit_us",):
            a, z = med(pt, "A", "full", key), med(pt, "AZ", "full", key)
            cells.append("–" if None in (a, z) else f"{z - a:+.1f}")
        dk = {}
        for cfg in ("A", "AZ"):
            T = {v: med(pt, cfg, v, "kernel_us") for v in VARIANTS}
            if T.get("ret") is not None:
                T["comp"] = T["ret"]
            dk[cfg] = derive({v: t for v, t in T.items() if t is not None})
        for k, sign in (
            ("T_base", 1),
            ("comm_ovh", 1),
            ("comp_ovh", 1),
            ("overlap", -1),
            ("weights_in", 1),
            ("ffn_in", 1),
            ("ret_exposed", 1),
        ):
            a = (dk["A"] or {}).get(k)
            z = (dk["AZ"] or {}).get(k)
            cells.append("–" if None in (a, z) else f"{sign * (z - a):+.1f}")
        print("| " + " | ".join(cells) + " |")


PHASES = ("span_us", "pre_kernel_crit_us", "kernel_us", "post_kernel_crit_us")


def print_split(pairs, paths):
    """NEW − OLD per point, µs: span phases on the critical core (full),
    the kernel's ablation sub-splits (weights a2w − comm, FFN nowt − comm,
    T_base, the return T_comm, exposed return full − ret), instruction
    overlay waits in the kernel (sync / back to back; overlay flags only)
    and the back-to-back period, idle gap and wall-clock pipelined call."""
    pairs = [tuple(p.split(":")) for p in pairs.split(",")]

    def med(pt, cfg, v, key, fallback=None):
        s = pt["programs"].get(f"{cfg}/{v}", {}).get("summary", {})
        got = (s.get(key) or {}).get("median")
        if got is None and fallback and "ffn_off" in VARIANTS[v]:
            # Without an FFN body a knob that only changes the body leaves
            # the program the same but for its name, and the persistent
            # compilation cache serves it under the first one's module name:
            # the OLD program's numbers are this one's.
            return med(pt, fallback, v, key)
        return got

    def ovl(pt, cfg, key):
        pc = pt["programs"].get(f"{cfg}/full", {})
        o = pc.get(key)
        if isinstance(o, list):
            vals = [x["in_kernel_us"] for x in o]
            return statistics.median(vals) if vals else None
        if o is None:
            # The digest is kept only where events were seen: none is 0.
            traced = pc.get("trace") if key == "overlay" else pc.get("pipelined_trace")
            return 0.0 if traced else None
        return o["in_kernel_us"]

    def pipe(pt, cfg, key):
        """Back-to-back: the median of two-call means where the digest has it
        (alternating programs; dsv41_k32z_trace.pipelined), else the median."""
        t = pt["programs"].get(f"{cfg}/full", {}).get("pipelined_trace") or {}
        two = key.replace("_us_max_core", "2_us_max_core")
        return t.get(two, t.get(key))

    def wall(pt, cfg):
        s = pt["programs"].get(f"{cfg}/full", {}).get("summary", {})
        w = s.get("wall_pipelined_ms")
        return statistics.median(w) * 1e3 if w else None

    cols = (
        "Δspan | Δpre | Δkernel | Δpost | Δgap | Δweights_in | Δffn_in | ΔT_base "
        "| Δreturn | Δret_exp | Δovl sync | Δovl b2b | Δb2b period | Δb2b idle "
        "| Δb2b span | Δwall pipe"
    )
    print(f"\n### Source split, NEW − OLD, µs (medians)\n")
    print(f"| rows/rank | routing | NEW vs OLD | {cols} |")
    print("|" + "---|" * (3 + cols.count("|") + 1))
    for path in paths:
        r = json.loads(Path(path).read_text())
        for pt in r["points"]:
            if pt.get("compile_only") or "programs" not in pt:
                continue
            for old, new in pairs:
                if (
                    f"{old}/full" not in pt["programs"]
                    or f"{new}/full" not in pt["programs"]
                ):
                    continue
                cells = [str(pt["bucket"]), pt["routing"], f"{new} vs {old}"]

                def d(a, b, sign=1):
                    return "–" if None in (a, b) else f"{sign * (b - a):+.1f}"

                for key in PHASES + ("gap_crit_us",):
                    cells.append(
                        d(med(pt, old, "full", key), med(pt, new, "full", key))
                    )
                T = {}
                for cfg, fb in ((old, None), (new, old)):
                    T[cfg] = {v: med(pt, cfg, v, "kernel_us", fb) for v in VARIANTS}

                def sub(cfg, a, b=None):
                    x = T[cfg].get(a)
                    y = T[cfg].get(b) if b else 0.0
                    return None if None in (x, y) else x - y

                for a, b in (
                    ("a2w", "comm"),
                    ("nowt", "comm"),
                    ("base", None),
                    ("comm", None),
                ):
                    cells.append(d(sub(old, a, b), sub(new, a, b)))
                re_old = (
                    None
                    if None in (T[old].get("full"), T[old].get("ret"))
                    else T[old]["full"] - T[old]["ret"]
                )
                re_new = (
                    None
                    if None in (T[new].get("full"), T[new].get("ret"))
                    else T[new]["full"] - T[new]["ret"]
                )
                cells.append(d(re_old, re_new))
                cells.append(d(ovl(pt, old, "overlay"), ovl(pt, new, "overlay")))
                cells.append(
                    d(
                        ovl(pt, old, "overlay_pipelined"),
                        ovl(pt, new, "overlay_pipelined"),
                    )
                )
                cells.append(
                    d(
                        pipe(pt, old, "period_us_max_core"),
                        pipe(pt, new, "period_us_max_core"),
                    )
                )
                for key in ("idle_us_max_core", "span_us_max_core"):
                    cells.append(d(pipe(pt, old, key), pipe(pt, new, key)))
                cells.append(d(wall(pt, old), wall(pt, new)))
                print("| " + " | ".join(cells) + " |")
    return 0


CATS = (
    "combine",
    "select",
    "copy",
    "wait_ag_tok",
    "wait_ag_meta",
    "wait_other",
    "other",
)


def print_ops(paths, variant, min_us=0.5):
    """A vs AZ per point: span phases on the critical core, op categories
    (max over cores), then every op >= min_us in device order (round 0)."""
    for p in paths:
        r = json.loads(Path(p).read_text())
        for pt in r["points"]:
            head = f"L{r['config']['layer']} {pt['bucket']} {pt['routing']} {variant}"
            print(f"\n### {head}\n")
            print(
                "| cfg | span | pre-kernel | kernel | post-kernel | gap | module "
                "| barrier | pipe period | pipe idle | pipe barrier | pipe span | wall pipe ms | "
                + " | ".join(CATS)
                + " |"
            )
            print("|" + "---|" * (13 + len(CATS)))
            timelines = {}
            for cfg in ("A", "AZ"):
                pc = pt["programs"].get(f"{cfg}/{variant}")
                if not pc or "summary" not in pc:
                    continue
                s = pc["summary"]
                cells = [cfg]
                for k in PHASES + ("gap_crit_us", "module_us", "barrier_crit_us"):
                    cells.append(fmt(s.get(k)))
                pt_ = pc.get("pipelined_trace") or {}
                for k in (
                    "period_us_max_core",
                    "idle_us_max_core",
                    "barrier_us_max_core",
                    "span_us_max_core",
                ):
                    cells.append("–" if k not in pt_ else f"{pt_[k]:.1f}")
                wp = s.get("wall_pipelined_ms")
                cells.append("–" if not wp else f"{statistics.median(wp):.4f}")
                for c in CATS:
                    cells.append(fmt(s.get("cat_us", {}).get(c)))
                print("| " + " | ".join(cells) + " |")
                tr = [t for t in pc.get("trace", []) if t]
                if tr:
                    timelines[cfg] = tr[0]["ops"]
            for cfg, ops in timelines.items():
                big = [o for o in ops if o[3] is not None and o[3] >= min_us]
                rest = sum(o[3] for o in ops if o[3] is not None and o[3] < min_us)
                print(
                    f"\n{cfg}: {len(ops)} ops, {len(big)} >= {min_us} µs "
                    f"(the rest sum to {rest:.1f} µs); [start µs] name (category) dur µs"
                )
                for name, cat, n, dur, start, _ in big:
                    st = "–" if start is None else f"{start:.1f}"
                    rep = f", x{n}" if n > 1 else ""
                    print(f"  [{st}] {name} ({cat}{rep}) {dur:.2f}")
    return 0


def print_drift(step1_csv, *paths):
    """Config A here vs Step 1 (serving mesh, T_actual = bucket): T_full,
    T_comm, T_comp, T_base at the kernel boundary and the span ("wrapper"
    in Step 1, whose comp/base spans subtract the pre-kernel waits); the
    kernel-boundary T_comp is the {ret_off} kernel, as in Step 1."""
    ref = {}
    with open(step1_csv, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["mesh"] == "serving"
                and row["config"] == "A"
                and row["bucket"] == row["t_actual"]
            ):
                ref[(int(row["bucket"]), row["routing"], row["boundary"])] = row
    print("\n### Config A drift vs Step 1 (11-19), µs: Step 1 → now (Δ%)\n")
    print(
        "| layer | rows/rank | routing | boundary | T_full | T_comm | T_comp | T_base |"
    )
    print("|" + "---|" * 8)
    rows_ = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        rows_ += [(r, pt) for pt in r["points"]]
    rows_.sort(key=lambda t: (t[1]["bucket"], t[1]["routing"] != "uniform"))
    for r, pt in rows_:
        for bnd, key, s1 in (
            ("kernel", "kernel_us", "kernel"),
            ("span", "span_us", "wrapper"),
        ):
            row = ref.get((pt["bucket"], pt["routing"], s1))
            if row is None:
                continue
            cells = [
                str(r["config"]["layer"]),
                str(pt["bucket"]),
                pt["routing"],
                bnd,
            ]
            comp = "ret" if key == "kernel_us" else "comp"
            for v, col in (
                ("full", "T_full"),
                ("comm", "T_comm"),
                (comp, "T_comp"),
                ("base", "T_base"),
            ):
                s = pt["programs"].get(f"A/{v}", {}).get("summary", {})
                now = (s.get(key) or {}).get("median")
                old = float(row[col]) if row.get(col) else None
                if now is None or old is None:
                    cells.append("–")
                else:
                    cells.append(
                        f"{old:.1f} → {now:.1f} ({100 * (now / old - 1):+.1f}%)"
                    )
            print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
