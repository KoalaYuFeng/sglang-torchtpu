#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 N2b: lower the DeepSeek-V4.1 Flash MoE layer (served shapes) for TPU
and digest each program: the fused EP kernel names in it, the text's sha256
and length. One record per (dispatch, expert format, rows per rank, ablation).

Runs on CPU (a TPU7x target, as the unit tests lower) or on a TPU host. Run it
against two trees (PYTHONPATH) and compare the digests to show a program is
byte-identical: the default K512 push program must be the base tree's.

    JAX_TRACEBACK_IN_LOCATIONS_LIMIT=0 JAX_PLATFORMS=cpu \\
    PYTHONPATH=$T/python:$T python3 dsv41_k32z_push_lowering.py --out lower.json

Dispatches: "all_gather" (A), "push" with #2102's tuned bundle (D, as the
dispatch x routing harness runs it), "push_defaults" (push, kernel defaults),
and the layer's own switch: "serving_push" / "serving_auto"
(``make_fused_ep_moe(dispatch="push" | "auto")``).

Step 7 K2: ``--push-variants o1,h0t1,...`` also lowers the tuned bundle with
each set of push-option tokens (dsv41_k32z_breakdown's D/DZ grammar), as
dispatch "push+<tokens>". Kernel names now include the table, combine and
SparseCore kernels, where the K2 visit order and select fold put their tokens.
Step 7 K2f: ``--ag-variants y1`` lowers the all-gather build with those
host-knob tokens set (dsv41_k32z_breakdown's A/AZ grammar), as dispatch
"all_gather+<tokens>".

``--op`` lowers the serving op instead: the body ``fused_ep_op._make_jax_fn``
builds, jitted as ``cpu_export_jax_op`` jits it (rank operand included), with
dispatches "all_gather" (the base tree's signature), "push" and "auto"; each
record carries the op's name. "bench_all_gather" / "bench_push" lower the same
body (the base tree's signature) with the layer benchmarks' kernel callable
for "all_gather" / "push" (A/AZ, D/DZ) put in through
``fused_ep_op.kernel_override``: the benchmark configs in the op's shape, to
compare a serving op's whole text with. After the loop, every serving_push
record is checked against push and every auto / serving_auto record against
push up to ``--push-max-rows`` rows per rank and all_gather above (same text).
Every record also lists its custom calls (``custom_calls``: call target,
kernel name and a digest of the call's attributes and types), which compare
programs whose glue differs (the serving op takes the rank as an operand, the
benchmark builds it). Compare two result files (e.g. this tree's and
K32Z_LOWER_TREE's) with

    python3 dsv41_k32z_push_lowering.py --compare new.json base.json
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
if os.environ.get(
    "JAX_PLATFORMS", ""
) == "cpu" and "device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    ).strip()

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The tree whose layer is lowered: this one, or K32Z_LOWER_TREE (a base tree
# to compare against; this file's helpers are imported from beside it).
TREE = Path(os.environ.get("K32Z_LOWER_TREE") or HERE.parents[1]).resolve()
for _p in (str(HERE), str(TREE), str(TREE / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dsv41_fused_ep_dispatch_dist as D  # noqa: E402

NUM_EXPERTS, HIDDEN, INTER, TOPK = 384, 5120, 2304, 6
# The layer's own kernels by name: the fused EP kernels, and (Step 7 K2) the
# table, combine and SparseCore pre-push kernels.
KERNEL_NAME_RE = re.compile(
    r"((?:push_fused_moe|fused_ep_moe_v2|moe_tc_routing_exchange|moe_push_tables"
    r"|moe_v2_combine|moe_sc_prepush)\w*)"
)
# One custom call per line of the printed module: its target, then its
# operands, attributes (a Pallas kernel's serialized body and its name) and
# types. SSA names and a location are not part of the program.
CUSTOM_CALL_RE = re.compile(r"stablehlo\.custom_call @([\w.$\-]+)(.*)$")
CALL_KERNEL_RE = re.compile(r'kernel_name = "([^"]*)"')
SSA_RE = re.compile(r"%[\w#.$\-]+")
LOC_RE = re.compile(r"\s+loc\(.*\)$")


def custom_calls(text):
    """Sorted [target, kernel name, sha256] of every custom call in a lowered
    text, the digest over the call with SSA names and location dropped."""
    out = []
    for line in text.splitlines():
        m = CUSTOM_CALL_RE.search(line)
        if not m:
            continue
        body = SSA_RE.sub("%", LOC_RE.sub("", m.group(2)))
        name = CALL_KERNEL_RE.search(body)
        out.append(
            [
                m.group(1),
                name.group(1) if name else "",
                hashlib.sha256(body.encode()).hexdigest(),
            ]
        )
    return sorted(out)


ABLATION_SETS = {
    "full": (),
    "comm": ("w_off", "ffn_off"),
    "comp": ("disp_off", "ret_off"),
    "base": ("w_off", "ffn_off", "ret_off", "disp_off"),
    "a2w": ("ffn_off",),
    "nowt": ("w_off",),
    "ret": ("ret_off",),
    "disp": ("disp_off",),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dispatches", default="push")
    p.add_argument("--formats", default="k512,k32z")
    p.add_argument("--buckets", default="8,2048", help="rows per rank")
    p.add_argument(
        "--ablations",
        default="none",
        help="'none' (no argument), or names of ABLATION_SETS ('full' passes "
        "an empty set)",
    )
    p.add_argument("--gate-up-bits", type=int, default=4)
    p.add_argument(
        "--push-variants",
        default="",
        help="push-option token sets (e.g. o1,h0t1d1,k2) lowered on top of "
        "the tuned bundle as dispatch push+<tokens> (Step 7 K2)",
    )
    p.add_argument(
        "--ag-variants",
        default="",
        help="host-knob token sets of the all-gather build (e.g. y1) lowered "
        "as dispatch all_gather+<tokens> (Step 7 K2f)",
    )
    p.add_argument(
        "--op", action="store_true", help="lower the serving op's body (fused_ep_op)"
    )
    p.add_argument("--push-max-rows", type=int, default=1024, help="auto's bound")
    p.add_argument("--text-dir", default=None, help="write each lowered text here")
    p.add_argument(
        "--compare", nargs=2, metavar="JSON", help="compare two result files, exit"
    )
    p.add_argument("--out")
    args = p.parse_args(argv)
    if not args.compare and not args.out:
        p.error("--out is required")
    return args


def _twin(rec, push_max_rows):
    """The dispatch whose program this record's must be, or None."""
    if rec["dispatch"] == "serving_push":
        return "push"
    if rec["dispatch"] in ("auto", "serving_auto"):
        return "push" if rec["bucket"] <= push_max_rows else "all_gather"
    return None


def same_program_checks(records, push_max_rows):
    """serving_push must lower to the benchmarks' D program; auto (op) and
    serving_auto (layer) to push up to the bound and to all_gather above."""
    by_key = {
        (r["path"], r["dispatch"], r["format"], r["bucket"], r["ablation"]): r
        for r in records
    }
    for rec in records:
        twin = _twin(rec, push_max_rows)
        other = by_key.get(
            (rec["path"], twin, rec["format"], rec["bucket"], rec["ablation"])
        )
        if other is None or "sha256" not in rec:
            continue
        equal = rec["sha256"] == other.get("sha256")
        rec["same_program_as"] = {"dispatch": twin, "equal": equal}
        D.log(
            f"[same] {rec['path']} {rec['dispatch']} {rec['format']} "
            f"b={rec['bucket']} {rec['ablation']}: {'==' if equal else '!='} {twin}"
        )


def compare_files(path_a, path_b):
    """Records present in both files: same program text (sha256) and op name."""

    def index(path):
        records = json.loads(Path(path).read_text())["records"]
        return {
            (
                r.get("path", "layer"),
                r["dispatch"],
                r["format"],
                r["bucket"],
                r["ablation"],
            ): r
            for r in records
        }

    a, b = index(path_a), index(path_b)
    common = sorted(set(a) & set(b), key=str)
    diffs = 0
    for key in common:
        ra, rb = a[key], b[key]
        same = (
            "sha256" in ra
            and ra.get("sha256") == rb.get("sha256")
            and ra.get("op_name") == rb.get("op_name")
        )
        diffs += not same
        print(
            f"{'same' if same else 'DIFF'} {' '.join(map(str, key))} "
            f"{ra.get('sha256', 'error')[:16]} {rb.get('sha256', 'error')[:16]} "
            f"{ra.get('op_name', '')} {rb.get('op_name', '')}"
        )
    for key in sorted(set(a) ^ set(b), key=str):
        print(f"only in {path_a if key in a else path_b}: {' '.join(map(str, key))}")
    print(f"{len(common) - diffs} same, {diffs} different")
    return 1 if diffs or not common else 0


@contextlib.contextmanager
def tpu7x_target():
    """Lower Pallas TPU kernels for TPU7x on a CPU host (the unit tests' way)."""
    import jax
    from jax._src import tpu_info as core_tpu_info
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    if jax.devices()[0].platform == "tpu":
        yield None
        return
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


@contextlib.contextmanager
def host_knobs(host, forced):
    """Set fused_moe_v2 host attributes for one lowering (Step 7 K2f)."""
    forced = dict(forced or {})
    saved = {a: getattr(host, a) for a in forced}
    for a, v in forced.items():
        setattr(host, a, v)
    try:
        yield
    finally:
        for a, v in saved.items():
            setattr(host, a, v)


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        return compare_files(*args.compare)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    fe = D._fe()
    package = fe.import_fused_moe_v2(isolated=True)
    from sglang_tpu.moe.pallas.fused_moe_v2 import host

    loaded = Path(package.__file__).resolve()
    if TREE not in loaded.parents:
        raise SystemExit(f"fused_moe_v2 came from {loaded}, not from {TREE}")

    devices = jax.devices()
    ep = 8
    if len(devices) < ep:
        raise SystemExit(f"needs {ep} devices, found {len(devices)}")
    mesh = Mesh(np.asarray(devices[:ep]), ("ep",))
    g_local = NUM_EXPERTS // ep
    on_tpu = devices[0].platform == "tpu"
    rows = NamedSharding(mesh, P("ep"))
    repl = NamedSharding(mesh, P())
    e = NUM_EXPERTS
    pad_i = -(-INTER // 512) * 512

    def operands(fmt):
        if fmt == "k512":
            return (
                jax.ShapeDtypeStruct(
                    (e, HIDDEN, 2 * pad_i), jnp.float4_e2m1fn, sharding=rows
                ),
                jax.ShapeDtypeStruct(
                    (e, pad_i, HIDDEN), jnp.float4_e2m1fn, sharding=rows
                ),
                jax.ShapeDtypeStruct(
                    (e * host.fp4_scale_rows(HIDDEN, 512, padded=True), 2 * pad_i),
                    jnp.float32,
                    sharding=rows,
                ),
                jax.ShapeDtypeStruct(
                    (e * host.fp4_scale_rows(pad_i, 512, padded=True), HIDDEN),
                    jnp.float32,
                    sharding=rows,
                ),
            )
        geo = host.k32z_table_geometry(HIDDEN, INTER, args.gate_up_bits)
        return (
            jax.ShapeDtypeStruct(
                (e, HIDDEN, 2 * INTER), jnp.float4_e2m1fn, sharding=rows
            ),
            jax.ShapeDtypeStruct((e, INTER, HIDDEN), jnp.float4_e2m1fn, sharding=rows),
            jax.ShapeDtypeStruct((e * geo.rows1, geo.cols1), jnp.uint8, sharding=rows),
            jax.ShapeDtypeStruct((e * geo.rows2, geo.cols2), jnp.uint8, sharding=rows),
        )

    variants = [t for t in args.push_variants.split(",") if t]
    knobs = {}
    push_host = {}
    if variants:
        import dsv41_k32z_breakdown as BD  # the D/DZ token grammar

        knobs = {f"push+{t}": BD.push_knobs_of("DZ" + t) for t in variants}
        # Step 11 E1: host-knob tokens of a push config (e.g. G1), set while
        # that variant lowers.
        push_host = {f"push+{t}": BD.knobs_of("DZ" + t) for t in variants}
        empty = [d for d, k in knobs.items() if not k and not push_host[d]]
        if empty:
            raise SystemExit(f"push variants without push-option tokens: {empty}")
    ag_variants = [t for t in args.ag_variants.split(",") if t]
    ag_knobs = {}
    if ag_variants:
        import dsv41_k32z_breakdown as BD  # the A/AZ token grammar

        ag_knobs = {f"all_gather+{t}": BD.knobs_of("AZ" + t) for t in ag_variants}
        empty = [d for d, k in ag_knobs.items() if not k]
        if empty:
            raise SystemExit(f"all-gather variants without host-knob tokens: {empty}")

    def kernel_fn(dispatch):
        if dispatch == "all_gather" or dispatch in ag_knobs:
            return None
        if dispatch in knobs:
            options = D.push_combined_options(g_local)
            options.update(knobs[dispatch])
        else:
            options = D.push_combined_options(g_local) if dispatch == "push" else {}
        return D.make_kernel_fn(package.fused_ep_moe_v2, "push", options)

    def layer_program(dispatch, fmt, extra):
        """make_fused_ep_moe, jitted: a benchmark config, or (serving_*) the
        layer's own dispatch switch."""
        if dispatch.startswith("serving_"):
            return fe.make_fused_ep_moe(
                mesh,
                logical_intermediate_size=INTER,
                expert_format=fmt,
                dispatch=dispatch.removeprefix("serving_"),
                push_max_rows=args.push_max_rows,
                **extra,
            )
        return fe.make_fused_ep_moe(
            mesh,
            logical_intermediate_size=INTER,
            expert_format=fmt,
            kernel_fn=kernel_fn(dispatch),
            **extra,
        )

    def op_program(dispatch, fmt):
        """The serving op's body, jitted as cpu_export_jax_op jits it, and the
        op's name (as V4.1 Flash's layers key it; None for a bench_* body)."""
        from sglang_tpu.models.deepseek_v41 import fused_ep_op as op_mod

        head = (ep, TOPK, True, 1.5, INTER, "ep", fmt, args.gate_up_bits)
        if dispatch.startswith("bench_"):
            # The op's body with the layer benchmarks' kernel callable (the
            # base tree's signature): A/AZ for bench_all_gather, D/DZ for
            # bench_push. _make_jax_fn binds the callable when it builds.
            bench = kernel_fn(dispatch.removeprefix("bench_"))
            with op_mod.kernel_override(bench):
                fn, _, (out_spec,) = op_mod._make_jax_fn(mesh, *head[:7])
            jitted = jax.jit(
                fn, keep_unused=True, out_shardings=NamedSharding(mesh, out_spec)
            )
            return jitted, None
        # all_gather passes no dispatch: the base tree's signatures.
        tail = () if dispatch == "all_gather" else (dispatch, args.push_max_rows)
        fn, _, (out_spec,) = op_mod._make_jax_fn(mesh, *head[:7], *tail)
        if tail:
            name = op_mod._op_name(*op_mod._op_key(*head, *tail))
        else:
            name = op_mod._op_name(*head)
        jitted = jax.jit(
            fn, keep_unused=True, out_shardings=NamedSharding(mesh, out_spec)
        )
        return jitted, name

    result = {
        "started_utc": D.utc_now(),
        "argv": sys.argv,
        "platform": devices[0].platform,
        "tree": str(TREE),
        "env_traceback_limit": os.environ.get("JAX_TRACEBACK_IN_LOCATIONS_LIMIT"),
        "jax": jax.__version__,
        "records": [],
    }
    path = "op" if args.op else "layer"
    ablations = [a for a in args.ablations.split(",") if a]
    if args.op and ablations != ["none"]:
        raise SystemExit("--op lowers the serving op, which takes no ablation")
    if args.op and (knobs or ag_knobs):
        raise SystemExit("--op takes no --push-variants / --ag-variants")
    dispatches = [d for d in args.dispatches.split(",") if d] + list(knobs)
    dispatches += list(ag_knobs)
    result["push_variants"] = knobs
    result["push_host_variants"] = push_host
    result["ag_variants"] = ag_knobs
    for dispatch in dispatches:
        for fmt in [f for f in args.formats.split(",") if f]:
            for bucket in [int(b) for b in args.buckets.split(",") if b]:
                for abl in ablations:
                    rec = {
                        "path": path,
                        "dispatch": dispatch,
                        "format": fmt,
                        "bucket": bucket,
                        "ablation": abl,
                    }
                    extra = {} if abl == "none" else {"ablation": ABLATION_SETS[abl]}
                    t0 = time.perf_counter()
                    try:
                        tokens = ep * bucket
                        abstract = (
                            jax.ShapeDtypeStruct(
                                (tokens, HIDDEN), jnp.bfloat16, sharding=rows
                            ),
                            jax.ShapeDtypeStruct(
                                (tokens, e), jnp.float32, sharding=rows
                            ),
                            *operands(fmt),
                            jax.ShapeDtypeStruct((e,), jnp.float32, sharding=repl),
                        )
                        if args.op:
                            moe, op_name = op_program(dispatch, fmt)
                            if op_name is not None:
                                rec["op_name"] = op_name
                            rank = jax.ShapeDtypeStruct(
                                (ep, 1), jnp.int32, sharding=rows
                            )
                            abstract += (rank,)
                        else:
                            moe = layer_program(dispatch, fmt, extra)
                        forced = ag_knobs.get(dispatch) or push_host.get(dispatch)
                        with tpu7x_target(), host_knobs(host, forced):
                            if on_tpu:
                                text = moe.lower(*abstract).as_text()
                            else:
                                text = (
                                    moe.trace(*abstract)
                                    .lower(lowering_platforms=("tpu",))
                                    .as_text()
                                )
                        rec["sha256"] = hashlib.sha256(text.encode()).hexdigest()
                        rec["chars"] = len(text)
                        rec["kernels"] = sorted(set(KERNEL_NAME_RE.findall(text)))
                        rec["custom_calls"] = custom_calls(text)
                        if args.text_dir:
                            os.makedirs(args.text_dir, exist_ok=True)
                            Path(
                                args.text_dir,
                                f"{path}_{dispatch}_{fmt}_{bucket}_{abl}.mlir",
                            ).write_text(text)
                        del text
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
                        rec["traceback"] = traceback.format_exc()[-3000:]
                    rec["seconds"] = round(time.perf_counter() - t0, 1)
                    result["records"].append(rec)
                    D.log(
                        f"[lower] {path} {dispatch} {fmt} b={bucket} {abl}: "
                        + (
                            f"{rec['sha256'][:16]} {rec['chars']} {rec['kernels']}"
                            if "sha256" in rec
                            else f"ERROR {rec['error'][:300]}"
                        )
                        + f" ({rec['seconds']} s)"
                    )
                    D.write_json(args.out, result)
    same_program_checks(result["records"], args.push_max_rows)
    result["finished_utc"] = D.utc_now()
    D.write_json(args.out, result)
    checks = [r["same_program_as"] for r in result["records"] if "same_program_as" in r]
    return 0 if all(c["equal"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
