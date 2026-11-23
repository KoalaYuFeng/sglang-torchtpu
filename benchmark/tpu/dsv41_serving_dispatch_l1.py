#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 E2E L1: the serving op's dispatch switch on one real V4.1 layer.

One DeepSeek-V4.1 Flash MoE layer (the checkpoint's experts, router and
correction bias), EP over the host's TPU devices, through the serving op:
``fused_ep_op._op_key`` -> ``_get_op`` -> ``_build_op`` -> ``_make_jax_fn`` (the
body ``cpu_export_jax_op`` exports) -> ``make_fused_ep_moe`` ->
``fused_ep_moe_v2``. Only torch_tpu is stood in for: the op's mesh is a JAX mesh
over the local chips, and its export is ``jax.jit`` of the same body with
``cpu_export_jax_op``'s settings (keep_unused, output sharded like x), called
with the global arrays torch_tpu hands it (rank operand included). The expert
operands are the L1 harness's (``dsv41_k32z_l1.load_weights``: the gmm
preparation for k512, the K32Z conversion for k32z; the CPU tests hold the
serving converters to the same bytes).

Per expert format and bucket (rows per EP rank):

* ``--buckets``: the all_gather and the push op, outputs compared bit for bit;
* ``--auto-buckets``: the auto op against the dispatch it must take (push up to
  ``--push-max-rows`` rows per rank, all_gather above): outputs bit for bit and
  the same fused EP kernels in the program (``same_text``: identical lowered
  text, recorded);
* every program has its dispatch's kernel (push_fused_moe / fused_ep_moe_v2);
  every output is finite with exactly zero padding rows.

Also checked: the all_gather ops keep their names, and the serving push options
are the layer benchmarks' config D (``push_combined_options``). Routings
``real`` (checkpoint router and bias) and ``uniform`` (designed 6-expert sets,
zero bias), padding rows as the L1 harness. Exits 1 if any check fails. Plain
JAX on the chips (no torch_tpu): run it under the TPU lock.

    PYTHONPATH=$T/python:$T python3 $T/benchmark/tpu/dsv41_serving_dispatch_l1.py \\
        --weights /weights/DeepSeek-V4.1-Flash --layer 3 --out dispatch_l1.json
"""

from __future__ import annotations

import os

# Never let `import torch` auto-load torch_tpu (it takes the libtpu lock).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import hashlib
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
for _p in (str(TREE / "python"), str(TREE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dsv41_fused_ep_dispatch_dist as D  # noqa: E402
import dsv41_k32z_l1 as L1  # noqa: E402

# Expert format -> dsv41_k32z_l1.load_weights' operand set.
WEIGHT_SETS = {"k512": "A", "k32z": "AZ"}
KERNEL_RE = re.compile(r"((?:push_fused_moe|fused_ep_moe_v2)\w*)")
# The main kernel each dispatch's program must carry.
MAIN_KERNEL = {"all_gather": "fused_ep_moe_v2", "push": "push_fused_moe"}
log = D.log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--formats", default="k512,k32z")
    p.add_argument(
        "--buckets",
        default="8,512,1024,2048",
        help="rows per EP rank: all_gather vs push",
    )
    p.add_argument("--auto-buckets", default="8,2048", help="rows per EP rank: auto")
    p.add_argument("--push-max-rows", type=int, default=1024, help="auto's bound")
    p.add_argument("--routings", default="real,uniform")
    p.add_argument("--pad-rows", type=int, default=2)
    p.add_argument(
        "--gate-up-bits", default="4", help="K32Z gate/up field width (serving: 4)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


class Checks:
    """Named pass/fail records, logged as they come."""

    def __init__(self):
        self.records = []

    def __call__(self, label, ok, **detail):
        self.records.append({"check": label, "ok": bool(ok), **detail})
        log(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" {detail}" if detail else ""))
        return ok

    @property
    def failed(self):
        return [r for r in self.records if not r["ok"]]


def build_ops(op_mod, jax, mesh, formats, bits, push_max_rows):
    """{(format, dispatch): (jitted op body, key, name)}, built by fused_ep_op
    itself with torch_tpu's mesh and export stood in for."""
    from jax.sharding import NamedSharding

    fe = op_mod.fe
    ep = mesh.devices.size
    (axis,) = mesh.axis_names

    def get_mesh(world_size, axis_name="ep"):
        if (world_size, axis_name) != (ep, axis):
            raise ValueError(f"unexpected op mesh {world_size} x {axis_name!r}")
        return mesh

    def export(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        # cpu_export_jax_op's jax.jit of the body, without the torch bridge.
        (out_spec,) = output_partition_specs
        return jax.jit(
            fn, keep_unused=True, out_shardings=NamedSharding(mesh, out_spec)
        )

    op_mod._get_mesh = get_mesh
    op_mod._export_op = export
    ops = {}
    for fmt in formats:
        width = bits if fmt == fe.EXPERT_FORMAT_K32Z else 0
        for dispatch in fe.DISPATCHES:
            key = op_mod._op_key(
                ep,
                fe.TOPK,
                fe.RENORMALIZE,
                fe.ROUTED_SCALING_FACTOR,
                fe.LOGICAL_INTER,
                axis,
                fmt,
                width,
                dispatch,
                push_max_rows,
            )
            ops[fmt, dispatch] = (op_mod._get_op(*key), key, op_mod._op_name(*key))
    return ops


def compile_program(op, abstract):
    """Lower + compile one op at one bucket: (compiled or None, record)."""
    rec = {}
    try:
        t0 = time.perf_counter()
        lowered = op.lower(*abstract)
        text = lowered.as_text()
        t1 = time.perf_counter()
        compiled = lowered.compile()
        rec.update(
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            chars=len(text),
            kernels=sorted(set(KERNEL_RE.findall(text))),
            lower_s=round(t1 - t0, 2),
            compile_s=round(time.perf_counter() - t1, 2),
        )
        return compiled, rec
    except Exception as exc:  # noqa: BLE001 - recorded, the run continues
        rec["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        return None, rec


def has_main_kernel(kernels, dispatch):
    """The dispatch's own fused EP kernel is in the program, the other's not."""
    other = MAIN_KERNEL["push" if dispatch == "all_gather" else "all_gather"]
    return any(k.startswith(MAIN_KERNEL[dispatch]) for k in kernels) and not any(
        k.startswith(other) for k in kernels
    )


def output_bits(a):
    return np.ascontiguousarray(a).view(np.uint8).reshape(a.shape[0], -1)


def run_bucket(args, check, ctx, bucket):
    """Every format's ops at one bucket: programs, then outputs per routing."""
    jax, jnp, ep = ctx["jax"], ctx["jnp"], ctx["ep"]
    rows_sh, repl = ctx["rows_sh"], ctx["repl"]
    tokens = ep * bucket
    want_auto = ctx["serving_fe"].resolve_dispatch(
        "auto", rows_per_rank=bucket, push_max_rows=args.push_max_rows
    )
    run = ["all_gather", "push"] if bucket in ctx["buckets"] else []
    if bucket in ctx["auto_buckets"]:
        run += [d for d in (want_auto, "auto") if d not in run]
    x = D.make_hidden(
        np.random.default_rng([args.seed, 1, bucket]),
        ep=ep,
        bucket=bucket,
        hidden=L1.HIDDEN,
        norm_w=ctx["norm_w"],
    )
    x_d = jax.device_put(x, rows_sh)
    routed = {r: (x_d, *route(args, ctx, bucket, r, x)) for r in ctx["routings"]}
    point = {"bucket": bucket, "auto_takes": want_auto, "formats": {}}
    e = L1.NUM_EXPERTS
    for fmt in ctx["formats"]:
        abstract = (
            x_d,
            jax.ShapeDtypeStruct((tokens, e), jnp.float32, sharding=rows_sh),
            *ctx["weights"][fmt],
            jax.ShapeDtypeStruct((e,), jnp.float32, sharding=repl),
            ctx["rank_d"],
        )
        compiled, programs = {}, {}
        for d in run:
            compiled[d], programs[d] = compile_program(ctx["ops"][fmt, d][0], abstract)
            log(
                f"[b={bucket}] {fmt} {d}: "
                + (
                    f"{programs[d]['lower_s']}s + {programs[d]['compile_s']}s "
                    f"{programs[d]['kernels']}"
                    if compiled[d] is not None
                    else f"FAILED {programs[d]['error'][:500]}"
                )
            )
            check(f"b={bucket} {fmt} {d} compiles", compiled[d] is not None)
        fpoint = {"programs": programs, "routings": {}}
        point["formats"][fmt] = fpoint
        for d in run:
            if compiled[d] is None:
                continue
            takes = want_auto if d == "auto" else d
            check(
                f"b={bucket} {fmt} {d} runs the {takes} kernel",
                has_main_kernel(programs[d]["kernels"], takes),
                kernels=programs[d]["kernels"],
            )
        if compiled.get("auto") is not None and compiled.get(want_auto) is not None:
            same_text = programs["auto"]["sha256"] == programs[want_auto]["sha256"]
            fpoint["auto_same_text"] = same_text
            check(
                f"b={bucket} {fmt} auto has {want_auto}'s kernels",
                programs["auto"]["kernels"] == programs[want_auto]["kernels"],
                same_text=same_text,
            )
        for routing, inputs in routed.items():
            tag = f"b={bucket} {fmt} {routing}"
            fpoint["routings"][routing] = run_routing(
                check, ctx, tag, fmt, inputs, compiled, want_auto
            )
    return point


def route(args, ctx, bucket, routing, x):
    """One routing's device inputs: (gating, score bias, pad rows, real rows)."""
    jax = ctx["jax"]
    gating, sbias, pad_index, _, _ = D.build_routing(
        routing,
        x=x,
        gate_w=ctx["gate_w"],
        bias=ctx["bias"],
        ep=ctx["ep"],
        bucket=bucket,
        num_experts=L1.NUM_EXPERTS,
        topk=L1.TOPK,
        pad_rows=args.pad_rows,
        seed=args.seed,
        jitter=args.jitter,
        zipf_s=1.2,
    )
    real = np.ones(gating.shape[0], bool)
    real[pad_index] = False
    return (
        jax.device_put(gating, ctx["rows_sh"]),
        jax.device_put(sbias, ctx["repl"]),
        pad_index,
        real,
    )


def run_routing(check, ctx, tag, fmt, inputs, compiled, want_auto):
    """Every compiled op on one routing's inputs; outputs checked and compared."""
    x_d, g_d, b_d, pad_index, real = inputs
    outs, rec = {}, {"pad_rows": int(pad_index.size)}
    for d, fn in compiled.items():
        if fn is None:
            continue
        try:
            t0 = time.perf_counter()
            out = np.asarray(fn(x_d, g_d, *ctx["weights"][fmt], b_d, ctx["rank_d"]))
            rec[f"{d}_call_s"] = round(time.perf_counter() - t0, 3)
        except Exception as exc:  # noqa: BLE001 - recorded, the run continues
            rec[f"{d}_error"] = f"{type(exc).__name__}: {exc}"[:4000]
            check(f"{tag} {d} runs", False, error=rec[f"{d}_error"][:500])
            continue
        o32 = out.astype(np.float32)
        check(
            f"{tag} {d} finite, pad rows 0",
            np.isfinite(o32).all() and not o32[pad_index].any(),
        )
        outs[d] = out
    pairs = []
    if "all_gather" in outs and "push" in outs:
        pairs.append(("push", "all_gather"))
    if "auto" in outs and want_auto in outs:
        pairs.append(("auto", want_auto))
    for a, b in pairs:
        diff_rows = (output_bits(outs[a]) != output_bits(outs[b])).any(axis=1)
        detail = {"rows_differing": int(diff_rows.sum())}
        if diff_rows.any():
            rel, worst, worst_row, max_abs = D.compare(
                outs[a].astype(np.float32), outs[b].astype(np.float32), real
            )
            detail.update(
                rel_l2=rel, worst_row_rel=worst, worst_row=worst_row, max_abs=max_abs
            )
        rec[f"{a}_vs_{b}"] = detail
        check(f"{tag} {a} bitwise = {b}", not diff_rows.any(), **detail)
    return rec


def main(argv=None):
    args = parse_args(argv)
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    fe = D._fe()
    fe.import_fused_moe_v2(isolated=True)
    import torch  # noqa: F401 - after JAX; must not bring torch_tpu along

    from sglang_tpu import dsv4_mxfp4_k32z as k32z
    from sglang_tpu.models.deepseek_v41 import fused_ep as serving_fe
    from sglang_tpu.models.deepseek_v41 import fused_ep_op as op_mod

    if "torch_tpu" in sys.modules:
        raise RuntimeError("torch_tpu got imported; it would take the TPU from JAX")
    formats = [f for f in args.formats.split(",") if f]
    unknown = sorted(set(formats) - set(WEIGHT_SETS))
    if unknown:
        raise ValueError(f"unknown formats {unknown}; known {sorted(WEIGHT_SETS)}")
    buckets = [int(b) for b in args.buckets.split(",") if b]
    auto_buckets = [int(b) for b in args.auto_buckets.split(",") if b]
    devices = jax.devices()
    ep = len(devices)
    mesh = Mesh(np.asarray(devices), ("ep",))
    check = Checks()
    result = {
        "started_utc": D.utc_now(),
        "argv": sys.argv,
        "config": {
            "layer": args.layer,
            "ep": ep,
            "formats": formats,
            "buckets_rows_per_rank": buckets,
            "auto_buckets_rows_per_rank": auto_buckets,
            "push_max_rows": args.push_max_rows,
            "routings": args.routings.split(","),
            "pad_rows_per_rank": args.pad_rows,
            "push_options": serving_fe.tuned_push_options(L1.NUM_EXPERTS // ep),
        },
        "environment": D.environment(jax, mesh),
        "checks": check.records,
        "points": [],
    }
    tree_id = TREE / "K32Z_TREE_ID"
    if tree_id.exists():
        result["tree_id"] = tree_id.read_text().strip()
    log(f"[l1] ep={ep} {devices[0].device_kind} layer={args.layer} {formats}")
    g_local = L1.NUM_EXPERTS // ep
    check(
        "serving push options = layer benchmarks' config D",
        serving_fe.tuned_push_options(g_local) == D.push_combined_options(g_local),
    )

    wargs = argparse.Namespace(
        weights=args.weights,
        layer=args.layer,
        configs=",".join(WEIGHT_SETS[f] for f in formats),
        no_reference=True,
        gate_up_bits=args.gate_up_bits,
    )
    wops, _, gate_w, bias, norm_w, bits = L1.load_weights(
        wargs, fe, k32z, jax, mesh, result
    )
    result["config"]["gate_up_bits"] = bits
    ops = build_ops(op_mod, jax, mesh, formats, bits, args.push_max_rows)
    result["ops"] = {
        f"{fmt} {d}": {"key": list(key), "name": name}
        for (fmt, d), (_, key, name) in ops.items()
    }
    # The names the all-gather ops had before the dispatch switch existed.
    head = f"sglang_tpu_pallas::dsv41_fused_ep_ep{ep}_k6_r1_s1_5_i2304"
    default_names = {
        "k512": f"{head}_c128_q512_ep",
        "k32z": f"{head}_k32z_g{bits}_c256_ep",
    }
    for fmt in formats:
        name = ops[fmt, "all_gather"][2]
        check(f"{fmt} all_gather op name kept", name == default_names[fmt], name=name)
    names = [name for _, _, name in ops.values()]
    check("op names distinct", len(set(names)) == len(names), names=names)
    D.write_json(args.out, result)

    ctx = {
        "jax": jax,
        "jnp": jnp,
        "ep": ep,
        "rows_sh": NamedSharding(mesh, P("ep")),
        "repl": NamedSharding(mesh, P()),
        "serving_fe": serving_fe,
        "buckets": buckets,
        "auto_buckets": auto_buckets,
        "formats": formats,
        "routings": [r for r in args.routings.split(",") if r],
        "weights": {f: wops[WEIGHT_SETS[f]] for f in formats},
        "ops": ops,
        "gate_w": gate_w,
        "bias": bias,
        "norm_w": norm_w,
    }
    ctx["rank_d"] = jax.device_put(
        np.arange(ep, dtype=np.int32).reshape(ep, 1), ctx["rows_sh"]
    )
    for bucket in sorted(set(buckets) | set(auto_buckets)):
        try:
            result["points"].append(run_bucket(args, check, ctx, bucket))
        except Exception as exc:  # noqa: BLE001 - recorded, the run continues
            check(f"b={bucket} runs", False, error=f"{type(exc).__name__}: {exc}"[:500])
            result["points"].append(
                {"bucket": bucket, "traceback": traceback.format_exc()[-4000:]}
            )
        D.write_json(args.out, result)
    result["finished_utc"] = D.utc_now()
    result["summary"] = {
        "checks": len(check.records),
        "failed": [r["check"] for r in check.failed],
    }
    D.write_json(args.out, result)
    log(f"[l1] {len(check.records)} checks, {len(check.failed)} failed")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
