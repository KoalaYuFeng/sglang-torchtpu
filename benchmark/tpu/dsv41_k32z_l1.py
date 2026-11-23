#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 N2a/N2b L1: one real DeepSeek-V4.1 Flash MoE layer, A / A+Z / D / D+Z.

    A   fused_ep_moe_v2, all-gather dispatch, K512 experts (the gmm preparation,
        block-512 FP4, 128-row tiles): the #2267 serving path
    AZ  the same layer and dispatch with K32Z experts (sglang_tpu/
        dsv4_mxfp4_k32z.py, lossless block-32 codes + shift tables, 256-row
        tiles)
    D   push dispatch with #2102's tuned bundle (the dispatch x routing
        harness's config D: push_combined_options, dispatch_ahead = 48), K512
    DZ  the same push layer with K32Z experts

A config may carry a benchmark-only ablation variant, ``D.comm`` etc. (see
VARIANTS; push configs only, since A's disp_off needs the compute stage):
those run once per point (compile + one call, outputs invalid by design) and
are neither validated nor timed.

Both run through ``fused_ep.make_fused_ep_moe`` on a 1-D ("ep",) mesh over the
host's TPU devices (plain JAX, no torch_tpu). Weights are read from the
checkpoint once per process: every rank's 48 experts go through the gmm
preparation (A) and the K32Z conversion (AZ), as serving does.

Reference: one FP32 program on the chips with the EXACT checkpoint (K32)
weights, value * 2^(e - 127) from the checkpoint's codes and exponents, at the
kernel's rounding points: x row-quantized to FP8 (rowquant_fp8), gate/up
accumulated in f32 (HIGHEST), clamped SwiGLU to bf16, row-quantized to FP8,
down in f32, times the row scale, to bf16, wire row-quantized to FP8, then the
router-weighted sum in f32. Router: the f32 reference route (sqrtsoftplus +
bias, top-6, renormalize, x1.5). Both configs are compared to it (rel L2 over
real rows, worst row, max abs), plus finiteness and exact-zero pad rows.

Timing (first look): per bucket and routing, ``--trials`` rounds, each
running every config in turn: warmups, the median of ``--iters`` synchronous
calls, the mean of ``--pipelined-iters`` back-to-back calls. Optional
jax.profiler traces (``--profile-dir``) are digested for the kernels' device
time per call and for copy/transpose ops (per-call relayouts); the compiled
HLO is scanned for copies of the uint8 tables. ``--vmem-probe`` bisects the
smallest VMEM limit each config compiles at (the compiler's figure).

    PYTHONPATH=$T/python:$T python3 $T/benchmark/tpu/dsv41_k32z_l1.py \\
        --weights /weights/DeepSeek-V4.1-Flash --layer 3 --out l1.json
"""

from __future__ import annotations

import os

# Never let `import torch` auto-load torch_tpu (it takes the libtpu lock).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import contextlib
import glob
import gzip
import json
import re
import statistics
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

import dsv41_fused_ep_dispatch_dist as D  # noqa: E402  (ported L1 helpers)

NUM_EXPERTS, HIDDEN, INTER, TOPK = 384, 5120, 2304, 6
CONFIGS = {"A": "k512", "AZ": "k32z", "D": "k512", "DZ": "k32z"}
PUSH_CONFIGS = ("D", "DZ")
# Benchmark-only kernel ablations (fused_moe_v2.kernel.ABLATIONS) by the name
# the Step 7 breakdown uses for them.
VARIANTS = {
    "full": (),
    "comm": ("w_off", "ffn_off"),
    "comp": ("disp_off", "ret_off"),
    "base": ("w_off", "ffn_off", "ret_off", "disp_off"),
    "a2w": ("ffn_off",),
    "nowt": ("w_off",),
    "ret": ("ret_off",),
    "disp": ("disp_off",),
}


def split_config(c):
    """("D", "comm") for "D.comm", ("A", None) for "A"; refuses unknowns."""
    base, _, variant = c.partition(".")
    if base not in CONFIGS:
        raise ValueError(f"unknown config {c!r}; known {sorted(CONFIGS)}")
    if variant and (variant not in VARIANTS or base not in PUSH_CONFIGS):
        raise ValueError(
            f"config {c!r}: variants {sorted(VARIANTS)} take a push config "
            f"{PUSH_CONFIGS}"
        )
    return base, variant or None


def weight_key(c):
    """The operand set a config runs on: "A" (K512) or "AZ" (K32Z)."""
    return "A" if CONFIGS[split_config(c)[0]] == "k512" else "AZ"


def make_moe(fe, package, mesh, c):
    """The jitted layer of config `c` (fused_ep.make_fused_ep_moe)."""
    base, variant = split_config(c)
    kernel_fn = None
    if base in PUSH_CONFIGS:
        g_local = NUM_EXPERTS // mesh.devices.size
        kernel_fn = D.make_kernel_fn(
            package.fused_ep_moe_v2, "push", D.push_combined_options(g_local)
        )
    extra = {"ablation": frozenset(VARIANTS[variant])} if variant else {}
    return fe.make_fused_ep_moe(
        mesh,
        logical_intermediate_size=INTER,
        expert_format=CONFIGS[base],
        kernel_fn=kernel_fn,
        **extra,
    )


BUCKETS = "8,16,32,64,512,1024,2048"
log = D.log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--configs", default="A,AZ")
    p.add_argument("--buckets", default=BUCKETS, help="rows per rank")
    p.add_argument("--routings", default="uniform,real")
    p.add_argument("--pad-rows", type=int, default=2)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--pipelined-iters", type=int, default=50)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--profile-calls", type=int, default=5)
    p.add_argument("--profile-routings", default="uniform")
    p.add_argument("--no-reference", action="store_true")
    p.add_argument(
        "--gate-up-bits", default="auto", help="2, 4 or auto (2 unless a shift > 3)"
    )
    p.add_argument(
        "--vmem-probe", default="", help="buckets to bisect the VMEM figure at"
    )
    p.add_argument("--vmem-steps", type=int, default=6)
    p.add_argument("--hlo-dir", default=None, help="write each compiled HLO here")
    p.add_argument(
        "--k32z-rungs",
        default="",
        help="benchmark only: the K32Z tile heights (default all multiples of 32); "
        "must keep 128 as a boundary, where the slab waits move to the visit head",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Weights.
# --------------------------------------------------------------------------- #
def load_weights(args, fe, k32z, jax, mesh, result):
    """Every rank's experts -> device operands of A and AZ, plus the exact
    exponents the reference needs. Returns ({cfg: (w1, w2, s1, s2)}, (e1, e2),
    gate_w, bias, norm_w, bits)."""
    import torch

    ep = mesh.devices.size
    g = NUM_EXPERTS // ep
    ckpt = D.Checkpoint(args.weights)
    layer = args.layer
    gate_w = ckpt.f32(f"layers.{layer}.ffn.gate.weight", (NUM_EXPERTS, HIDDEN))
    bias = ckpt.f32(f"layers.{layer}.ffn.gate.bias", (NUM_EXPERTS,))
    norm_name = f"layers.{layer}.ffn_norm.weight"
    norm_w = (
        ckpt.f32(norm_name, (HIDDEN,))
        if ckpt.has(norm_name)
        else np.ones(HIDDEN, np.float32)
    )
    want = sorted({weight_key(c) for c in args.configs.split(",") if c})
    if not args.no_reference and "AZ" not in want:
        # The reference reads the K32Z codes (the checkpoint's own e2m1).
        want.append("AZ")
    shards = {c: [[], [], [], []] for c in want}
    exps = [[], []]
    bits = None if args.gate_up_bits == "auto" else int(args.gate_up_bits)
    max_shift = [0, 0]
    log_rows = []
    t_all = time.perf_counter()
    for r in range(ep):
        t0 = time.perf_counter()
        experts = list(range(r * g, (r + 1) * g))
        (w13, s13, w2, s2), _ = D.read_rank_experts(ckpt, layer, experts, HIDDEN, INTER)
        t1 = time.perf_counter()
        e13 = k32z.e8m0_exponents(s13)  # int32 [g, 2I, H/32]
        e2 = k32z.e8m0_exponents(s2)
        for i, e in enumerate((e13, e2)):
            max_shift[i] = max(max_shift[i], int((e - e.amin(-1, keepdim=True)).max()))
        dev = mesh.devices.flat[r]
        # The reference's exact exponents, [g, K/32, N] uint8.
        exps[0].append(
            jax.device_put(
                e13.transpose(1, 2).contiguous().to(torch.uint8).numpy(), dev
            )
        )
        exps[1].append(
            jax.device_put(e2.transpose(1, 2).contiguous().to(torch.uint8).numpy(), dev)
        )
        row = {"rank": r, "read_s": round(t1 - t0, 2)}
        if "AZ" in want:
            t2 = time.perf_counter()
            use = bits if bits is not None else 2
            try:
                ops = k32z.convert_mxfp4_experts_to_k32z(
                    w13, s13, w2, s2, gate_up_bits=use
                )
            except ValueError:
                if bits is not None:
                    raise
                # auto: a gate/up shift above 3 needs the 4-bit fields; every
                # rank must agree, so the width is fixed from here on and the
                # ranks before are redone.
                log(f"[weights] rank {r}: gate/up shift > 3, switching to 4-bit fields")
                bits = 4
                return load_weights_with_bits(args, fe, k32z, jax, mesh, result, bits)
            bits = use
            row["k32z_convert_s"] = round(time.perf_counter() - t2, 2)
            codes1 = fe.unpack_kpacked_fp4_codes(ops.w13.numpy())
            codes2 = fe.unpack_kpacked_fp4_codes(ops.w2.numpy())
            w1d, how1 = D.put_float4(fe.codes_to_float4(codes1), dev)
            w2d, how2 = D.put_float4(fe.codes_to_float4(codes2), dev)
            s1 = ops.w13_shift.numpy()
            s2t = ops.w2_shift.numpy()
            shards["AZ"][0].append(w1d)
            shards["AZ"][1].append(w2d)
            shards["AZ"][2].append(jax.device_put(s1.reshape(-1, s1.shape[-1]), dev))
            shards["AZ"][3].append(jax.device_put(s2t.reshape(-1, s2t.shape[-1]), dev))
            row["k32z_put"] = [how1, how2]
            row["k32z_max_shift"] = list(ops.max_shift)
            del ops, codes1, codes2
        if "A" in want:
            t3 = time.perf_counter()
            prep = fe.run_gmm_prep(w13.clone(), s13.clone(), w2.clone(), s2.clone())
            kw = fe.gmm_to_fused_ep_weights(
                prep["w13_weight"],
                prep["w13_weight_scale"],
                prep["w2_weight"],
                prep["w2_weight_scale"],
                scale_layout="flat",
            )
            row["k512_prep_s"] = round(time.perf_counter() - t3, 2)
            w1d, how1 = D.put_float4(kw.w1, dev)
            w2d, how2 = D.put_float4(kw.w2, dev)
            shards["A"][0].append(w1d)
            shards["A"][1].append(w2d)
            shards["A"][2].append(jax.device_put(kw.w1_scale, dev))
            shards["A"][3].append(jax.device_put(kw.w2_scale, dev))
            row["k512_put"] = [how1, how2]
            del prep, kw
        del w13, s13, w2, s2, e13, e2
        row["total_s"] = round(time.perf_counter() - t0, 2)
        log_rows.append(row)
        log(f"[weights] {row}")
    ckpt.close()

    def glob_(parts, shape):
        return D.assemble(mesh, parts, shape)

    ops = {}
    pad_i = -(-INTER // 512) * 512
    for c in want:
        w1s, w2s, s1s, s2s = shards[c]
        e = NUM_EXPERTS
        if c == "A":
            ops[c] = (
                glob_(w1s, (e, HIDDEN, 2 * pad_i)),
                glob_(w2s, (e, pad_i, HIDDEN)),
                glob_(s1s, (e * fe.fp4_scale_rows(HIDDEN, padded=True), 2 * pad_i)),
                glob_(s2s, (e * fe.fp4_scale_rows(pad_i, padded=True), HIDDEN)),
            )
        else:
            r1, c1 = s1s[0].shape
            r2, c2 = s2s[0].shape
            ops[c] = (
                glob_(w1s, (e, HIDDEN, 2 * INTER)),
                glob_(w2s, (e, INTER, HIDDEN)),
                glob_(s1s, (ep * r1, c1)),
                glob_(s2s, (ep * r2, c2)),
            )
    ref_exps = (
        glob_(exps[0], (NUM_EXPERTS, HIDDEN // 32, 2 * INTER)),
        glob_(exps[1], (NUM_EXPERTS, INTER // 32, HIDDEN)),
    )
    result["weights"] = {
        "total_s": round(time.perf_counter() - t_all, 1),
        "ranks": log_rows,
        "layer_max_shift_gate_up_down": max_shift,
        "gate_up_bits": bits,
        "operands": {c: [[str(a.dtype), list(a.shape)] for a in ops[c]] for c in ops},
    }
    return ops, ref_exps, gate_w, bias, norm_w, bits


def load_weights_with_bits(args, fe, k32z, jax, mesh, result, bits):
    args.gate_up_bits = str(bits)
    return load_weights(args, fe, k32z, jax, mesh, result)


# --------------------------------------------------------------------------- #
# The FP32 reference with exact K32 weights at the kernel's rounding points.
# --------------------------------------------------------------------------- #
def make_reference(mesh, fe, kernel):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    (axis,) = mesh.axis_names
    hi = lax.Precision.HIGHEST

    def local(x, gating, bias, w1c, w2c, e1, e2):
        g_local = w1c.shape[0]
        weights, ids, valid = fe.route(
            gating, bias, topk=TOPK, renormalize=True, routed_scaling_factor=1.5
        )
        xq, xs = kernel.rowquant_fp8(x.astype(jnp.bfloat16))
        xq32 = xq.astype(jnp.float32)
        first = lax.axis_index(axis) * g_local

        def dense(codes, exps, e):
            c = lax.dynamic_index_in_dim(codes, e, 0, keepdims=False).astype(
                jnp.float32
            )
            s = jnp.exp2(
                lax.dynamic_index_in_dim(exps, e, 0, keepdims=False).astype(jnp.float32)
                - 127.0
            )
            k, n = c.shape
            return (c.reshape(k // 32, 32, n) * s[:, None, :]).reshape(k, n)

        def body(e, acc):
            w1 = dense(w1c, e1, e)
            w2 = dense(w2c, e2, e)
            coef = jnp.sum(jnp.where(ids == first + e, weights, 0.0), -1, keepdims=True)
            acc1 = jnp.dot(xq32, w1, precision=hi) * xs
            mid = fe.clamped_swiglu(acc1[:, :INTER], acc1[:, INTER:]).astype(
                jnp.bfloat16
            )
            mid_q, mid_s = kernel.rowquant_fp8(mid)
            down = (
                jnp.dot(mid_q.astype(jnp.float32), w2, precision=hi) * mid_s
            ).astype(jnp.bfloat16)
            q, s = kernel.rowquant_fp8(down)
            return acc + coef * (q.astype(jnp.float32) * s)

        out = lax.fori_loop(0, g_local, body, jnp.zeros(x.shape, jnp.float32))
        return jnp.where(valid, lax.psum(out, axis), 0.0)

    sm = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(axis), P(axis), P(axis), P(axis)),
        out_specs=P(),
        check_vma=False,
    )
    return jax.jit(sm)


# --------------------------------------------------------------------------- #
# Profiles and compiled HLO.
# --------------------------------------------------------------------------- #
# The TensorCore kernels a digest reports per call: the fused EP kernels (v2
# and push), the combine, and the push layer's exchange / table kernels.
KERNEL_PREFIXES = (
    "fused_ep_moe_v2",
    "push_fused_moe",
    "moe_v2_combine",
    "moe_tc_routing_exchange",
    "moe_push_tables",
)


def digest_trace(pdir, calls):
    """Device ops of one traced point: per-name count and total, the fused EP
    kernels' per-call device time (max over cores), copy/transpose ops."""
    files = glob.glob(os.path.join(pdir, "**", "*.trace.json.gz"), recursive=True)
    if not files:
        return {"error": "no trace.json.gz"}
    with gzip.open(sorted(files)[-1]) as f:
        tr = json.load(f)
    procs = {}
    for ev in tr.get("traceEvents", []):
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            procs[ev["pid"]] = ev.get("args", {}).get("name", "")
    tpu = {
        pid for pid, name in procs.items() if "TPU" in name and "SparseCore" not in name
    }
    per_core = {}
    ops = {}
    for ev in tr.get("traceEvents", []):
        if ev.get("ph") != "X" or ev.get("pid") not in tpu:
            continue
        name, dur = ev.get("name", ""), float(ev.get("dur", 0.0))
        o = ops.setdefault(name, [0, 0.0])
        o[0] += 1
        o[1] += dur
        if name.startswith(KERNEL_PREFIXES):
            per_core.setdefault(name, {}).setdefault(ev["pid"], []).append(dur)
    kernels = {}
    for name, cores in per_core.items():
        meds = [statistics.median(v) for v in cores.values()]
        kernels[name] = {
            "per_call_us_max_core_median": round(max(meds), 2),
            "per_call_us_min_core_median": round(min(meds), 2),
            "cores": len(cores),
            "calls_per_core": sorted({len(v) for v in cores.values()}),
        }
    top = sorted(ops.items(), key=lambda kv: -kv[1][1])[:25]
    relayout = {
        n: {"count": c, "us_per_call_all_cores": round(t / max(1, calls), 2)}
        for n, (c, t) in ops.items()
        if re.search(r"copy|transpose", n, re.I)
    }
    return {
        "kernels": kernels,
        "top_ops": [[n, c, round(t / max(1, calls), 2)] for n, (c, t) in top],
        "copy_transpose_ops": relayout,
    }


def hlo_table_copies(text):
    """Lines of the compiled HLO that copy or transpose a uint8 array."""
    out = []
    for line in text.splitlines():
        if "u8[" in line and re.search(
            r"=\s*u8\[[^\]]*\]\S*\s+(copy|transpose)\(", line
        ):
            out.append(line.strip()[:300])
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    fe = D._fe()
    package = fe.import_fused_moe_v2(isolated=True)
    import torch  # noqa: F401 - after JAX; must not bring torch_tpu along

    from sglang_tpu import dsv4_mxfp4_k32z as k32z
    from sglang_tpu.moe.pallas.fused_moe_v2 import host, kernel
    from sglang_tpu.moe.pallas.fused_moe_v2 import layer as v2_layer
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import kernel as push_kernel
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer as push_layer

    if "torch_tpu" in sys.modules:
        raise RuntimeError("torch_tpu got imported; it would take the TPU from JAX")
    if args.k32z_rungs:
        rungs = tuple(int(r) for r in args.k32z_rungs.split(","))
        if host.K32Z_STREAMED_ROWS not in rungs or rungs != tuple(sorted(rungs)):
            raise ValueError(f"K32Z rungs {rungs} must be ascending and keep 128")
        host.k32z_tile_height_rungs = lambda tile_m, _rungs=None: tuple(
            r for r in rungs if r <= tile_m
        )
        log(f"[l1] benchmark-only K32Z rungs {rungs}")
    devices = jax.devices()
    ep = len(devices)
    mesh = Mesh(np.asarray(devices), ("ep",))
    configs = [c for c in args.configs.split(",") if c]
    for c in configs:
        split_config(c)
    variants = {c for c in configs if split_config(c)[1]}
    buckets = [int(b) for b in args.buckets.split(",") if b]
    routings = [r for r in args.routings.split(",") if r]
    result = {
        "started_utc": D.utc_now(),
        "argv": sys.argv,
        "config": {
            "layer": args.layer,
            "ep": ep,
            "configs": {
                c: {
                    "format": CONFIGS[split_config(c)[0]],
                    "dispatch": "push"
                    if split_config(c)[0] in PUSH_CONFIGS
                    else "all_gather",
                    "ablation": sorted(VARIANTS.get(split_config(c)[1] or "full")),
                }
                for c in configs
            },
            "push_options": D.push_combined_options(NUM_EXPERTS // ep),
            "buckets_rows_per_rank": buckets,
            "routings": routings,
            "pad_rows_per_rank": args.pad_rows,
            "trials": args.trials,
            "warmup": args.warmup,
            "iters_sync": args.iters,
            "pipelined_iters": args.pipelined_iters,
            "k32z_rungs": args.k32z_rungs or "default",
        },
        "environment": D.environment(jax, mesh),
        "points": [],
    }
    tree_id = TREE / "K32Z_TREE_ID"
    if tree_id.exists():
        result["tree_id"] = tree_id.read_text().strip()
    log(
        f"[l1] ep={ep} {devices[0].device_kind} layer={args.layer} configs={configs} buckets={buckets}"
    )
    wops, ref_exps, gate_w, bias, norm_w, bits = load_weights(
        args, fe, k32z, jax, mesh, result
    )
    # Each config's operands: the K512 or the K32Z set.
    ops = {c: wops[weight_key(c)] for c in configs}
    result["config"]["gate_up_bits"] = bits
    D.write_json(args.out, result)
    repl = NamedSharding(mesh, P())
    rows_sh = NamedSharding(mesh, P("ep"))
    bias_d = jax.device_put(bias, repl)
    zero_bias_d = jax.device_put(np.zeros(NUM_EXPERTS, np.float32), repl)
    moes = {c: make_moe(fe, package, mesh, c) for c in configs}
    reference = None if args.no_reference else make_reference(mesh, fe, kernel)

    for bucket in buckets:
        tokens = ep * bucket
        x = D.make_hidden(
            np.random.default_rng([args.seed, 1, bucket]),
            ep=ep,
            bucket=bucket,
            hidden=HIDDEN,
            norm_w=norm_w,
        )
        x_d = jax.device_put(x, rows_sh)
        x_rep = jax.device_put(x, repl)
        compiled = {}
        cinfo = {}
        for c in configs:
            abstract = (
                jax.ShapeDtypeStruct((tokens, HIDDEN), x_d.dtype, sharding=rows_sh),
                jax.ShapeDtypeStruct(
                    (tokens, NUM_EXPERTS), jnp.float32, sharding=rows_sh
                ),
                *ops[c],
                jax.ShapeDtypeStruct((NUM_EXPERTS,), jnp.float32, sharding=repl),
            )
            try:
                t0 = time.perf_counter()
                lowered = moes[c].lower(*abstract)
                t1 = time.perf_counter()
                compiled[c] = lowered.compile()
                t2 = time.perf_counter()
                text = compiled[c].as_text()
                cinfo[c] = {
                    "lower_s": round(t1 - t0, 2),
                    "compile_s": round(t2 - t1, 2),
                    "kernels": D.kernel_names(lowered, compiled[c]),
                    "u8_copies_in_hlo": hlo_table_copies(text),
                }
                if args.hlo_dir:
                    os.makedirs(args.hlo_dir, exist_ok=True)
                    Path(args.hlo_dir, f"{c}_{bucket}.hlo.txt").write_text(text)
                del lowered, text
                log(
                    f"[b={bucket}] compiled {c}: {cinfo[c]['lower_s']}s + {cinfo[c]['compile_s']}s {cinfo[c]['kernels']}"
                )
            except Exception as exc:  # noqa: BLE001
                cinfo[c] = {
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "traceback": traceback.format_exc()[-4000:],
                }
                log(f"[b={bucket}] compile {c} FAILED: {cinfo[c]['error'][:500]}")
        for routing in routings:
            point = {
                "bucket": bucket,
                "routing": routing,
                "compile": cinfo,
                "configs": {},
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
                    num_experts=NUM_EXPERTS,
                    topk=TOPK,
                    pad_rows=args.pad_rows,
                    seed=args.seed,
                    jitter=args.jitter,
                    zipf_s=1.2,
                )
                real = np.ones(tokens, bool)
                real[pad_index] = False
                rows = np.nonzero(real)[0]
                sel, gap = D.host_select(gating[rows], sbias, TOPK)
                point["selection"] = {
                    "near_tie_rows_gap_lt_1e-5": int((gap < 1e-5).sum()),
                    "min_gap": float(gap.min()) if gap.size else None,
                }
                if designed is not None:
                    same = np.sort(sel, 1) == np.sort(designed, 1)
                    point["selection"]["mismatched_rows"] = int((~same.all(1)).sum())
                st = D.routing_summary(
                    sel, rows, ep=ep, bucket=bucket, num_experts=NUM_EXPERTS
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
                want = None
                if reference is not None:
                    t0 = time.perf_counter()
                    want = np.asarray(
                        reference(
                            x_rep,
                            jax.device_put(gating, repl),
                            b_d,
                            wops["AZ"][0],
                            wops["AZ"][1],
                            *ref_exps,
                        )
                    )
                    point["reference_s"] = round(time.perf_counter() - t0, 2)
            except Exception as exc:  # noqa: BLE001
                point["error"] = f"{type(exc).__name__}: {exc}"[:4000]
                point["traceback"] = traceback.format_exc()[-4000:]
                log(f"[b={bucket} {routing}] setup FAILED: {point['error'][:500]}")
                D.write_json(args.out, result)
                continue
            ready = [c for c in configs if c in compiled]
            outs = {}
            for c in ready:
                pc = point["configs"].setdefault(c, {"trials": []})
                try:
                    out = compiled[c](x_d, g_d, *ops[c], b_d)
                    out.block_until_ready()
                    o = np.asarray(out.astype(jnp.float32))
                    pc["nonfinite"] = int((~np.isfinite(o)).sum())
                    pc["pad_rows_exact_zero"] = bool(np.all(o[pad_index] == 0))
                    if c in variants:
                        # Ran once; the output is invalid by design.
                        pc["variant_ran"] = True
                        continue
                    if want is not None:
                        rel, worst, wi, mx = D.compare(o, want, rows)
                        pc["vs_reference"] = {
                            "rel_l2": rel,
                            "worst_row_rel": worst,
                            "worst_row_near_tie": bool(gap[wi] < 1e-5),
                            "max_abs": mx,
                        }
                    outs[c] = o
                except Exception as exc:  # noqa: BLE001
                    pc["error"] = f"{type(exc).__name__}: {exc}"[:4000]
                    pc["traceback"] = traceback.format_exc()[-4000:]
                    log(f"[b={bucket} {routing} {c}] run FAILED: {pc['error'][:500]}")
            for a, b in (("AZ", "A"), ("DZ", "D"), ("D", "A"), ("DZ", "AZ")):
                if a in outs and b in outs:
                    rel, worst, wi, mx = D.compare(outs[a], outs[b], rows)
                    point[f"{a}_vs_{b}"] = {
                        "rel_l2": rel,
                        "worst_row_rel": worst,
                        "max_abs": mx,
                    }
            del outs, want
            ok = [
                c
                for c in ready
                if "error" not in point["configs"].get(c, {}) and c not in variants
            ]
            for trial in range(args.trials):
                for c in ok:
                    _, lat = D.run_point(
                        compiled[c],
                        (x_d, g_d, *ops[c], b_d),
                        warmup=args.warmup if trial == 0 else 1,
                        iters=args.iters,
                        pipelined_iters=args.pipelined_iters,
                    )
                    point["configs"][c]["trials"].append(lat)
            if args.profile_dir and routing in args.profile_routings.split(","):
                for c in ok:
                    pdir = os.path.join(args.profile_dir, f"{c}_{bucket}_{routing}")
                    try:
                        with jax.profiler.trace(pdir):
                            for _ in range(args.profile_calls):
                                compiled[c](x_d, g_d, *ops[c], b_d).block_until_ready()
                        point["configs"][c]["profile"] = digest_trace(
                            pdir, args.profile_calls
                        )
                    except Exception as exc:  # noqa: BLE001
                        point["configs"][c]["profile"] = {
                            "error": f"{type(exc).__name__}: {exc}"[:1000]
                        }
            parts = []
            for c in ok:
                pc = point["configs"][c]
                pipe = [t["pipelined_mean"] for t in pc["trials"]]
                med = [t["median"] for t in pc["trials"]]
                pc["pipelined_mean_ms_trials"] = pipe
                pc["sync_median_ms_trials"] = med
                vr = pc.get("vs_reference", {})
                kern = pc.get("profile", {}).get("kernels", {})
                kus = {k: v["per_call_us_max_core_median"] for k, v in kern.items()}
                parts.append(
                    f"{c}: pipe {statistics.median(pipe):.4f} ms {pipe} sync {statistics.median(med):.4f} "
                    f"ref rel {vr.get('rel_l2', float('nan')):.4f} worst {vr.get('worst_row_rel', float('nan')):.4f} "
                    f"nonfinite {pc.get('nonfinite')} pad0 {pc.get('pad_rows_exact_zero')} {kus}"
                )
            ran = [c for c in ready if point["configs"].get(c, {}).get("variant_ran")]
            pairs = {k: round(v["rel_l2"], 5) for k, v in point.items() if "_vs_" in k}
            log(
                f"[b={bucket} {routing}] "
                + " | ".join(parts)
                + f" | {pairs}"
                + (f" | variants ran once: {ran}" if variants else "")
            )
            del g_d
            D.write_json(args.out, result)
        del compiled, x_d, x_rep

    if args.vmem_probe:
        result["vmem_probe"] = vmem_probe(
            args,
            fe,
            package,
            host,
            kernel,
            (v2_layer, push_layer),
            push_kernel,
            mesh,
            ops,
            [c for c in configs if c not in variants],
        )
    result["finished_utc"] = D.utc_now()
    D.write_json(args.out, result)
    log(f"[l1] done -> {args.out}")
    return 0


def vmem_probe(
    args, fe, package, host, kernel, layers, push_kernel, mesh, ops, configs
):
    """The smallest VMEM limit each config's kernel compiles at, bisected over
    the compiler's own answer (the figure host.py's bound is checked against),
    plus the scoped-allocation message at 1 MiB."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    out = []
    ep = mesh.devices.size
    real_limit = host.vmem_limit
    rows_sh = NamedSharding(mesh, P("ep"))
    repl = NamedSharding(mesh, P())

    @contextlib.contextmanager
    def limit(value):
        # Only the pallas_calls' own limits (host.vmem_limit() with no device
        # record); fit_weight_slots passes its record and keeps the real one.
        # The combine kernel reads the same limit, so its cache is cleared too.
        host.vmem_limit = lambda info=None: value if info is None else real_limit(info)
        caches = [m._LAYER_SM_CACHE for m in layers] + [
            kernel._BUILD_CACHE,
            kernel._COMBINE_CACHE,
            push_kernel._BUILD_CACHE,
        ]
        for cache in caches:
            cache.clear()
        try:
            yield
        finally:
            host.vmem_limit = real_limit
            for cache in caches:
                cache.clear()

    for c in configs:
        for bucket in [int(b) for b in args.vmem_probe.split(",") if b]:
            tokens = ep * bucket
            abstract = (
                jax.ShapeDtypeStruct((tokens, HIDDEN), jnp.bfloat16, sharding=rows_sh),
                jax.ShapeDtypeStruct(
                    (tokens, NUM_EXPERTS), jnp.float32, sharding=rows_sh
                ),
                *ops[c],
                jax.ShapeDtypeStruct((NUM_EXPERTS,), jnp.float32, sharding=repl),
            )

            def fits(value):
                with limit(value):
                    fn = make_moe(fe, package, mesh, c)
                    try:
                        fn.lower(*abstract).compile()
                        return True, None
                    except Exception as exc:  # noqa: BLE001
                        return False, repr(exc)[:3000]

            t0 = time.perf_counter()
            _, msg_small = fits(1 << 20)
            scoped = re.search(
                r"Scoped allocation with size ([\d.]+[KMG]?)", msg_small or ""
            )
            lo, hi = 40 << 20, real_limit()
            ok_hi, msg_hi = fits(hi)
            n = 2
            while ok_hi and n < 2 + args.vmem_steps and hi - lo > (256 << 10):
                mid = (lo + hi) // 2
                ok, _ = fits(mid)
                n += 1
                if ok:
                    hi = mid
                else:
                    lo = mid
            rec = {
                "config": c,
                "bucket": bucket,
                "limit_bytes": real_limit(),
                "fits_at_limit": ok_hi,
                "min_fit_bytes_upper": hi if ok_hi else None,
                "max_fail_bytes": lo,
                "compiles": n,
                "seconds": round(time.perf_counter() - t0, 1),
                "scoped_allocation_at_1MiB": scoped.group(1) if scoped else None,
                "msg_at_1MiB": msg_small,
                "msg_at_limit": msg_hi,
            }
            out.append(rec)
            log(
                f"[vmem] {c} b={bucket}: fits {ok_hi} min fit <= {hi / 2**20:.2f} MiB (fail {lo / 2**20:.2f}); scoped {rec['scoped_allocation_at_1MiB']}"
            )
    return out


if __name__ == "__main__":
    sys.exit(main())
