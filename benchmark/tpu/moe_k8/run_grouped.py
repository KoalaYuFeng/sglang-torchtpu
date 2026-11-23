# SPDX-License-Identifier: Apache-2.0
"""K8 single-TensorCore driver for k32z_grouped_ffn (pod only).

  prep:  python3 run_grouped.py prep --layer 3 --rank 5 [--bits 4]
         one EP rank's 48 experts of a checkpoint layer as K32Z operands (the harness's own conversion:
         dsv41_fused_ep_dispatch_dist.read_rank_experts + dsv4_mxfp4_k32z.convert_mxfp4_experts_to_k32z) ->
         $WCACHE/L<layer>_r<rank>_gu<bits>/{w1c,w2c,w1s,w2s}.npy (codes uint8 one per byte, tables uint8)
  run:   python3 run_grouped.py run --layer 3 --rank 5 --configs CFG [--validate N] [--time] [--trace DIR] --out J
         CFG: comma list of  rt:<bucket>:<layer>[:<rank>]  (per-expert rows of that EP rank in the K8 replay
         stats, default its critical rank) or  me:<M_e>  (every expert M_e rows) or  sizes:<file.json>.
         --validate N: per hit expert, up to N sampled rows against the K32Z reference (jnp f32 HIGHEST on the
         same device: the K32Z decode of every weight (e2m1 -> e5m2 + 4 d), the same requantization points);
         --time: wall clock of 20 calls after 5 warmups (as MaxKernel's harness); --trace: 5 calls in a
         profiler session, the kernel op's device time.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TPU_BENCH = os.path.dirname(HERE)
TREE = os.path.dirname(os.path.dirname(TPU_BENCH))
for p in (HERE, TPU_BENCH, os.path.join(TREE, "python"), TREE):
    if p not in sys.path:
        sys.path.insert(0, p)

WCACHE = os.environ.get("WCACHE", "/root/scratch/k8/wcache")
STATS = os.environ.get("K8_STATS", "/root/scratch/k8/res/replay/replay_stats.json")
WEIGHTS = "/weights/DeepSeek-V4.1-Flash"
G = 48


def cache_dir(layer, rank, bits):
    return os.path.join(WCACHE, f"L{layer}_r{rank}_gu{bits}")


def prep(a):
    import dsv41_fused_ep_dispatch_dist as D
    from sglang_tpu import dsv4_mxfp4_k32z as k32z

    fe = D._fe()
    d = cache_dir(a.layer, a.rank, a.bits)
    os.makedirs(d, exist_ok=True)
    ckpt = D.Checkpoint(WEIGHTS)
    experts = list(range(a.rank * G, (a.rank + 1) * G))
    t0 = time.time()
    (w13, s13, w2, s2), _ = D.read_rank_experts(ckpt, a.layer, experts, 5120, 2304)
    ops = k32z.convert_mxfp4_experts_to_k32z(w13, s13, w2, s2, gate_up_bits=a.bits)
    np.save(os.path.join(d, "w1c.npy"), fe.unpack_kpacked_fp4_codes(ops.w13.numpy()))
    np.save(os.path.join(d, "w2c.npy"), fe.unpack_kpacked_fp4_codes(ops.w2.numpy()))
    np.save(os.path.join(d, "w1s.npy"), ops.w13_shift.numpy())
    np.save(os.path.join(d, "w2s.npy"), ops.w2_shift.numpy())
    norm = f"layers.{a.layer}.ffn_norm.weight"
    np.save(os.path.join(d, "norm.npy"), ckpt.f32(norm, (5120,)) if ckpt.has(norm) else np.ones(5120, np.float32))
    print(json.dumps({"dir": d, "max_shift": list(ops.max_shift), "s": round(time.time() - t0, 1),
                      "w1s": list(ops.w13_shift.shape), "w2s": list(ops.w2_shift.shape)}))


def load_ops(layer, rank, bits):
    import ml_dtypes

    d = cache_dir(layer, rank, bits)
    f4 = np.dtype(ml_dtypes.float4_e2m1fn)
    w1c = np.load(os.path.join(d, "w1c.npy"), mmap_mode="r")
    w2c = np.load(os.path.join(d, "w2c.npy"), mmap_mode="r")
    return {
        "w1c": w1c, "w2c": w2c,
        "w1": np.asarray(w1c).view(f4), "w2": np.asarray(w2c).view(f4),
        "w1s": np.load(os.path.join(d, "w1s.npy")), "w2s": np.load(os.path.join(d, "w2s.npy")),
        "norm": np.load(os.path.join(d, "norm.npy")),
    }


def config_sizes(tok, rank_default):
    kind, *rest = tok.split(":")
    if kind == "me":
        return np.full(G, int(rest[0]), np.int32), {"kind": "me", "M_e": int(rest[0])}
    if kind == "rt":
        bucket, layer = int(rest[0]), int(rest[1])
        st = json.load(open(STATS))["sets"][f"b{bucket}_L{layer}"]
        rank = int(rest[2]) if len(rest) > 2 else (st["critical_rank"] if rank_default is None else rank_default)
        sizes = np.asarray(st["counts"][rank], np.int32)
        return sizes, {"kind": "rt", "bucket": bucket, "layer": layer, "rank": rank,
                       "pairs": int(sizes.sum()), "hit": int((sizes > 0).sum()), "hot": int(sizes.max())}
    if kind == "sizes":
        return np.asarray(json.load(open(rest[0])), np.int32), {"kind": "sizes", "file": rest[0]}
    raise ValueError(tok)


def plan_for(info):
    """The fused kernel's build plan at the bucket a config stands for (uniform M_e rows per expert is a rank's
    48 * M_e pairs = 8 * M_e rows per rank at top-6)."""
    b = info.get("bucket")
    if b is None:
        b = 8 * info.get("M_e", 1)
    return "decode" if b <= 64 else ("mid" if b <= 512 else ("t1" if b <= 1024 else "t2"))


def make_inputs(sizes, norm, seed=0):
    """x e4m3 [M, 40, 128] (rowquant of RMS-normed normal rows x ffn_norm), scale slab [S, 128], starts."""
    import jax.numpy as jnp
    import k32z_grouped_ffn as K

    m = int(sizes.sum())
    starts = np.zeros(G, np.int32)
    starts[1:] = np.cumsum(sizes)[:-1]
    rng = np.random.default_rng([seed, m])
    h = rng.standard_normal((m, 5120), dtype=np.float32)
    h = h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1e-6) * norm[None, :]
    q, s = K.rowquant_fp8(jnp.asarray(h, jnp.bfloat16))
    s_rows = -(-m // 128) + 3
    slab = np.zeros(s_rows * 128, np.float32)
    slab[:m] = np.asarray(s, np.float32).reshape(-1)
    return q.reshape(m, 40, 128), slab.reshape(s_rows, 128), starts


def k32z_decode(codes, table, bits, groups, base_bytes):
    """Exact K32Z weight values (f32) [K, N] = e5m2(e2m1(code)) with 4*d added to the byte, and the column
    scale 2^(b - 127) [N]."""
    import ml_dtypes

    k, n = codes.shape
    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
    e5 = e2m1.astype(ml_dtypes.float8_e5m2).view(np.uint8)
    per = 8 // bits
    t = np.arange(n) // 128
    col = (t // per) * 128 + np.arange(n) % 128
    d = (table[:groups][:, col].astype(np.int32) >> (bits * (t % per))[None, :]) & ((1 << bits) - 1)  # [G, N]
    byte = e5[codes].astype(np.int32) + 4 * np.repeat(d, 32, axis=0)[:k]
    assert byte.max() <= 255
    w = byte.astype(np.uint8).view(ml_dtypes.float8_e5m2).astype(np.float32)
    colscale = np.exp2(base_bytes.astype(np.float64) - 127.0).astype(np.float32)
    return w, colscale


def reference_rows(ops, e, xq_rows, xs_rows, bits):
    """The K32Z reference for some rows of expert e: f32 HIGHEST matmuls on the exact K32Z weights, the
    kernel's own requantization functions (bf16 mid row quant, wire row quant), out = bf16(q * s)."""
    import jax
    import jax.numpy as jnp
    import k32z_grouped_ffn as K

    w2s = ops["w2s"][e]
    n1, n2 = 4608, 5120
    t1 = np.arange(n1) // 128
    t2 = np.arange(n2) // 128
    b1 = w2s[74 + t1 % 2, (t1 // 2) * 128 + np.arange(n1) % 128]
    b2 = w2s[72 + t2 % 2, (t2 // 2) * 128 + np.arange(n2) % 128]
    w1, c1 = k32z_decode(np.asarray(ops["w1c"][e]), ops["w1s"][e], bits, 160, b1)
    w2, c2 = k32z_decode(np.asarray(ops["w2c"][e]), w2s, 4, 72, b2)
    hp = jax.lax.Precision.HIGHEST
    x = jnp.asarray(xq_rows).reshape(-1, 5120).astype(jnp.float32)
    acc1 = jnp.dot(x, jnp.asarray(w1), precision=hp)
    mid_q, mid_s = K._intermediate_rows(acc1, jnp.asarray(xs_rows)[:, None], 2304, w1s=jnp.asarray(c1)[None, :])
    acc2 = jnp.dot(mid_q.astype(jnp.float32), jnp.asarray(w2), precision=hp)
    q, s = K.wire_row_and_scale(acc2, mid_s, w2s=jnp.asarray(c2)[None, :])
    return np.asarray((q.astype(jnp.float32) * s).astype(jnp.bfloat16).astype(jnp.float32))


def kernel_device_us(trace_dir, name):
    import glob

    import dsv41_k32z_trace as TR

    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    dev = TR.load(files[-1])
    out = {}
    for d, rec in dev.items():
        durs = [(e - s) / 1e3 for s, e, n, _ in rec["ops"] if name in n]
        if durs:
            out[d] = durs
            # K9: instruction-overlay waits inside the kernel op windows (only with the LLO / overlay
            # profiler flags; else none recorded): (total us, count) per kernel call.
            wins = [(s, e) for s, e, n, _ in rec["ops"] if name in n]
            ov = [(s, e) for s, e, _ in rec.get("overlay", []) if any(ws <= s < we for ws, we in wins)]
            out[("ovl", d)] = (sum(e - s for s, e in ov) / 1e3 / len(wins), len(ov) / len(wins))
    return out


def run(a):
    import jax
    import jax.numpy as jnp  # noqa: F401
    import k32z_grouped_ffn as K

    ops = load_ops(a.layer, a.rank, a.bits)
    dev = jax.devices()[0]
    w = [jax.device_put(ops[k], dev) for k in ("w1", "w2", "w1s", "w2s")]
    variants = [(m, int(s)) for m, s in (v.split(":") for v in a.variants.split(","))]
    # K9: knob variants (k32z_grouped_ffn.build knobs), ";"-separated, "base" = none; every (mode, w_split)
    # variant runs once per knob variant. --alloc-rows pads x / the scale slab so configs share shapes.
    kvars = [k.strip() for k in a.knobs.split(";")] if a.knobs else ["base"]
    results = []
    for tok in a.configs.split(","):
        sizes, info = config_sizes(tok, a.cfg_rank)
        plan = a.plan or plan_for(info)
        xq, slab, starts = make_inputs(sizes, ops["norm"])
        total = int(sizes.sum())
        if a.alloc_rows and a.alloc_rows > total:
            xq = np.concatenate([np.asarray(xq), np.zeros((a.alloc_rows - total,) + xq.shape[1:], xq.dtype)])
            s_rows = -(-a.alloc_rows // 128) + 3
            slab = np.concatenate([slab, np.zeros((s_rows - slab.shape[0], 128), slab.dtype)])
        args = [jax.device_put(v, dev) for v in (xq, slab)] + w + [
            jax.device_put(sizes, dev), jax.device_put(starts, dev)]
        first_full = None
        for kv, (mode, w_split) in [(k, v) for k in kvars for v in variants]:
            fn = jax.jit(lambda *xs, _m=mode, _s=w_split, _k=kv: K.computation(
                *xs, plan=plan, mode=_m, w_split=_s, knobs=_k))
            t0 = time.perf_counter()
            try:
                fn = fn.lower(*args).compile()
            except Exception as exc:  # noqa: BLE001  (K9: one failed variant does not end the batch)
                rec = {"config": tok, "plan": plan, "knobs": kv, "mode": mode,
                       "error": f"{type(exc).__name__}: {exc}"[:600]}
                print("VARIANT Error " + json.dumps(rec), flush=True)
                results.append(rec)
                continue
            compile_s = time.perf_counter() - t0
            try:
                code_mb = round(fn.memory_analysis().generated_code_size_in_bytes / 2**20, 4)
            except Exception as exc:  # noqa: BLE001
                code_mb = f"{type(exc).__name__}"
            try:
                out = jax.block_until_ready(fn(*args))
            except Exception as exc:  # noqa: BLE001
                rec = {"config": tok, "plan": plan, "knobs": kv, "mode": mode,
                       "error": f"run {type(exc).__name__}: {exc}"[:600]}
                print("VARIANT Error " + json.dumps(rec), flush=True)
                results.append(rec)
                continue
            rec = {"config": tok, "info": info, "plan": plan, "knobs": kv, "mode": mode, "w_split": w_split,
                   "rows": total, "compile_s": round(compile_s, 2), "code_mb": code_mb,
                   "first_call_s": round(time.perf_counter() - t0, 2)}
            if mode == "full":
                outn = np.asarray(out, np.float32).reshape(-1, 5120)[:total]
                rec["nonfinite"] = int((~np.isfinite(outn)).sum())
                if first_full is None:
                    first_full = outn
                    rec["bitwise_ref"] = f"{kv}:{mode}:{w_split}"
                else:
                    rec["bitwise_vs_first_full"] = bool(np.array_equal(outn, first_full))
                    if not rec["bitwise_vs_first_full"]:
                        d = outn - first_full
                        rec["rel_l2_vs_first"] = float(np.linalg.norm(d) / max(np.linalg.norm(first_full), 1e-30))
            if a.validate and mode == "full" and w_split == variants[0][1] and kv == kvars[0]:
                xqn = np.asarray(xq)
                xs_rows = slab.reshape(-1)
                errs = []
                rng = np.random.default_rng(1)
                num = den = 0.0
                for e in np.nonzero(sizes)[0]:
                    rows = np.arange(starts[e], starts[e] + sizes[e])
                    pick = np.sort(rng.choice(rows, min(a.validate, len(rows)), replace=False))
                    ref = reference_rows(ops, int(e), xqn[pick], xs_rows[pick], a.bits)
                    got = outn[pick]
                    num += float(np.sum((got - ref) ** 2))
                    den += float(np.sum(ref ** 2))
                    errs.append((int(e), float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)),
                                 int(np.sum(got != ref)), int(got.size)))
                rec["validate"] = {"rel_l2_all": float(np.sqrt(num / max(den, 1e-30))),
                                   "rel_l2_max_expert": max(x[1] for x in errs),
                                   "elems_differ": sum(x[2] for x in errs), "elems": sum(x[3] for x in errs),
                                   "experts": len(errs)}
            if a.time:
                for _ in range(5):
                    r = fn(*args)
                jax.block_until_ready(r)
                ts = []
                for _ in range(a.trials):
                    t0 = time.perf_counter()
                    for _ in range(20):
                        r = fn(*args)
                    jax.block_until_ready(r)
                    ts.append((time.perf_counter() - t0) / 20 * 1e6)
                rec["wall_us"] = [round(t, 1) for t in ts]
            if a.trace:
                td = os.path.join(a.trace, f"{tok}_{kv}_{mode}_{w_split}".replace(":", "_").replace("/", "_")
                                  .replace(",", "-"))
                os.makedirs(td, exist_ok=True)
                jax.block_until_ready(fn(*args))
                with jax.profiler.trace(td):
                    for _ in range(5):
                        r = fn(*args)
                    jax.block_until_ready(r)
                ku = kernel_device_us(td, "k8_k32z_grouped_ffn")
                vals = [v for d, x in ku.items() if not isinstance(d, tuple) for v in x]
                rec["kernel_us"] = round(statistics.median(vals), 1) if vals else None
                rec["kernel_us_all"] = [round(v, 1) for v in vals]
                ovl = [x for d, x in ku.items() if isinstance(d, tuple)]
                if ovl:
                    rec["ovl_us_per_call"] = round(max(o[0] for o in ovl), 2)
                    rec["ovl_n_per_call"] = round(max(o[1] for o in ovl), 1)
            print(json.dumps(rec), flush=True)
            results.append(rec)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "run"])
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--configs", default="rt:8:3")
    p.add_argument("--cfg-rank", type=int, default=None, help="EP rank of rt configs (default: critical)")
    p.add_argument("--plan", default=None)
    p.add_argument("--variants", default="full:1",
                   help="mode:w_split list (mode full | dma | comp); full:1 is the baseline")
    p.add_argument("--knobs", default=None, help="K9: ';'-separated knob variants (base = none)")
    p.add_argument("--alloc-rows", type=int, default=0, help="K9: pad x / the scale slab to this many rows")
    p.add_argument("--validate", type=int, default=0)
    p.add_argument("--time", action="store_true")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--trace", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    prep(a) if a.cmd == "prep" else run(a)


if __name__ == "__main__":
    main()
