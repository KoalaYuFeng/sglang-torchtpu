# SPDX-License-Identifier: Apache-2.0
"""K8 8b audit of a MaxKernel candidate (pod only): numerics and speed against the baseline (k32z_grouped_ffn).

  python3 audit_candidate.py --cand /path/optimized.py --task-inputs /path/get_inputs.py [--sweep me:1,me:4,...]
                             [--validate 8] [--out audit.json]

Per config (the task's own get_inputs() configs, then the uniform M_e sweep built like run_grouped.py):
  - candidate vs baseline output: rel-L2, max abs diff, fraction of elements that differ (bitwise check);
  - candidate and baseline vs the K32Z f32 reference (run_grouped.reference_rows on up to N sampled rows of every
    hit expert): rel-L2 over the sampled rows (main's rule: <= 0.040 on every set, baseline 0.034-0.035);
  - device time per call: the XLA module of each jitted function (median of 5 profiled calls, TensorCore 0), and
    the wall clock of 20 calls after 5 warmups (as MaxKernel's harness).
"""
import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module_us(trace_dir, module_name):
    import dsv41_k32z_trace as TR

    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    dev = TR.load(files[-1])
    d0 = min(dev)
    return [(e - s) / 1e3 for s, e, n in dev[d0]["modules"] if n.startswith(module_name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", required=True)
    p.add_argument("--task-inputs", required=True)
    p.add_argument("--sweep", default="")
    p.add_argument("--validate", type=int, default=8)
    p.add_argument("--trace", default="/root/scratch/k8/prof/audit")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import jax
    import run_grouped as RG
    import k32z_grouped_ffn as K

    cand = load_module(a.cand, "k8_candidate")
    task = load_module(a.task_inputs, "k8_task_inputs")
    ops = RG.load_ops(3, 5, 4)
    dev = jax.devices()[0]

    def base_fn(*xs, plan):
        return K.computation(*xs, plan=plan)

    def cand_fn(*xs, plan):
        return cand.computation(*xs, plan)

    jb = jax.jit(base_fn, static_argnames=("plan",))
    jc = jax.jit(cand_fn, static_argnames=("plan",))
    configs = []
    for i, (dyn, st) in enumerate(task.get_inputs()):
        configs.append((f"task{i}", dyn, st[0]))
    w = [jax.device_put(ops[k], dev) for k in ("w1", "w2", "w1s", "w2s")]
    for tok in [t for t in a.sweep.split(",") if t]:
        sizes, info = RG.config_sizes(tok, None)
        xq, slab, starts = RG.make_inputs(sizes, ops["norm"])
        dyn = [jax.device_put(v, dev) for v in (xq, slab)] + w + [
            jax.device_put(sizes, dev), jax.device_put(starts, dev)]
        configs.append((tok, dyn, RG.plan_for(info)))
    results = []
    for name, dyn, plan in configs:
        rec = {"config": name, "plan": plan}
        ob = np.asarray(jax.block_until_ready(jb(*dyn, plan=plan)), np.float32).reshape(-1, 5120)
        t0 = time.perf_counter()
        oc = np.asarray(jax.block_until_ready(jc(*dyn, plan=plan)), np.float32).reshape(-1, 5120)
        rec["cand_first_call_s"] = round(time.perf_counter() - t0, 2)
        rec["rows"] = int(ob.shape[0])
        rec["vs_base"] = {
            "rel_l2": float(np.linalg.norm(oc - ob) / max(np.linalg.norm(ob), 1e-30)),
            "max_abs": float(np.max(np.abs(oc - ob))),
            "frac_differ": float(np.mean(oc != ob)),
            "bitwise": bool(np.array_equal(oc, ob)),
            "nonfinite": int((~np.isfinite(oc)).sum()),
        }
        if a.validate:
            sizes = np.asarray(dyn[6])
            starts = np.asarray(dyn[7])
            xq = np.asarray(dyn[0])
            xs = np.asarray(dyn[1]).reshape(-1)
            rng = np.random.default_rng(1)
            nb = nc = den = 0.0
            for e in np.nonzero(sizes)[0]:
                rows = np.arange(starts[e], starts[e] + sizes[e])
                pick = np.sort(rng.choice(rows, min(a.validate, len(rows)), replace=False))
                ref = RG.reference_rows(ops, int(e), xq[pick], xs[pick], 4)
                nb += float(np.sum((ob[pick] - ref) ** 2))
                nc += float(np.sum((oc[pick] - ref) ** 2))
                den += float(np.sum(ref ** 2))
            rec["vs_ref"] = {"base_rel_l2": float(np.sqrt(nb / den)), "cand_rel_l2": float(np.sqrt(nc / den))}
        for tag, fn in (("base", jb), ("cand", jc)):
            for _ in range(5):
                r = fn(*dyn, plan=plan)
            jax.block_until_ready(r)
            t0 = time.perf_counter()
            for _ in range(20):
                r = fn(*dyn, plan=plan)
            jax.block_until_ready(r)
            rec[f"{tag}_wall_us"] = round((time.perf_counter() - t0) / 20 * 1e6, 1)
            td = os.path.join(a.trace, f"{name}_{tag}".replace(":", "_"))
            os.makedirs(td, exist_ok=True)
            with jax.profiler.trace(td):
                for _ in range(5):
                    r = fn(*dyn, plan=plan)
                jax.block_until_ready(r)
            us = module_us(td, f"jit_{tag}_fn")
            rec[f"{tag}_device_us"] = round(statistics.median(us), 1) if us else None
        if rec.get("base_device_us") and rec.get("cand_device_us"):
            rec["speedup_device"] = round(rec["base_device_us"] / rec["cand_device_us"], 4)
        print(json.dumps(rec), flush=True)
        results.append(rec)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
