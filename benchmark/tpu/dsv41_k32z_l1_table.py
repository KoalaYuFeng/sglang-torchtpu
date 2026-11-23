#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Markdown tables from dsv41_k32z_l1.py result JSON files.

Per (bucket, routing): each config's pipelined mean (median of the trials,
with the trial spread), the fused EP kernel's device time per call (max over
cores, from the profile, where traced), new / base, and the validation
against the exact-K32 reference (rel L2, worst row), non-finite count, pad
rows. The pair defaults to A vs AZ; ``--pair D,DZ`` compares the push pair.

    python3 dsv41_k32z_l1_table.py [--pair D,DZ] l1_layer3.json [...]
"""

import json
import statistics
import sys


def kernel_us(cfg):
    kern = cfg.get("profile", {}).get("kernels", {})
    for name, v in kern.items():
        if name.startswith(("fused_ep_moe_v2", "push_fused_moe")):
            return v["per_call_us_max_core_median"]
    return None


def table(path, base="A", new="AZ"):
    r = json.load(open(path))
    c = r["config"]
    print(
        f"\n### layer {c['layer']}, gate/up bits {c.get('gate_up_bits')}, "
        f"K32Z rungs {c.get('k32z_rungs', 'default')}, trials {c['trials']} ({path})\n"
    )
    b, n = base, new
    print(
        f"| rows/rank | routing | {b} pipe ms [trials] | {n} pipe ms [trials] | "
        f"{n}/{b} | {b} kernel us | {n} kernel us | {n}/{b} kernel | "
        f"{b} vs ref (worst) | {n} vs ref (worst) | nonfinite {b}/{n} | pad rows 0 |"
    )
    print("|" + "---|" * 12)
    for p in r["points"]:
        a, z = p["configs"].get(base, {}), p["configs"].get(new, {})
        if "error" in p or not a.get("trials") or not z.get("trials"):
            print(
                f"| {p['bucket']} | {p['routing']} | error: {p.get('error', '')[:80]} |"
            )
            continue
        pa = [t["pipelined_mean"] for t in a["trials"]]
        pz = [t["pipelined_mean"] for t in z["trials"]]
        ma, mz = statistics.median(pa), statistics.median(pz)
        ka, kz = kernel_us(a), kernel_us(z)
        ra, rz = a.get("vs_reference", {}), z.get("vs_reference", {})
        kr = f"{kz / ka:.3f}" if ka and kz else "-"
        print(
            f"| {p['bucket']} | {p['routing']} | {ma:.4f} {[round(v, 4) for v in pa]} | "
            f"{mz:.4f} {[round(v, 4) for v in pz]} | {mz / ma:.3f} | "
            f"{ka if ka else '-'} | {kz if kz else '-'} | {kr} | "
            f"{ra.get('rel_l2', float('nan')):.4f} ({ra.get('worst_row_rel', float('nan')):.4f}) | "
            f"{rz.get('rel_l2', float('nan')):.4f} ({rz.get('worst_row_rel', float('nan')):.4f}) | "
            f"{a.get('nonfinite')}/{z.get('nonfinite')} | "
            f"{a.get('pad_rows_exact_zero')}/{z.get('pad_rows_exact_zero')} |"
        )
    w = r.get("weights", {})
    print(
        f"\nweights: total {w.get('total_s')} s, layer max shift (gate/up, down) "
        f"{w.get('layer_max_shift_gate_up_down')}; K32Z convert per rank (CPU) "
        f"{[x.get('k32z_convert_s') for x in w.get('ranks', [])]} s; K512 prep per rank "
        f"{[x.get('k512_prep_s') for x in w.get('ranks', [])]} s"
    )
    comp = {}
    for p in r["points"]:
        for cfg, v in p.get("compile", {}).items():
            comp.setdefault(cfg, []).append(
                (p["bucket"], v.get("lower_s"), v.get("compile_s"))
            )
    for cfg, v in comp.items():
        print(f"compile {cfg}: {sorted(set(v))}")
    for v in r.get("vmem_probe", []):
        print(
            f"vmem {v['config']} b={v['bucket']}: fits at limit {v['fits_at_limit']}, "
            f"min fit <= {v['min_fit_bytes_upper']} B, fail at {v['max_fail_bytes']} B, "
            f"scoped at 1 MiB {v['scoped_allocation_at_1MiB']}, compiles {v['compiles']}"
        )


if __name__ == "__main__":
    argv = sys.argv[1:]
    pair = ("A", "AZ")
    if argv[:1] == ["--pair"]:
        pair = tuple(argv[1].split(","))
        argv = argv[2:]
    for path in argv:
        table(path, *pair)
