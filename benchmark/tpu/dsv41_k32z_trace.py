# SPDX-License-Identifier: Apache-2.0
"""Per-TensorCore digest of a jax.profiler xplane.pb (Step 7 N3a breakdown).

Ported from the Step 1 breakdown's trace digest (moe_research/trace_p1.py,
11-19) so the two steps measure the same quantities:

  span      = last TensorCore op end - first TensorCore op start of one call
              of the program (its XLA module window), max over cores; the
              launch barrier before the first op is excluded;
  kernel    = the fused EP kernel op's device time, max over cores;
  categories (summed op durations per call and core): kernel, combine,
              select, copy, wait_ag_tok, wait_ag_meta, tc_ag_*, wait_other,
              other.

A trace session may hold several programs; each is told apart by its XLA
module name (``jit_<function name>``), so every program needs its own
executable. On top of the Step 1 digest, ``ops`` lists every TensorCore op of
a call in device order (per-call duration, max over cores, median over calls;
start relative to the call's first op on the core where the span is
longest) and ``busy`` / ``gap`` split that core's span into op time and idle
time between ops.
"""

from __future__ import annotations

import bisect
import re
import statistics
from collections import defaultdict

KERNEL_RE = re.compile(r"^(fused_ep_moe_v2|push_fused_moe)")
SHORT_RE = re.compile(r"^%?([\w.\-]+)")
PLANE_RE = re.compile(r"^/device:TPU:(\d+)(?: SparseCore (\d+))?$")
MODULE = "__module__"


def _short(name):
    m = SHORT_RE.match(name)
    return m.group(1) if m else name[:80]


def _ag_kind(long):
    if "f8e4m3fn[" in long:
        return "tok"
    if "s32[" in long:
        return "meta"
    return "other"


def tc_category(short, long):
    if KERNEL_RE.match(short):
        return "kernel"
    if short.startswith("copy"):
        return "copy"
    if short.startswith("moe_v2_combine"):
        return "combine"
    if short.startswith("_select_kernel"):
        return "select"
    done = "done" in short
    if "all-gather" in short:
        kind = _ag_kind(long)
        if done:
            return f"wait_ag_{kind}"
        if "start" in short or "update" in short:
            return "other"
        return f"tc_ag_{kind}"
    if done:
        return "wait_other"
    return "other"


def load(path):
    """{device id: {"modules": [(s, e, name)], "ops": [(s, e, short, cat)]}} (ns)."""
    from jax._src.lib import _profile_data as pd

    data = pd.ProfileData.from_file(str(path))
    dev = defaultdict(lambda: {"modules": [], "ops": [], "overlay": []})
    for plane in data.planes:
        m = PLANE_RE.match(plane.name)
        if not m or m.group(2) is not None:
            continue
        rec = dev[int(m.group(1))]
        for line in plane.lines:
            if line.name == "XLA Modules":
                for ev in line.events:
                    rec["modules"].append(
                        (ev.start_ns, ev.start_ns + ev.duration_ns, _short(ev.name))
                    )
            elif line.name in ("TC Overlay", "XLA TraceMe"):
                # Only with --xla_tpu_enable_llo_profiling and the overlay
                # profiler flags: instruction overlay waits (libtpu 0.0.50
                # puts them on the XLA TraceMe line as "overlay-wait").
                for ev in line.events:
                    if line.name == "XLA TraceMe" and not ev.name.startswith(
                        "overlay-wait"
                    ):
                        continue
                    rec["overlay"].append(
                        (ev.start_ns, ev.start_ns + ev.duration_ns, ev.name[:80])
                    )
            elif line.name == "XLA Ops":
                for ev in line.events:
                    s = _short(ev.name)
                    rec["ops"].append(
                        (
                            ev.start_ns,
                            ev.start_ns + ev.duration_ns,
                            s,
                            tc_category(s, ev.name),
                        )
                    )
    for rec in dev.values():
        rec["modules"].sort()
        rec["ops"].sort()
        rec["overlay"].sort()
    return dict(dev)


def q(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return {"median": v[0], "p10": v[0], "p90": v[0], "n": 1}
    qs = statistics.quantiles(v, n=10, method="inclusive")
    return {"median": statistics.median(v), "p10": qs[0], "p90": qs[-1], "n": len(v)}


def programs_in(dev):
    names = defaultdict(int)
    for rec in dev.values():
        for _, _, n in rec["modules"]:
            names[n] += 1
    return dict(names)


def _union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def module_windows(dev, program):
    """{core: [(start, end)]} of the module windows of `program` (ns)."""
    return {
        d: [(ms, me) for ms, me, n in rec["modules"] if n.split(".")[0] == program]
        for d, rec in sorted(dev.items())
    }


def calls_of(dev, program):
    """Per call: {core: [ops]} for module `program`; module windows in MODULE."""
    per_core = {}
    for d, wins in module_windows(dev, program).items():
        ops = dev[d]["ops"]
        starts = [o[0] for o in ops]
        calls = []
        for ms, me in wins:
            lo = bisect.bisect_left(starts, ms - 1)
            hi = bisect.bisect_right(starts, me + 1)
            if hi > lo:
                calls.append(ops[lo:hi] + [(ms, me, MODULE, MODULE)])
        per_core[d] = calls
    if not per_core or not all(per_core.values()):
        return []
    n = min(len(v) for v in per_core.values())
    return [{d: per_core[d][i] for d in per_core} for i in range(n)]


def digest(dev, program):
    """Span, kernel, categories and the op timeline of every call of `program`."""
    calls = calls_of(dev, program)
    if not calls:
        return None
    rows = []
    for full_row in calls:
        c = {"span": {}, "cats": defaultdict(dict), "kernel": {}, "module": {}}
        row = {}
        for d, inside in full_row.items():
            ((ms, me, _, _),) = [o for o in inside if o[2] == MODULE]
            inside = row[d] = [o for o in inside if o[2] != MODULE]
            first = min(o[0] for o in inside)
            last = max(o[1] for o in inside)
            c["span"][d] = (last - first) / 1e3
            # Module window and the launch barrier before the first op.
            c["module"][d] = ((me - ms) / 1e3, (first - ms) / 1e3)
            for s, e, _, cat in inside:
                c["cats"][cat][d] = c["cats"][cat].get(d, 0.0) + (e - s) / 1e3
            k = [o for o in inside if o[3] == "kernel"]
            if k:
                c["kernel"][d] = sum(o[1] - o[0] for o in k) / 1e3
        crit = max(c["span"], key=c["span"].get)
        inside = row[crit]
        first = min(o[0] for o in inside)
        busy = _union([(o[0], o[1]) for o in inside]) / 1e3
        c["span_max"] = c["span"][crit]
        c["busy_crit"] = busy
        c["gap_crit"] = c["span"][crit] - busy
        # Critical core: first op -> kernel start, kernel end -> last op.
        kops = [o for o in inside if o[3] == "kernel"]
        if kops:
            last = max(o[1] for o in inside)
            c["pre_kernel_crit"] = (min(o[0] for o in kops) - first) / 1e3
            c["post_kernel_crit"] = (last - max(o[1] for o in kops)) / 1e3
        c["kernel_max"] = max(c["kernel"].values()) if c["kernel"] else None
        c["module_max"] = max(v[0] for v in c["module"].values())
        c["barrier_crit"] = c["module"][crit][1]
        c["cat_max"] = {cat: max(v.values()) for cat, v in c["cats"].items()}
        # Op timeline: per op name, duration max over cores; start on the
        # critical core relative to its first op.
        seq = defaultdict(lambda: {"dur": {}, "start": None, "n": 0, "cat": ""})
        for d, ops in row.items():
            for s, e, name, cat in ops:
                r = seq[name]
                r["dur"][d] = r["dur"].get(d, 0.0) + (e - s) / 1e3
                r["cat"] = cat
                if d == crit:
                    r["n"] += 1
                    st = (s - first) / 1e3
                    r["start"] = st if r["start"] is None else min(r["start"], st)
        c["ops"] = {
            n: (max(r["dur"].values()), r["start"], r["cat"], r["n"])
            for n, r in seq.items()
        }
        rows.append(c)
    r3 = lambda v: None if v is None else round(v, 3)  # noqa: E731
    cats = sorted({k for c in rows for k in c["cat_max"]})
    names = sorted({n for c in rows for n in c["ops"]})
    timeline = []
    for n in names:
        durs = [c["ops"][n][0] for c in rows if n in c["ops"]]
        starts = [
            c["ops"][n][1] for c in rows if n in c["ops"] and c["ops"][n][1] is not None
        ]
        cat = next(c["ops"][n][2] for c in rows if n in c["ops"])
        cnt = max(c["ops"][n][3] for c in rows if n in c["ops"])
        timeline.append(
            [
                n,
                cat,
                cnt,
                r3(statistics.median(durs)),
                r3(statistics.median(starts)) if starts else None,
                len(durs),
            ]
        )
    timeline.sort(key=lambda t: (t[4] is None, t[4] if t[4] is not None else 0))
    return {
        "program": program,
        "calls": len(rows),
        "cores": len(calls[0]),
        "span_max_us": [r3(c["span_max"]) for c in rows],
        "kernel_max_us": [r3(c["kernel_max"]) for c in rows],
        # Step 8 (K8): the fused kernel's time on every core, per call, in
        # device order: which rank is critical under skewed routing.
        "kernel_by_core_us": [
            [r3(c["kernel"].get(d)) for d in sorted(c["span"])] for c in rows
        ],
        "busy_crit_us": [r3(c["busy_crit"]) for c in rows],
        "gap_crit_us": [r3(c["gap_crit"]) for c in rows],
        "pre_kernel_crit_us": [r3(c.get("pre_kernel_crit")) for c in rows],
        "post_kernel_crit_us": [r3(c.get("post_kernel_crit")) for c in rows],
        "module_max_us": [r3(c["module_max"]) for c in rows],
        "barrier_crit_us": [r3(c["barrier_crit"]) for c in rows],
        "cat_max_us": {cat: [r3(c["cat_max"].get(cat)) for c in rows] for cat in cats},
        # [name, category, count on the critical core, dur max-core median
        #  over calls, start (critical core, median), calls seen]
        "ops": timeline,
    }


def pipelined(dev, program, max_idle_us=100.0):
    """Back-to-back calls of `program`: per core, consecutive module windows
    whose idle gap (next start - previous end) is below max_idle_us; the
    period (start to start), idle gap, and of the later window its launch
    barrier (start -> first op), span (first -> last op) and tail (last op ->
    end); medians over pairs, max over cores. For period, idle gap and span
    also the median of two-call means (``*2_us_max_core``) and the mean
    (``*_mean_us_max_core``) over pairs, max over cores: K32Z programs that
    pay instruction-overlay waits alternate a slow and a fast call back to
    back, so a plain median picks one of the two modes (the ledger's N3b
    rule, 02bea73d1f)."""
    per_core = {}
    for d, wins in module_windows(dev, program).items():
        ops = dev[d]["ops"]
        starts = [o[0] for o in ops]
        pairs = []
        for a, b in zip(wins, wins[1:]):
            if (b[0] - a[1]) / 1e3 >= max_idle_us:
                continue
            lo = bisect.bisect_left(starts, b[0] - 1)
            hi = bisect.bisect_right(starts, b[1] + 1)
            if hi <= lo:
                continue
            first = min(o[0] for o in ops[lo:hi])
            last = max(o[1] for o in ops[lo:hi])
            pairs.append(
                (
                    (b[0] - a[0]) / 1e3,
                    (b[0] - a[1]) / 1e3,
                    (first - b[0]) / 1e3,
                    (last - first) / 1e3,
                    (b[1] - last) / 1e3,
                )
            )
        if pairs:
            meds = [statistics.median(p[i] for p in pairs) for i in range(5)]
            for i in (0, 1, 3):
                vals = [p[i] for p in pairs]
                two = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
                meds.append(statistics.median(two) if two else None)
                meds.append(statistics.mean(vals))
            per_core[d] = tuple(meds) + (len(pairs),)
    if not per_core:
        return None

    def mx(i):
        vals = [v[i] for v in per_core.values() if v[i] is not None]
        return round(max(vals), 3) if vals else None

    return {
        "period_us_max_core": mx(0),
        "idle_us_max_core": mx(1),
        "barrier_us_max_core": mx(2),
        "span_us_max_core": mx(3),
        "tail_us_max_core": mx(4),
        "period2_us_max_core": mx(5),
        "period_mean_us_max_core": mx(6),
        "idle2_us_max_core": mx(7),
        "idle_mean_us_max_core": mx(8),
        "span2_us_max_core": mx(9),
        "span_mean_us_max_core": mx(10),
        "pairs_min_core": min(v[11] for v in per_core.values()),
    }


def overlay(dev, program):
    """TC Overlay events per module window of `program` (needs the LLO and
    overlay profiler flags): time before the first op, inside the fused
    kernel, and elsewhere; per window, max over cores; median over windows."""
    per_window = defaultdict(list)
    for d, wins in module_windows(dev, program).items():
        ops = dev[d]["ops"]
        ovl = dev[d].get("overlay", [])
        starts = [o[0] for o in ops]
        for i, (ms, me) in enumerate(wins):
            lo = bisect.bisect_left(starts, ms - 1)
            hi = bisect.bisect_right(starts, me + 1)
            inside = ops[lo:hi]
            if not inside:
                continue
            first = min(o[0] for o in inside)
            k = [o for o in inside if o[3] == "kernel"]
            ks, ke = (min(o[0] for o in k), max(o[1] for o in k)) if k else (0, 0)
            ev = [e for e in ovl if ms <= e[0] <= me]
            pre = sum(e[1] - e[0] for e in ev if e[0] < first) / 1e3
            kev = [e for e in ev if ks <= e[0] < ke]
            ker = sum(e[1] - e[0] for e in kev) / 1e3
            tot = sum(e[1] - e[0] for e in ev) / 1e3
            # Where in the kernel the waits sit (Step 7 R2): the first one's
            # offset from the kernel's start, and how many there are.
            off = (min(e[0] for e in kev) - ks) / 1e3 if kev else 0.0
            per_window[i].append((pre, ker, tot - pre - ker, len(ev), off, len(kev)))
    if not per_window:
        return None
    rows = [
        tuple(max(v[j] for v in vals) for j in range(6)) for vals in per_window.values()
    ]
    return {
        "windows": len(rows),
        "before_first_op_us": round(statistics.median(r[0] for r in rows), 3),
        "in_kernel_us": round(statistics.median(r[1] for r in rows), 3),
        "in_kernel_mean_us": round(statistics.mean(r[1] for r in rows), 3),
        "in_kernel_first_us": round(statistics.median(r[4] for r in rows), 3),
        "in_kernel_events": statistics.median(r[5] for r in rows),
        "other_us": round(statistics.median(r[2] for r in rows), 3),
        "events": statistics.median(r[3] for r in rows),
    }
