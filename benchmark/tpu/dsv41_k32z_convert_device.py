#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 N2a: the K32Z conversion of one rank's MoE layer on the TPU device.

Reads one rank's 48 routed experts of a DeepSeek-V4.1 Flash layer (the
FusedMoE parameter layout SGLang's loader leaves), moves them to the torch_tpu
device and times, per trial (torch.accelerator.synchronize at each boundary):

* convert   sglang_tpu.dsv4_mxfp4_k32z.convert_mxfp4_experts_to_k32z
* relayout  fused_ep_op.relayout_gmm_weight + materialize_float4_x2 (w13, w2)
* tables    the flat shift tables through the host (fused_ep_op's path)
* k512      (--k512) the gmm preparation on the same device tensors, for
            comparison (the K512 converter K32Z replaces)

The first trial's device outputs are checked byte for byte against the CPU
conversion of the same tensors. A chip process: run it under the TPU lock.

    PYTHONPATH=$T/python:$T python3 $T/benchmark/tpu/dsv41_k32z_convert_device.py \\
        --weights /weights/DeepSeek-V4.1-Flash --layer 3 --out conv.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
for _p in (str(TREE / "python"), str(TREE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--experts-per-rank", type=int, default=48)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--k512", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import dsv41_fused_ep_dispatch_dist as D
    import numpy as np
    import torch

    fe = D._fe()
    from sglang_tpu import dsv4_mxfp4_k32z as k32z
    from sglang_tpu.models.deepseek_v41 import fused_ep_op as op

    g = args.experts_per_rank
    experts = list(range(args.rank * g, (args.rank + 1) * g))
    ckpt = D.Checkpoint(args.weights)
    t0 = time.perf_counter()
    (w13, s13, w2, s2), dtypes = D.read_rank_experts(
        ckpt, args.layer, experts, 5120, 2304
    )
    ckpt.close()
    res = {
        "layer": args.layer,
        "rank": args.rank,
        "experts": [experts[0], experts[-1]],
        "source_dtypes": dtypes,
        "read_s": round(time.perf_counter() - t0, 2),
        "torch": torch.__version__,
    }
    t0 = time.perf_counter()
    cpu = k32z.convert_mxfp4_experts_to_k32z(w13, s13, w2, s2)
    res["cpu_convert_s"] = round(time.perf_counter() - t0, 2)
    res["max_shift"] = list(cpu.max_shift)

    import torch_tpu  # noqa: F401 - registers the "tpu" device; takes the chips

    dev = torch.device("tpu")
    sync = torch.accelerator.synchronize
    res["trials"] = []
    for trial in range(args.trials):
        rec = {}
        try:
            src = [t.to(dev) for t in (w13, s13, w2, s2)]
            sync()
            t0 = time.perf_counter()
            ops = k32z.convert_mxfp4_experts_to_k32z(*src)
            sync()
            t1 = time.perf_counter()
            w1x = op.materialize_float4_x2(op.relayout_gmm_weight(ops.w13))
            w2x = op.materialize_float4_x2(op.relayout_gmm_weight(ops.w2))
            sync()
            t2 = time.perf_counter()
            tabs = []
            for t in (ops.w13_shift, ops.w2_shift):
                e, rows, cols = t.shape
                tabs.append(t.to("cpu", copy=True).reshape(e * rows, cols).to(dev))
            sync()
            t3 = time.perf_counter()
            rec.update(
                convert_s=round(t1 - t0, 3),
                relayout_s=round(t2 - t1, 3),
                tables_s=round(t3 - t2, 3),
                total_s=round(t3 - t0, 3),
            )
            if trial == 0:
                same = {}
                for name in ("w13", "w13_shift", "w2", "w2_shift"):
                    a = getattr(ops, name).to("cpu").numpy()
                    b = getattr(cpu, name).numpy()
                    same[name] = bool(np.array_equal(a, b))
                same["max_shift"] = list(ops.max_shift) == list(cpu.max_shift)
                ref1 = op.relayout_gmm_weight(cpu.w13).numpy()
                ref2 = op.relayout_gmm_weight(cpu.w2).numpy()
                same["w1_float4_x2"] = bool(
                    np.array_equal(w1x.view(torch.uint8).to("cpu").numpy(), ref1)
                )
                same["w2_float4_x2"] = bool(
                    np.array_equal(w2x.view(torch.uint8).to("cpu").numpy(), ref2)
                )
                same["tables_flat"] = bool(
                    np.array_equal(
                        tabs[0].to("cpu").numpy(),
                        cpu.w13_shift.numpy().reshape(tabs[0].shape),
                    )
                    and np.array_equal(
                        tabs[1].to("cpu").numpy(),
                        cpu.w2_shift.numpy().reshape(tabs[1].shape),
                    )
                )
                rec["device_equals_cpu"] = same
            del ops, w1x, w2x, tabs
            if args.k512:
                src = [t.to(dev) for t in (w13, s13, w2, s2)]
                sync()
                t4 = time.perf_counter()
                prep = fe.run_gmm_prep(*src)
                sync()
                rec["k512_prep_s"] = round(time.perf_counter() - t4, 3)
                rec["k512_outputs"] = {
                    k: [str(v.dtype), list(v.shape), str(v.device)]
                    for k, v in prep.items()
                }
                del prep
            del src
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
            rec["traceback"] = traceback.format_exc()[-3000:]
        res["trials"].append(rec)
        print(f"[convert] trial {trial}: {rec}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.exit(main())
