#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fused_ep_moe_v2 dispatch modes under different token-routing distributions.

One DeepSeek-V4.1 Flash MoE layer (384 routed experts, top-6, sqrtsoftplus +
score bias, renormalize, x1.5, block-512 FP4 weights), EP over the host's local
TPU devices on a 1-D ``("ep",)`` mesh. Plain JAX, no SGLang server, no
torch_tpu. Derived from the L1 harness ``dsv41_fused_ep_layer.py``: weights
load and relayout exactly as there, once per process.

Configs (``--configs``, default A,B,D), all through
``fused_ep.make_fused_ep_moe(kernel_fn=...)`` with a wrapper that overrides
``dispatch`` and adds ``push_options``:

    A  dispatch="all_gather"                      (serving default)
    B  dispatch="push", push_options={}           (kernel defaults)
    C  dispatch="push", {"tc_exchange": True}
    D  dispatch="push", the combined bundle of
       test/registered/tpu/kernels/moe/test_fused_ep_moe_push.py
       (dispatch_ahead = local physical experts)

Routings (``--routings``): f32 gating logits [T, E] for the same hidden states.

    real      x @ gate_w.T with the checkpoint's gate bias (the L1 path)
    uniform   6 distinct experts, uniform over all experts
    local     6 distinct experts on the token's own rank (no remote pairs)
    dup1      6 distinct experts all on rank (r + 1) % ep
    hot       Zipf(s) popularity over a seeded expert permutation, 6 distinct
              per token, sampled without replacement
    half_pad  uniform, but the last half of every rank's rows are padding

Crafted routings give the designed experts logit 8 + U[0, jitter) and every
other expert -8 + U[0, jitter), and pass a ZERO score bias (same code path),
so the selection is exact; the host re-runs the selection (sqrtsoftplus +
bias, top-6) and counts rows whose set differs from the design (must be 0).
All routings except half_pad keep the L1 harness's last ``--pad-rows`` rows
of every rank as padding (all -inf); half_pad pads bucket // 2 rows.

Loop: for bucket -> compile every config -> for routing -> build inputs once
-> for config: warmups, median of ``--iters`` synchronous calls, mean over
``--pipelined-iters`` back-to-back calls, optional jax.profiler trace of
``--profile-calls`` calls into ``<profile-dir>/<cfg>_<bucket>_<routing>``.
Numerics per (bucket, routing): every config against config A (rel L2 over
real rows, max abs), pad rows exactly 0, no non-finite values; the FP32
reference only for ``--reference-buckets`` (or every bucket with
``--reference``). A failure in one point is recorded and the run continues;
the JSON is rewritten after every routing.

Run on the pod (tree = this checkout):

    PYTHONPATH=$TREE/python:$TREE python $TREE/benchmark/tpu/dsv41_fused_ep_dispatch_dist.py \\
        --weights /weights/DeepSeek-V4.1-Flash --layer 3 --profile-dir /tmp/prof --out dist.json

``--cpu-smoke`` runs everything on 8 virtual CPU devices with a reduced
geometry (E=64, so 8 local experts >= top-6) and the FP32 reference standing
in for every config (plumbing check only; the push options still go through
the package's dispatch check).
"""

from __future__ import annotations

import os

# Never let `import torch` auto-load torch_tpu (it takes the libtpu lock).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import datetime
import hashlib
import importlib.util
import json
import logging
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parents[2]

REL_L2_BOUND = 0.06  # test_fused_ep_moe_v2.py FP8_RELATIVE_BOUND (vs FP32 reference)
ROW_BOUND = 0.075  # test_fused_ep_moe_v2.py FP8_TOKEN_BOUND (vs FP32 reference)
# Cross-dispatch sanity band (vs config A over real rows). Not a kernel spec:
# the two dispatches use different router lowerings (#2102 README), so the
# outputs are compared numerically; upstream's synthetic test sees < 1e-4.
VS_A_REL_L2_BAND = 1e-2

ROUTINGS = ("real", "uniform", "local", "dup1", "hot", "half_pad")
CRAFTED_HIGH = 8.0
CRAFTED_LOW = -8.0


def _fe():
    """fused_ep (serving) and its reference helpers as one namespace.

    The reference module lives in sglang_tpu/models/deepseek_v41/ at the
    commit this was written for; a later revision moves it next to the unit
    tests, so fall back to that file."""
    from sglang_tpu.models.deepseek_v41 import fused_ep

    try:
        from sglang_tpu.models.deepseek_v41 import fused_ep_reference
    except ImportError:
        name = "dsv41_fused_ep_reference"
        if name in sys.modules:
            fused_ep_reference = sys.modules[name]
        else:
            path = TREE / "test/registered/unit/hardware_backend/tpu" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            fused_ep_reference = importlib.util.module_from_spec(spec)
            sys.modules[name] = fused_ep_reference
            spec.loader.exec_module(fused_ep_reference)
    return types.SimpleNamespace(**{**vars(fused_ep_reference), **vars(fused_ep)})


def _first_attr(ns, *names):
    for n in names:
        if hasattr(ns, n):
            return getattr(ns, n)
    raise AttributeError(f"none of {names} found")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{utc_now()}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--weights", help="checkpoint dir with model.safetensors.index.json"
    )
    src.add_argument("--synthetic", action="store_true", help="random experts")
    p.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="8 virtual CPU devices, reduced geometry (E=64, H=1024, I=768), "
        "reference as every config: plumbing check only",
    )
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--ep", type=int, default=None, help="default: all local devices")
    p.add_argument("--configs", default="A,B,D", help="subset of A,B,C,D")
    p.add_argument("--routings", default=",".join(ROUTINGS))
    p.add_argument(
        "--buckets",
        default=None,
        help="rows per rank (default 8,64,288,2048; cpu-smoke 8,16)",
    )
    p.add_argument(
        "--pad-rows",
        type=int,
        default=2,
        help="-inf gating rows at the end of every rank (all routings but half_pad)",
    )
    p.add_argument("--warmup", type=int, default=None, help="default 3 (smoke 1)")
    p.add_argument(
        "--iters",
        type=int,
        default=None,
        help="synchronous calls (default 20, smoke 2)",
    )
    p.add_argument(
        "--pipelined-iters",
        type=int,
        default=None,
        help="back-to-back calls (default 50, smoke 2)",
    )
    p.add_argument(
        "--profile-calls",
        type=int,
        default=None,
        help="calls traced (default 5, smoke 2)",
    )
    p.add_argument("--profile-dir", default=None, help="jax.profiler trace per point")
    p.add_argument(
        "--profile-routings",
        default=None,
        help="restrict tracing to these routings (default: all)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.5, help="crafted logit jitter")
    p.add_argument("--zipf-s", type=float, default=1.2, help="hot routing exponent")
    p.add_argument("--scale-layout", choices=("flat", "grouped"), default="flat")
    p.add_argument(
        "--reference",
        action="store_true",
        help="FP32 reference comparison on every bucket (slow at large buckets)",
    )
    p.add_argument(
        "--reference-buckets",
        default="",
        help="FP32 reference comparison on these buckets only (e.g. 8,64)",
    )
    p.add_argument("--out", default=None, help="result JSON path")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Configs.
# --------------------------------------------------------------------------- #
def push_combined_options(g_local):
    """test_fused_ep_moe_push.py's combined options ("the combined options the
    model can opt into"), copied verbatim; dispatch_ahead there is
    physical_count // EP, i.e. the local physical experts per rank."""
    return dict(
        sc_prepush=2,
        tc_exchange=True,
        onehot_slots=True,
        tail_barrier=False,
        hbm_staging=True,
        dispatch_ahead=g_local,
        matmul_scale_repeat=True,
        exchange_row_gather=True,
        bf16_sort_counts=True,
        slim_prepush_barrier=True,
        fused_dispatch_meta=True,
        table_kernel=True,
        exchange_tables=True,
        kernel_plan_tables=True,
        light_prepush=2,
        single_barrier=True,
        early_scale_scatter=2,
    )


def resolve_configs(names, g_local, push_option_names):
    table = {
        "A": ("all_gather", None, "all_gather (serving default)"),
        "B": ("push", {}, "push, kernel defaults"),
        "C": ("push", {"tc_exchange": True}, "push + tc_exchange"),
        "D": (
            "push",
            push_combined_options(g_local),
            "push, combined bundle of test_fused_ep_moe_push.py",
        ),
    }
    out = {}
    for name in names:
        if name not in table:
            raise ValueError(f"unknown config {name!r}; known {sorted(table)}")
        dispatch, options, label = table[name]
        if options:
            unknown = sorted(set(options) - set(push_option_names))
            if unknown:
                raise ValueError(f"config {name}: unknown push options {unknown}")
        out[name] = {"dispatch": dispatch, "push_options": options, "label": label}
    return out


def make_kernel_fn(base_fn, dispatch, push_options, check_dispatch=None):
    """fused_ep_moe_v2-compatible callable that forces this config's dispatch."""

    def kernel_fn(*args, **kw):
        kw["dispatch"] = dispatch
        if dispatch == "push":
            kw["push_options"] = dict(push_options or {})
        else:
            kw.pop("push_options", None)
        if check_dispatch is not None:  # CPU stand-in: validate the plumbing
            check_dispatch(
                dispatch,
                push_options=kw.get("push_options"),
                shared_expert=None,
                shared_limit=None,
                stage="fused",
                routed=None,
                gather_positions=None,
            )
        return base_fn(*args, **kw)

    return kernel_fn


# --------------------------------------------------------------------------- #
# Checkpoint access: raw safetensors reads, so no dtype support is assumed.
# --------------------------------------------------------------------------- #
class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._files = {}

    def close(self):
        for fd, _, _ in self._files.values():
            os.close(fd)
        self._files.clear()

    def _file(self, fname):
        if fname not in self._files:
            fd = os.open(self.root / fname, os.O_RDONLY)
            n = int.from_bytes(os.pread(fd, 8, 0), "little")
            header = json.loads(os.pread(fd, n, 8))
            self._files[fname] = (fd, 8 + n, header)
        return self._files[fname]

    def has(self, name):
        return name in self.weight_map

    def raw(self, name):
        """(dtype string, header shape, uint8 bytes)."""
        if name not in self.weight_map:
            near = sorted(k for k in self.weight_map if k.rsplit(".", 2)[0] in name)[:8]
            raise KeyError(f"{name} not in the index; nearby: {near}")
        fd, base, header = self._file(self.weight_map[name])
        info = header[name]
        start, end = info["data_offsets"]
        buf = os.pread(fd, end - start, base + start)
        if len(buf) != end - start:
            raise OSError(f"short read for {name}")
        return info["dtype"], info["shape"], np.frombuffer(buf, np.uint8)

    def f32(self, name, shape):
        dtype, _, raw = self.raw(name)
        if dtype == "F32":
            out = raw.view(np.float32)
        elif dtype == "BF16":
            out = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            out = raw.view(np.float16).astype(np.float32)
        else:
            raise TypeError(f"{name}: unexpected dtype {dtype}")
        return out.reshape(shape).copy()

    def fp4_bytes(self, name, out_dim, in_dim):
        """Packed E2M1 weight [out, in/2] bytes (whatever the header dtype)."""
        dtype, shape, raw = self.raw(name)
        if raw.size != out_dim * in_dim // 2:
            raise ValueError(f"{name}: {dtype}{shape} is not [{out_dim}, {in_dim}] fp4")
        return raw.reshape(out_dim, in_dim // 2), dtype

    def scale_f32(self, name, out_dim, in_dim):
        """Block-32 scale [out, in/32] as f32, the way SGLang's loader fills
        its f32 scale parameter from an E8M0 tensor."""
        fe = _fe()

        dtype, _, raw = self.raw(name)
        shape = (out_dim, in_dim // 32)
        if dtype in ("F8_E8M0", "U8"):
            return fe.e8m0_to_float32(raw.reshape(shape)), dtype
        if dtype == "F32":
            import torch

            return torch.from_numpy(raw.view(np.float32).reshape(shape).copy()), dtype
        raise TypeError(f"{name}: unexpected scale dtype {dtype}")


def expert_name(layer, e, proj, kind):
    return f"layers.{layer}.ffn.experts.{e}.{proj}.{kind}"


def read_rank_experts(ckpt, layer, experts, hidden, inter):
    """Checkpoint experts -> SGLang FusedMoE parameter layout (torch, CPU)."""
    import torch

    g = len(experts)
    w13 = torch.empty((g, 2 * inter, hidden // 2), dtype=torch.uint8)
    s13 = torch.empty((g, 2 * inter, hidden // 32), dtype=torch.float32)
    w2 = torch.empty((g, hidden, inter // 2), dtype=torch.uint8)
    s2 = torch.empty((g, hidden, inter // 32), dtype=torch.float32)
    dtypes = set()
    for i, e in enumerate(experts):
        # w1 = gate_proj -> rows [0, I); w3 = up_proj -> rows [I, 2I)
        # (fused_moe_triton/layer.py:760-765).
        for half, proj in ((0, "w1"), (1, "w3")):
            rows = slice(half * inter, (half + 1) * inter)
            wb, wdt = ckpt.fp4_bytes(
                expert_name(layer, e, proj, "weight"), inter, hidden
            )
            sc, sdt = ckpt.scale_f32(
                expert_name(layer, e, proj, "scale"), inter, hidden
            )
            w13.numpy()[i, rows] = wb  # wb is a read-only buffer view
            s13[i, rows] = sc
            dtypes.update((wdt, sdt))
        wb, wdt = ckpt.fp4_bytes(expert_name(layer, e, "w2", "weight"), hidden, inter)
        sc, sdt = ckpt.scale_f32(expert_name(layer, e, "w2", "scale"), hidden, inter)
        w2.numpy()[i] = wb
        s2[i] = sc
        dtypes.update((wdt, sdt))
    return (w13, s13, w2, s2), sorted(dtypes)


def synthetic_rank_experts(rng, g, hidden, inter):
    """Random block-32 MXFP4 experts; per-32-block exponents 2^-10..2^-6."""
    import torch

    fe = _fe()

    w13 = torch.from_numpy(
        rng.integers(0, 256, (g, 2 * inter, hidden // 2), dtype=np.uint8)
    )
    w2 = torch.from_numpy(rng.integers(0, 256, (g, hidden, inter // 2), dtype=np.uint8))
    s13 = fe.e8m0_to_float32(
        rng.integers(117, 122, (g, 2 * inter, hidden // 32), dtype=np.uint8)
    )
    s2 = fe.e8m0_to_float32(
        rng.integers(117, 122, (g, hidden, inter // 32), dtype=np.uint8)
    )
    return (w13.clone(), s13, w2.clone(), s2), ["synthetic"]


def expert_vs_checkpoint(
    ckpt_expert, w1_k, s1_k, w2_k, s2_k, *, inter, padded_inter, rng
):
    """One expert, official math on the checkpoint's block-32 values vs the same
    math on the kernel-layout weights (as in dsv41_fused_ep_layer.py): pins the
    w1/w3 order from checkpoint names to kernel columns."""
    fe = _fe()

    w13_b, s13, w2_b, s2 = (np.asarray(t) for t in ckpt_expert)
    lut = np.asarray(fe.FP4_E2M1_VALUES, np.float32)

    def decode(b, s):  # [out, in/2] bytes, [out, in/32] f32 -> [out, in]
        v = np.stack((lut[b & 0x0F], lut[b >> 4]), axis=-1).reshape(b.shape[0], -1)
        return (v.reshape(v.shape[0], -1, 32) * s[:, :, None]).reshape(v.shape)

    w13 = decode(w13_b, s13)  # [2I, H]
    w2 = decode(w2_b, s2)  # [H, I]
    hidden = w2.shape[0]
    x = rng.standard_normal((64, hidden), dtype=np.float32)
    x /= np.sqrt(np.mean(x * x, axis=-1, keepdims=True))

    def ffn(gate, up, down):
        gate = np.minimum(gate, fe.SWIGLU_LIMIT)
        up = np.clip(up, -fe.SWIGLU_LIMIT, fe.SWIGLU_LIMIT)
        return (gate / (1 + np.exp(-gate)) * up) @ down

    want = ffn(x @ w13[:inter].T, x @ w13[inter:].T, w2.T)
    k1 = fe.dequantize_kernel_layout(w1_k[None], s1_k[None])[0]
    k2 = fe.dequantize_kernel_layout(w2_k[None], s2_k[None])[0]
    h = x @ k1
    gate, up = h[:, :inter], h[:, padded_inter : padded_inter + inter]
    got = ffn(gate, up, k2[:inter])
    swapped = ffn(up, gate, k2[:inter])
    norm = np.linalg.norm(want)
    rel = float(np.linalg.norm(got - want) / norm)
    rel_swapped = float(np.linalg.norm(swapped - want) / norm)
    return {
        "rel_l2": rel,
        "rel_l2_gate_up_swapped": rel_swapped,
        "order_ok": bool(rel < 0.5 * rel_swapped),
    }


# --------------------------------------------------------------------------- #
# Device placement.
# --------------------------------------------------------------------------- #
def put_float4(arr, device):
    """Host ml_dtypes float4 -> device float4. Falls back to shipping the
    uint8 codes and decoding on device (exact) if the direct put is refused."""
    import jax
    import jax.numpy as jnp

    fe = _fe()

    try:
        return jax.device_put(arr, device), "direct"
    except Exception as exc:  # noqa: BLE001 - record and fall back
        codes = jax.device_put(arr.view(np.uint8), device)
        lut = jnp.asarray(fe.FP4_E2M1_VALUES, jnp.float32)
        out = jax.jit(lambda c: lut[c.astype(jnp.int32)].astype(jnp.float4_e2m1fn))(
            codes
        )
        return out, f"lut-fallback ({type(exc).__name__})"


def verify_shard(dev_arr, host_arr, sample):
    """Read sampled experts back as f32 and compare bitwise with the host."""
    import jax
    import jax.numpy as jnp

    idx = np.asarray(sample)
    got = np.asarray(jax.jit(lambda a: a[idx].astype(jnp.float32))(dev_arr))
    want = host_arr[idx].astype(np.float32)
    return bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))


def assemble(mesh, per_rank, global_shape):
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sharding = NamedSharding(mesh, P(mesh.axis_names[0]))
    by_device = {a.devices().pop(): a for a in per_rank}
    arrays = [
        by_device[d] for d in sharding.addressable_devices_indices_map(global_shape)
    ]
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


def block_starts_by_mesh_index(mesh, arr):
    """Axis-0 start of the shard that each mesh position holds."""
    pos = {d: i for i, d in enumerate(mesh.devices.flat)}
    starts = [None] * len(pos)
    for s in arr.addressable_shards:
        starts[pos[s.device]] = s.index[0].start or 0
    return starts


# --------------------------------------------------------------------------- #
# Inputs, routings and comparisons.
# --------------------------------------------------------------------------- #
def make_hidden(rng, *, ep, bucket, hidden, norm_w):
    """The L1 harness's hidden states (same RNG stream as its make_inputs)."""
    import ml_dtypes

    tokens = ep * bucket
    z = rng.standard_normal((tokens, hidden), dtype=np.float32)
    z = z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + 1e-6)
    return (z * norm_w[None, :]).astype(ml_dtypes.bfloat16)


def pad_rows_index(*, ep, bucket, pad):
    return np.concatenate(
        [np.arange(r * bucket + bucket - pad, (r + 1) * bucket) for r in range(ep)]
    ).astype(np.int64)


def random_subsets(rng, n, pool, k):
    """n independent uniformly random k-subsets of range(pool), [n, k]."""
    if pool < k:
        raise ValueError(f"cannot draw {k} distinct experts from {pool}")
    return np.argpartition(rng.random((n, pool)), k - 1, axis=1)[:, :k]


def designed_experts(routing, rng, rows, *, ep, bucket, num_experts, topk, zipf_s):
    """Designed expert sets [len(rows), topk] for the given global rows."""
    g = num_experts // ep
    n = len(rows)
    rank = rows // bucket
    if routing in ("uniform", "half_pad"):
        return random_subsets(rng, n, num_experts, topk), None
    if routing == "local":
        return random_subsets(rng, n, g, topk) + (rank * g)[:, None], None
    if routing == "dup1":
        dst = (rank + 1) % ep
        return random_subsets(rng, n, g, topk) + (dst * g)[:, None], None
    if routing == "hot":
        perm = rng.permutation(num_experts)
        weight = np.empty(num_experts, np.float64)
        weight[perm] = np.arange(1, num_experts + 1, dtype=np.float64) ** -zipf_s
        weight /= weight.sum()
        # Weighted sampling without replacement (Efraimidis-Spirakis with
        # exponential keys): the k smallest Exp(1) / w.
        keys = rng.exponential(size=(n, num_experts)) / weight[None, :]
        ids = np.argpartition(keys, topk - 1, axis=1)[:, :topk]
        return ids, {"zipf_s": zipf_s, "hottest_experts": perm[:10].tolist()}
    raise ValueError(f"unknown crafted routing {routing!r}")


def host_select(gating, bias, topk):
    """Kernel-equivalent selection in f64: sqrtsoftplus(logits) + bias, top-k.
    Returns (ids [n, k] unordered, gap between the k-th and (k+1)-th score)."""
    g = gating.astype(np.float64)
    scores = np.sqrt(np.logaddexp(0.0, g)) + bias[None, :].astype(np.float64)
    part = np.argpartition(-scores, topk, axis=1)
    ids = part[:, :topk]
    chosen = np.take_along_axis(scores, ids, axis=1).min(axis=1)
    rest = np.take_along_axis(scores, part[:, topk:], axis=1).max(axis=1)
    return ids, chosen - rest


REPLAY_ENV = "DSV41_REPLAY_GATING"


def is_replay(routing):
    return routing.startswith("replay")


def replay_routing(routing, *, bias, ep, bucket, num_experts):
    """Routing ``replay<i>``: set i of ``$DSV41_REPLAY_GATING`` ("{bucket}" in
    the path is replaced by the bucket), f32 [sets, ep * bucket, E] in rank
    order. Each set replays the expert ids a serving run selected (routing
    dumps of models/deepseek_v41/routing_dump.py, selected offline with the
    dumped layer's own score bias) as crafted logits taken with a zero score
    bias, so any layer's selection replays on the benchmark layer's weights.
    Rows with no finite logit are the padding. ``bias`` is unused."""
    del bias
    path = os.environ[REPLAY_ENV].replace("{bucket}", str(bucket))
    sets = np.load(path, mmap_mode="r")
    index = int(routing[len("replay") :] or 0)
    gating = np.array(sets[index], np.float32)
    if gating.shape != (ep * bucket, num_experts):
        raise ValueError(
            f"{path} set {index} has shape {gating.shape}, expected "
            f"{(ep * bucket, num_experts)}"
        )
    pad_index = np.nonzero(~np.isfinite(gating).any(axis=1))[0].astype(np.int64)
    pads = np.bincount(pad_index // bucket, minlength=ep)
    info = {
        "pad_rows_per_rank": int(pads.max()),
        "pad_rows_by_rank": [int(v) for v in pads],
        "replay": f"{path}[{index}]",
    }
    return gating, np.zeros(num_experts, np.float32), pad_index, None, info


def build_routing(
    routing,
    *,
    x,
    gate_w,
    bias,
    ep,
    bucket,
    num_experts,
    topk,
    pad_rows,
    seed,
    jitter,
    zipf_s,
):
    """(gating f32 [T, E], score bias [E], pad index, designed ids or None, info)."""
    tokens = ep * bucket
    if is_replay(routing):
        return replay_routing(
            routing, bias=bias, ep=ep, bucket=bucket, num_experts=num_experts
        )
    pad = bucket // 2 if routing == "half_pad" else min(pad_rows, bucket - 1)
    pad_index = pad_rows_index(ep=ep, bucket=bucket, pad=pad)
    real = np.ones(tokens, bool)
    real[pad_index] = False
    info = {"pad_rows_per_rank": int(pad)}
    if routing == "real":
        gating = x.astype(np.float32) @ gate_w.T
        gating[pad_index] = -np.inf
        return gating.astype(np.float32), bias, pad_index, None, info
    rng = np.random.default_rng([seed, 2, bucket, ROUTINGS.index(routing)])
    rows = np.nonzero(real)[0]
    ids, extra = designed_experts(
        routing,
        rng,
        rows,
        ep=ep,
        bucket=bucket,
        num_experts=num_experts,
        topk=topk,
        zipf_s=zipf_s,
    )
    if extra:
        info.update(extra)
    gating = (CRAFTED_LOW + jitter * rng.random((tokens, num_experts))).astype(
        np.float32
    )
    gating[rows[:, None], ids] = (CRAFTED_HIGH + jitter * rng.random(ids.shape)).astype(
        np.float32
    )
    gating[pad_index] = -np.inf
    return gating, np.zeros(num_experts, np.float32), pad_index, ids, info


def routing_summary(ids, rows, *, ep, bucket, num_experts):
    """Load and data-movement summary of the selected (token, expert) pairs.
    Expert e lives on EP rank e // g (contiguous blocks, identity mesh order)."""
    g = num_experts // ep
    n, k = ids.shape
    src = rows // bucket  # the token's rank
    dst = ids // g  # owning rank of every selected expert
    remote = dst != src[:, None]
    per_expert = np.bincount(ids.ravel(), minlength=num_experts)
    recv_pairs = np.bincount(dst.ravel(), minlength=ep)
    remote_recv = np.bincount(dst[remote], minlength=ep)
    remote_sent = np.bincount(
        np.broadcast_to(src[:, None], dst.shape)[remote], minlength=ep
    )
    hit = np.zeros((n, ep), bool)
    hit[np.arange(n)[:, None], dst] = True
    hit[np.arange(n), src] = False
    distinct_remote = hit.sum(axis=1)
    remote_pairs = remote.sum(axis=1)
    experts_hit = (per_expert > 0).reshape(ep, g).sum(axis=1)
    real_per_rank = np.bincount(src, minlength=ep)
    ag_rows = (ep - 1) * bucket

    def mm(a):
        a = np.asarray(a, np.float64)
        mean = float(a.mean())
        return {
            "max": int(a.max()),
            "mean": round(mean, 3),
            "max_over_mean": round(float(a.max()) / mean, 3) if mean else None,
        }

    def avg(a):
        return round(float(np.mean(a)), 4) if n else 0.0

    return {
        "real_rows": int(n),
        "real_rows_per_rank": [int(v) for v in real_per_rank],
        "pairs": int(n * k),
        "recv_pairs_per_rank": mm(recv_pairs),
        "recv_pairs_by_rank": [int(v) for v in recv_pairs],
        "max_pairs_per_expert": int(per_expert.max()),
        "experts_hit_per_rank": mm(experts_hit),
        "mean_remote_pairs_per_token": avg(remote_pairs),
        "mean_distinct_remote_ranks_per_token": avg(distinct_remote),
        "mean_duplicate_remote_pairs_per_token": avg(remote_pairs - distinct_remote),
        "all_gather_rows_received_per_rank": ag_rows,
        "push_remote_rows_received_per_rank": mm(remote_recv),
        "push_remote_rows_sent_per_rank": mm(remote_sent),
        "push_over_all_gather_rows": (
            round(float(remote_recv.mean()) / ag_rows, 4) if ag_rows else None
        ),
    }


def compare(out, ref, real_rows):
    o = np.asarray(out, np.float64)[real_rows]
    r = np.asarray(ref, np.float64)[real_rows]
    diff = o - r
    denom = np.linalg.norm(r)
    rel = float(np.linalg.norm(diff) / denom) if denom else float(np.linalg.norm(diff))
    rnorm = np.linalg.norm(r, axis=-1)
    per_row = np.linalg.norm(diff, axis=-1) / np.where(rnorm == 0, 1.0, rnorm)
    worst = int(np.argmax(per_row)) if per_row.size else 0
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    return rel, (float(per_row[worst]) if per_row.size else 0.0), worst, max_abs


def kernel_names(lowered, compiled):
    names = set()
    for get in (lambda: lowered.as_text(), lambda: compiled.as_text()):
        try:
            text = get()
        except Exception:  # noqa: BLE001 - best effort
            continue
        names.update(re.findall(r'kernel_name\s*=\s*\\?"([A-Za-z0-9_.\-]+)', text))
        names.update(
            re.findall(
                r"\b(fused_ep_moe_v2\w*|moe_v2_combine\w*|push_fused_moe\w*|"
                r"moe_sc_prepush\w*|moe_tc_routing_exchange\w*|moe_push_tables\w*)",
                text,
            )
        )
    return sorted(names)


class _BuildLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "built program" in msg:
            self.messages.append(msg)


def file_sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def environment(jax, mesh):
    def git(*a):
        try:
            return subprocess.run(
                ["git", "-C", str(TREE), *a], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    kernel_dir = TREE / "sglang_tpu/moe/pallas/fused_moe_v2"
    devices = list(mesh.devices.flat)
    try:
        backend_version = devices[0].client.platform_version
    except Exception:  # noqa: BLE001
        backend_version = None
    return {
        "python": sys.version.split()[0],
        "host": platform.node(),
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", None),
        "backend_platform_version": backend_version,
        "device_kind": devices[0].device_kind,
        "devices": [str(d) for d in devices],
        "tree": str(TREE),
        "tree_head": git("rev-parse", "HEAD"),
        "tree_dirty_files": len((git("status", "--porcelain") or "").splitlines()),
        "sha256": {
            "fused_ep.py": file_sha(
                TREE / "sglang_tpu/models/deepseek_v41/fused_ep.py"
            ),
            "kernel.py": file_sha(kernel_dir / "kernel.py"),
            "layer.py": file_sha(kernel_dir / "layer.py"),
            "host.py": file_sha(kernel_dir / "host.py"),
            "push/layer.py": file_sha(kernel_dir / "push_fused_moe/layer.py"),
            "push/kernel.py": file_sha(kernel_dir / "push_fused_moe/kernel.py"),
            "script": file_sha(__file__),
        },
        "env": {
            k: os.environ.get(k)
            for k in (
                "LIBTPU_INIT_ARGS",
                "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
            )
        },
        "torch_tpu_imported": "torch_tpu" in sys.modules,
    }


def write_json(path, obj):
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def latency_stats(lat, pipelined):
    q = statistics.quantiles(lat, n=10) if len(lat) >= 2 else [lat[0]] * 9
    return {
        "median": round(statistics.median(lat), 4),
        "p10": round(q[0], 4),
        "p90": round(q[-1], 4),
        "min": round(min(lat), 4),
        "pipelined_mean": round(pipelined, 4),
        "n_sync": len(lat),
    }


# --------------------------------------------------------------------------- #
def load_weights(
    args, fe, jax, mesh, *, num_experts, hidden, inter, padded_inter, result
):
    """Read (or draw) every rank's experts, prepare, relayout and place them
    exactly as the L1 harness does. Returns ((w1, w2, s1, s2), gate_w, bias,
    norm_w)."""
    ep = mesh.devices.size
    g_local = num_experts // ep
    rng = np.random.default_rng(args.seed)
    ckpt = Checkpoint(args.weights) if args.weights else None
    if ckpt is not None:
        layer = args.layer
        gate_w = ckpt.f32(f"layers.{layer}.ffn.gate.weight", (num_experts, hidden))
        bias = ckpt.f32(f"layers.{layer}.ffn.gate.bias", (num_experts,))
        norm_name = f"layers.{layer}.ffn_norm.weight"
        norm_w = (
            ckpt.f32(norm_name, (hidden,))
            if ckpt.has(norm_name)
            else np.ones(hidden, np.float32)
        )
        result["checkpoint"] = {
            "gate_dtype": ckpt.raw(f"layers.{layer}.ffn.gate.weight")[0],
            "bias_abs_mean": float(np.abs(bias).mean()),
            "ffn_norm_found": ckpt.has(norm_name),
            "ffn_norm_abs_mean": float(np.abs(norm_w).mean()),
        }
    else:
        gate_w = rng.normal(0, 1 / np.sqrt(hidden), (num_experts, hidden)).astype(
            np.float32
        )
        bias = rng.normal(0, 0.05, (num_experts,)).astype(np.float32)
        norm_w = np.ones(hidden, np.float32)

    # Names at the PR head; the earlier L1 tree called them run_pr1_prep /
    # pr1_to_fused_ep_weights.
    gmm_prep = _first_attr(fe, "run_gmm_prep", "run_pr1_prep")
    to_kernel = _first_attr(fe, "gmm_to_fused_ep_weights", "pr1_to_fused_ep_weights")
    shards = {"w1": [], "w2": [], "w1_scale": [], "w2_scale": []}
    host_sample = {}
    weight_log = []
    t_weights = time.perf_counter()
    for r in range(ep):
        experts = list(range(r * g_local, (r + 1) * g_local))
        t0 = time.perf_counter()
        if ckpt is not None:
            tensors, dtypes = read_rank_experts(
                ckpt, args.layer, experts, hidden, inter
            )
        else:
            tensors, dtypes = synthetic_rank_experts(
                np.random.default_rng([args.seed, r]), g_local, hidden, inter
            )
        t1 = time.perf_counter()
        # The preparation frees its inputs; keep local expert 0 for the check.
        first = [t[0].clone().numpy() for t in tensors] if r == 0 else None
        prep = gmm_prep(*tensors)
        del tensors
        t2 = time.perf_counter()
        kw = to_kernel(
            prep["w13_weight"],
            prep["w13_weight_scale"],
            prep["w2_weight"],
            prep["w2_weight_scale"],
            scale_layout=args.scale_layout,
        )
        del prep
        t3 = time.perf_counter()
        if first is not None:

            def expert0_scale(s, k):
                nb = k // 512
                return s.reshape(g_local, -1, s.shape[-1])[0, :nb]

            check = expert_vs_checkpoint(
                first,
                kw.w1[0],
                expert0_scale(kw.w1_scale, hidden),
                kw.w2[0],
                expert0_scale(kw.w2_scale, padded_inter),
                inter=inter,
                padded_inter=padded_inter,
                rng=np.random.default_rng([args.seed, 2]),
            )
            host_sample["expert0_vs_checkpoint_block32"] = check
            log(f"[weights] expert 0 vs checkpoint block-32 math: {check}")
            del first
        device = mesh.devices.flat[r]
        w1_d, how1 = put_float4(kw.w1, device)
        w2_d, how2 = put_float4(kw.w2, device)
        s1_d = jax.device_put(kw.w1_scale, device)
        s2_d = jax.device_put(kw.w2_scale, device)
        sample = [0, g_local - 1]
        ok = verify_shard(w1_d, kw.w1, sample) and verify_shard(w2_d, kw.w2, sample)
        ok = ok and np.array_equal(np.asarray(s1_d), kw.w1_scale)
        ok = ok and np.array_equal(np.asarray(s2_d), kw.w2_scale)
        t4 = time.perf_counter()
        for k, v in zip(shards, (w1_d, w2_d, s1_d, s2_d)):
            shards[k].append(v)
        del kw
        entry = {
            "rank": r,
            "experts": [experts[0], experts[-1]],
            "source_dtypes": dtypes,
            "read_s": round(t1 - t0, 3),
            "prep_s": round(t2 - t1, 3),
            "relayout_s": round(t3 - t2, 3),
            "device_put_and_verify_s": round(t4 - t3, 3),
            "put_mode": [how1, how2],
            "readback_bit_exact": bool(ok),
            "host_maxrss_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1 << 30 if sys.platform == "darwin" else 1 << 20),
                2,
            ),
        }
        weight_log.append(entry)
        log(
            f"[weights] rank {r}: read {entry['read_s']}s prep {entry['prep_s']}s "
            f"relayout {entry['relayout_s']}s put {entry['device_put_and_verify_s']}s "
            f"{how1} readback_ok={ok}"
        )
        if not ok:
            raise RuntimeError(
                f"rank {r}: device weights differ from the host relayout"
            )
    if ckpt is not None:
        ckpt.close()
    w1 = assemble(mesh, shards["w1"], (num_experts, hidden, 2 * padded_inter))
    w2 = assemble(mesh, shards["w2"], (num_experts, padded_inter, hidden))
    if args.scale_layout == "flat":
        rows1 = fe.fp4_scale_rows(hidden, padded=True)
        rows2 = fe.fp4_scale_rows(padded_inter, padded=True)
        s1 = assemble(mesh, shards["w1_scale"], (num_experts * rows1, 2 * padded_inter))
        s2 = assemble(mesh, shards["w2_scale"], (num_experts * rows2, hidden))
    else:
        s1 = assemble(
            mesh, shards["w1_scale"], (num_experts, hidden // 512, 2 * padded_inter)
        )
        s2 = assemble(
            mesh, shards["w2_scale"], (num_experts, padded_inter // 512, hidden)
        )
    del shards
    result["weights"] = {
        "total_s": round(time.perf_counter() - t_weights, 2),
        "ranks": weight_log,
        "w1": [str(w1.dtype), list(w1.shape)],
        "w2": [str(w2.dtype), list(w2.shape)],
        "w1_scale": [str(s1.dtype), list(s1.shape)],
        "w2_scale": [str(s2.dtype), list(s2.shape)],
        **host_sample,
    }
    # Expert placement as the kernel sees it: mesh position r must hold the
    # contiguous expert block [r * g, (r + 1) * g). make_fused_ep_moe passes
    # mesh_ep_ranks = tuple(range(ep)) (identity), so mesh position == EP rank
    # and _relabel_expert_ids_to_mesh_order leaves expert ids unchanged.
    starts = block_starts_by_mesh_index(mesh, w1)
    contiguous = starts == [r * g_local for r in range(ep)]
    result["placement"] = {
        "w1_block_start_by_mesh_index": starts,
        "experts_contiguous_by_mesh_index": contiguous,
    }
    log(
        f"[weights] done in {result['weights']['total_s']}s: w1 {w1.dtype}{w1.shape} "
        f"w2 {w2.dtype}{w2.shape} s1 {s1.shape} s2 {s2.shape}; expert blocks "
        f"contiguous by mesh index: {contiguous}"
    )
    if not contiguous:
        raise RuntimeError(f"unexpected expert placement {starts}")
    return (w1, w2, s1, s2), gate_w, bias, norm_w


def run_point(fn, operands, *, warmup, iters, pipelined_iters):
    """Warmups, synchronous latency samples, then a back-to-back mean."""
    out = None
    for _ in range(warmup):
        out = fn(*operands)
        out.block_until_ready()
    lat = []
    for _ in range(iters):
        ts = time.perf_counter()
        out = fn(*operands)
        out.block_until_ready()
        lat.append((time.perf_counter() - ts) * 1e3)
    ts = time.perf_counter()
    last = None
    for _ in range(pipelined_iters):
        last = fn(*operands)
    if last is not None:
        last.block_until_ready()
    pipe = (time.perf_counter() - ts) * 1e3 / max(1, pipelined_iters)
    if out is None:
        out = last
    return out, latency_stats(lat or [float("nan")], pipe)


def point_line(bucket, routing, name, point):
    if "error" in point:
        return f"[b={bucket} {routing} {name}] FAILED {point['error'][:300]}"
    lat = point.get("latency_ms", {})
    vs = point.get("vs_baseline")
    vr = point.get("vs_reference")
    parts = [
        f"[b={bucket} {routing} {name}] med {lat.get('median', float('nan')):.3f} ms "
        f"pipe {lat.get('pipelined_mean', float('nan')):.3f} ms"
    ]
    if vs:
        parts.append(
            f"vs {vs['baseline']} rel {vs['rel_l2']:.2e} worst {vs['worst_row_rel']:.2e} "
            f"maxabs {vs['max_abs']:.2e}"
        )
    if vr:
        parts.append(f"vs ref rel {vr['rel_l2']:.4f} worst {vr['worst_row_rel']:.4f}")
    parts.append(
        f"pad0 {point.get('pad_rows_exact_zero')} nonfinite {point.get('nonfinite')} "
        f"pass {point.get('pass')}" + (" prof" if "profile" in point else "")
    )
    return " | ".join(parts)


def main(argv=None):
    args = parse_args(argv)
    if args.cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        flags = os.environ.get("XLA_FLAGS", "")
        if "device_count" not in flags:
            os.environ["XLA_FLAGS"] = (
                flags + " --xla_force_host_platform_device_count=8"
            ).strip()
    for p in (str(TREE / "python"), str(TREE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    smoke = args.cpu_smoke
    warmup = args.warmup if args.warmup is not None else (1 if smoke else 3)
    iters = args.iters if args.iters is not None else (2 if smoke else 20)
    pipelined_iters = (
        args.pipelined_iters
        if args.pipelined_iters is not None
        else (2 if smoke else 50)
    )
    profile_calls = (
        args.profile_calls if args.profile_calls is not None else (2 if smoke else 5)
    )

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    fe = _fe()
    package = fe.import_fused_moe_v2(isolated=True)
    from sglang_tpu.moe.pallas.fused_moe_v2 import layer as v2_layer
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import PUSH_OPTIONS

    build_log = _BuildLog()
    logging.getLogger("sglang_tpu").addHandler(build_log)
    import torch  # noqa: F401 - after JAX; must not bring torch_tpu along

    if "torch_tpu" in sys.modules:
        raise RuntimeError("torch_tpu got imported; it would take the TPU from JAX")

    if smoke:
        num_experts, hidden, inter, topk = 64, 1024, 768, fe.TOPK
        default_buckets = "8,16"
    else:
        num_experts, hidden, inter = fe.NUM_EXPERTS, fe.HIDDEN, fe.LOGICAL_INTER
        topk = fe.TOPK
        default_buckets = "8,64,288,2048"
    buckets = [int(b) for b in (args.buckets or default_buckets).split(",") if b]
    padded_inter = -(-inter // 512) * 512
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    routings = [r.strip() for r in args.routings.split(",") if r.strip()]
    bad = sorted(set(routings) - set(ROUTINGS))
    if bad:
        raise ValueError(f"unknown routings {bad}; known {ROUTINGS}")
    profile_routings = (
        {r.strip() for r in args.profile_routings.split(",") if r.strip()}
        if args.profile_routings
        else set(routings)
    )
    reference_buckets = (
        set(buckets)
        if args.reference
        else {int(b) for b in args.reference_buckets.split(",") if b}
    )

    devices = jax.devices()
    ep = args.ep or len(devices)
    if num_experts % ep or ep > len(devices):
        raise ValueError(
            f"ep={ep} must divide {num_experts} experts and fit {len(devices)} devices"
        )
    mesh = Mesh(np.asarray(devices[:ep]), ("ep",))
    g_local = num_experts // ep
    configs = resolve_configs(config_names, g_local, PUSH_OPTIONS)
    mode = ("cpu-smoke-" if smoke else "") + ("real" if args.weights else "synthetic")
    log(
        f"[dist] mode={mode} ep={ep} devices={devices[0].device_kind} E={num_experts} "
        f"H={hidden} I={inter}->{padded_inter} layer={args.layer} "
        f"configs={config_names} routings={routings} buckets={buckets}"
    )
    out_path = args.out or (
        f"dsv41_fused_ep_dispatch_dist_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    result = {
        "started_utc": utc_now(),
        "config": {
            "mode": mode,
            "weights": args.weights,
            "layer": args.layer if args.weights else None,
            "ep": ep,
            "num_experts": num_experts,
            "local_experts": g_local,
            "hidden": hidden,
            "logical_inter": inter,
            "padded_inter": padded_inter,
            "topk": topk,
            "scale_layout": args.scale_layout,
            "buckets_rows_per_rank": buckets,
            "pad_rows_per_rank": args.pad_rows,
            "warmup": warmup,
            "iters_sync": iters,
            "pipelined_iters": pipelined_iters,
            "profile_calls": profile_calls if args.profile_dir else 0,
            "profile_dir": args.profile_dir,
            "profile_routings": sorted(profile_routings) if args.profile_dir else [],
            "reference_buckets": sorted(reference_buckets),
            "seed": args.seed,
            "routings": routings,
            "crafted": {
                "designed_logit": f"{CRAFTED_HIGH} + U[0, {args.jitter})",
                "other_logit": f"{CRAFTED_LOW} + U[0, {args.jitter})",
                "score_bias": "zeros",
                "hot_zipf_s": args.zipf_s,
                "half_pad_rows_per_rank": "bucket // 2",
            },
            "configs": configs,
            "push_options_known": list(PUSH_OPTIONS),
            "kernel_settings": dict(
                capacity=fe.CAPACITY,
                rhs_qb=fe.RHS_QB,
                act_fn=fe.ACT_FN,
                scoring_fn=fe.SCORING_FN,
                routed_scaling_factor=fe.ROUTED_SCALING_FACTOR,
                renormalize=fe.RENORMALIZE,
                sharded_plan=True,
                logical_intermediate_size=inter,
                preprocess_topology=None,
                mesh_ep_ranks="identity",
                weight_format="fp4",
                dispatch=f"per config (wrapper overrides fused_ep.DISPATCH={fe.DISPATCH})",
            ),
            "vs_A_rel_l2_band": VS_A_REL_L2_BAND,
            "reference_bounds": {"rel_l2": REL_L2_BOUND, "worst_row": ROW_BOUND},
        },
        "environment": environment(jax, mesh),
        "buckets": [],
    }

    (w1, w2, s1, s2), gate_w, bias, norm_w = load_weights(
        args,
        fe,
        jax,
        mesh,
        num_experts=num_experts,
        hidden=hidden,
        inter=inter,
        padded_inter=padded_inter,
        result=result,
    )
    repl = NamedSharding(mesh, P())
    bias_d = jax.device_put(bias, repl)
    zero_bias_d = jax.device_put(np.zeros(num_experts, np.float32), repl)
    shard_tokens = NamedSharding(mesh, P("ep"))
    write_json(out_path, result)

    # ---------------------------------------------------------------- ops
    if smoke:
        stand_in = fe.make_reference_moe_sharded(mesh, logical_intermediate_size=inter)

        def base_fn(x, w1_, w2_, s1_, s2_, gating, *, score_bias, **_):
            return stand_in(x, gating, w1_, w2_, s1_, s2_, score_bias).astype(x.dtype)

        check_dispatch = getattr(v2_layer, "_check_dispatch", None)
    else:
        base_fn = package.fused_ep_moe_v2
        check_dispatch = None
    moes = {
        name: fe.make_fused_ep_moe(
            mesh,
            logical_intermediate_size=inter,
            kernel_fn=make_kernel_fn(
                base_fn, c["dispatch"], c["push_options"], check_dispatch
            ),
        )
        for name, c in configs.items()
    }
    ref = (
        fe.make_reference_moe_sharded(mesh, logical_intermediate_size=inter)
        if reference_buckets
        else None
    )

    n_points = n_pass = 0
    for bucket in buckets:
        tokens = ep * bucket
        bentry = {
            "bucket_per_rank": bucket,
            "tokens": tokens,
            "compile": {},
            "routings": {},
        }
        result["buckets"].append(bentry)
        try:
            x = make_hidden(
                np.random.default_rng([args.seed, 1, bucket]),
                ep=ep,
                bucket=bucket,
                hidden=hidden,
                norm_w=norm_w,
            )
            x_d = jax.device_put(x, shard_tokens)
            starts = block_starts_by_mesh_index(mesh, x_d)
            bentry["token_block_start_by_mesh_index"] = starts
            if starts != [r * bucket for r in range(ep)]:
                raise RuntimeError(f"unexpected token placement {starts}")
        except Exception as exc:  # noqa: BLE001
            bentry["error"] = f"{type(exc).__name__}: {exc}"
            bentry["traceback"] = traceback.format_exc()
            log(f"[b={bucket}] input setup FAILED: {bentry['error']}")
            write_json(out_path, result)
            continue
        abstract = (
            jax.ShapeDtypeStruct((tokens, hidden), x_d.dtype, sharding=shard_tokens),
            jax.ShapeDtypeStruct(
                (tokens, num_experts), jnp.float32, sharding=shard_tokens
            ),
            w1,
            w2,
            s1,
            s2,
            jax.ShapeDtypeStruct((num_experts,), jnp.float32, sharding=repl),
        )
        compiled = {}
        for name in configs:
            centry = bentry["compile"].setdefault(name, {})
            n_logs = len(build_log.messages)
            try:
                t0 = time.perf_counter()
                lowered = moes[name].lower(*abstract)
                t1 = time.perf_counter()
                compiled[name] = lowered.compile()
                t2 = time.perf_counter()
                centry.update(
                    trace_lower_s=round(t1 - t0, 3),
                    compile_s=round(t2 - t1, 3),
                    kernel_names=kernel_names(lowered, compiled[name]),
                    kernel_build_log=build_log.messages[n_logs:],
                )
                del lowered
                log(
                    f"[b={bucket}] compiled {name}: lower {t1 - t0:.2f}s "
                    f"compile {t2 - t1:.2f}s"
                )
            except Exception as exc:  # noqa: BLE001
                centry["error"] = f"{type(exc).__name__}: {exc}"
                centry["traceback"] = traceback.format_exc()
                log(f"[b={bucket}] compile {name} FAILED: {centry['error'][:300]}")
        write_json(out_path, result)

        for routing in routings:
            rentry = {"points": {}}
            bentry["routings"][routing] = rentry
            try:
                gating, score_bias, pad_index, designed, info = build_routing(
                    routing,
                    x=x,
                    gate_w=gate_w,
                    bias=bias,
                    ep=ep,
                    bucket=bucket,
                    num_experts=num_experts,
                    topk=topk,
                    pad_rows=args.pad_rows,
                    seed=args.seed,
                    jitter=args.jitter,
                    zipf_s=args.zipf_s,
                )
                rentry.update(info)
                real = np.ones(tokens, bool)
                real[pad_index] = False
                rows = np.nonzero(real)[0]
                sel, gap = host_select(gating[rows], score_bias, topk)
                check = {
                    "near_tie_rows_gap_lt_1e-5": int((gap < 1e-5).sum()),
                    "min_gap_topk_vs_next": float(gap.min()) if gap.size else None,
                }
                if designed is not None:
                    same = np.sort(sel, axis=1) == np.sort(designed, axis=1)
                    check["mismatched_rows"] = int((~same.all(axis=1)).sum())
                rentry["selection_check"] = check
                rentry["stats"] = routing_summary(
                    sel, rows, ep=ep, bucket=bucket, num_experts=num_experts
                )
                g_d = jax.device_put(gating, shard_tokens)
                b_d = bias_d if routing == "real" else zero_bias_d
                operands = (x_d, g_d, w1, w2, s1, s2, b_d)
                want = None
                if ref is not None and bucket in reference_buckets:
                    want = np.asarray(ref(*operands))
            except Exception as exc:  # noqa: BLE001
                rentry["error"] = f"{type(exc).__name__}: {exc}"
                rentry["traceback"] = traceback.format_exc()
                log(
                    f"[b={bucket} {routing}] routing setup FAILED: {rentry['error'][:300]}"
                )
                write_json(out_path, result)
                continue
            st = rentry["stats"]
            log(
                f"[b={bucket} {routing}] real_rows={st['real_rows']} "
                f"recv_pairs/rank max/mean={st['recv_pairs_per_rank']['max']}/"
                f"{st['recv_pairs_per_rank']['mean']} "
                f"max_pairs/expert={st['max_pairs_per_expert']} "
                f"remote_pairs/token={st['mean_remote_pairs_per_token']} "
                f"distinct_remote_ranks/token={st['mean_distinct_remote_ranks_per_token']} "
                f"push/ag rows={st['push_over_all_gather_rows']} "
                f"selection_mismatch={check.get('mismatched_rows', '-')}"
            )
            base_out, base_name = None, None
            for name in configs:
                point = {}
                rentry["points"][name] = point
                n_points += 1
                if name not in compiled:
                    point["error"] = "not compiled for this bucket"
                    point["pass"] = False
                    log(point_line(bucket, routing, name, point))
                    continue
                try:
                    fn = compiled[name]
                    out, point["latency_ms"] = run_point(
                        fn,
                        operands,
                        warmup=warmup,
                        iters=iters,
                        pipelined_iters=pipelined_iters,
                    )
                    if args.profile_dir and routing in profile_routings:
                        pdir = os.path.join(
                            args.profile_dir, f"{name}_{bucket}_{routing}"
                        )
                        t0 = time.perf_counter()
                        with jax.profiler.trace(pdir):
                            for _ in range(profile_calls):
                                fn(*operands).block_until_ready()
                        point["profile"] = {
                            "dir": pdir,
                            "calls": profile_calls,
                            "seconds": round(time.perf_counter() - t0, 2),
                        }
                    out_np = np.asarray(out.astype(jnp.float32))
                    del out
                    point["nonfinite"] = int((~np.isfinite(out_np)).sum())
                    point["pad_rows_exact_zero"] = bool(np.all(out_np[pad_index] == 0))
                    ok = point["nonfinite"] == 0 and point["pad_rows_exact_zero"]
                    if base_out is None:
                        base_out, base_name = out_np, name
                    else:
                        rel, worst, worst_i, max_abs = compare(out_np, base_out, rows)
                        point["vs_baseline"] = {
                            "baseline": base_name,
                            "rel_l2": rel,
                            "worst_row_rel": worst,
                            "worst_row_near_tie": bool(gap[worst_i] < 1e-5),
                            "max_abs": max_abs,
                        }
                        ok = ok and rel < VS_A_REL_L2_BAND
                    if want is not None:
                        rel, worst, worst_i, max_abs = compare(out_np, want, rows)
                        point["vs_reference"] = {
                            "rel_l2": rel,
                            "worst_row_rel": worst,
                            "worst_row_near_tie": bool(gap[worst_i] < 1e-5),
                            "max_abs": max_abs,
                            "ref_pad_rows_zero": bool(np.all(want[pad_index] == 0)),
                        }
                        ok = ok and rel < REL_L2_BOUND and worst < ROW_BOUND
                    point["pass"] = bool(ok)
                    del out_np
                except Exception as exc:  # noqa: BLE001
                    point["pass"] = False
                    point["error"] = f"{type(exc).__name__}: {exc}"
                    point["traceback"] = traceback.format_exc()
                n_pass += bool(point.get("pass"))
                log(point_line(bucket, routing, name, point))
            del base_out, operands, g_d, gating, want
            write_json(out_path, result)
        del compiled, x_d
        write_json(out_path, result)

    crafted_ok = all(
        r.get("selection_check", {}).get("mismatched_rows", 0) == 0
        for b in result["buckets"]
        for r in b["routings"].values()
    )
    result["summary"] = {
        "points": n_points,
        "points_pass": n_pass,
        "crafted_selection_exact": crafted_ok,
        "expert0_order_ok": result["weights"]["expert0_vs_checkpoint_block32"][
            "order_ok"
        ],
    }
    result["all_pass"] = bool(
        n_points
        and n_pass == n_points
        and crafted_ok
        and result["summary"]["expert0_order_ok"]
        and not any("error" in b for b in result["buckets"])
        and not any(
            "error" in r for b in result["buckets"] for r in b["routings"].values()
        )
    )
    result["finished_utc"] = utc_now()
    write_json(out_path, result)
    log(
        f"[dist] points {n_pass}/{n_points} pass; crafted selection exact: {crafted_ok}"
    )
    log(f"[dist] all_pass={result['all_pass']} -> {out_path}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
