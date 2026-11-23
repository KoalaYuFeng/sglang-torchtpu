# SPDX-License-Identifier: Apache-2.0
"""Step 10 L4 probe P-a (pod only): the on-chip copy of one K32Z expert (18.31 MB) between sibling TensorCores.

One Pallas kernel on all 8 devices (mesh axis "d", jax.devices() order, so mesh index = device id; devices 2k and
2k+1 share a chip). After an all-pairs entry barrier:
  copy        device SRC starts 4 remote DMAs (HBM -> DST's HBM) of one expert's K32Z operands -- gate/up codes
              [640, 4608] u32 (11.8 MB), down codes [288, 5120] u32 (5.9 MB), the two uint8 shift tables
              [160, 2304] / [96, 2560] (0.61 MB) -- optionally each split into CHUNKS DMAs; SRC waits its send
              semaphore, DST its receive semaphore; the other devices only take the barrier.
  stream      every device streams N_EXP experts' (gate/up, down) codes HBM -> VMEM through 2 slots, as the fused
              kernel's weight prefetch does (refill e + 1 while e is waited), priority 1 like WEIGHT_DMA_PRIORITY.
  streamcopy  both at once: SRC starts the copy right after the barrier, then streams; DST streams, then waits
              the copy's receive; SRC waits its send at the end.
Per-device kernel time = median over traced calls of the pallas op (profiler), plus wall clock.

  python3 l4_probe.py --modes copy,stream,streamcopy --src 5 --dst 4 --n-exp 24 --chunks 1,4 --out J --trace DIR
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(TREE, "python"), TREE):
    if p not in sys.path:
        sys.path.insert(0, p)

HIDDEN, INTER = 5120, 2304


def shapes():
    from sglang_tpu.moe.pallas.fused_moe_v2.host import k32z_table_geometry

    geo = k32z_table_geometry(HIDDEN, INTER, 4)
    # Packed fp4 codes as u32 words: [K / 8, N].
    return {
        "w1": ((HIDDEN // 8, 2 * INTER), np.uint32),
        "w2": ((INTER // 8, HIDDEN), np.uint32),
        "w1s": ((geo.rows1, geo.cols1), np.uint8),
        "w2s": ((geo.rows2, geo.cols2), np.uint8),
    }


def build(mesh, mode, src, dst, n_exp, chunks):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sh = shapes()
    names = ("w1", "w2", "w1s", "w2s")
    MESH = pl.DeviceIdType.MESH
    ndev = mesh.shape["d"]
    do_copy = mode in ("copy", "streamcopy")
    do_stream = mode in ("stream", "streamcopy")

    def kernel(*refs):
        w = refs[:4]  # [n_exp, ...] per array, HBM
        hot = refs[4:8]  # outputs [...] per array, HBM
        slots_w1, slots_w2, wsem, csem = refs[8:]
        me = lax.axis_index("d")
        barrier = pltpu.get_barrier_semaphore()
        for i in range(ndev):
            pl.semaphore_signal(barrier, 1, device_id=(jnp.int32(i),), device_id_type=MESH)
        pl.semaphore_wait(barrier, ndev)

        def pieces():
            for k in range(4):
                rows = sh[names[k]][0][0]
                step = -(-rows // chunks)
                for c in range(chunks):
                    lo, hi = c * step, min(rows, (c + 1) * step)
                    if lo < hi:
                        yield k, lo, hi - lo

        def copies():
            return [
                pltpu.make_async_remote_copy(
                    src_ref=w[k].at[0, pl.ds(lo, n)],
                    dst_ref=hot[k].at[pl.ds(lo, n)],
                    send_sem=csem.at[0],
                    recv_sem=csem.at[1],
                    device_id=(jnp.int32(dst),),
                    device_id_type=MESH,
                )
                for k, lo, n in pieces()
            ]

        def wait_bytes(which):
            # Byte-count waits on one semaphore, the fused kernel's _rows_wait pattern.
            for k, lo, n in pieces():
                ref = w[k].at[0, pl.ds(lo, n)] if which == 0 else hot[k].at[pl.ds(lo, n)]
                pltpu.make_async_copy(ref, ref, csem.at[which]).wait()

        if do_copy:

            @pl.when(me == src)
            def _():
                for cp in copies():
                    cp.start()

        if do_stream:

            def wcopy(e, s):
                return (
                    pltpu.make_async_copy(w[0].at[e], slots_w1.at[s], wsem.at[0, s]),
                    pltpu.make_async_copy(w[1].at[e], slots_w2.at[s], wsem.at[1, s]),
                )

            for cp in wcopy(0, 0):
                cp.start(priority=1)

            def body(e, carry):
                s = lax.rem(e, 2)

                @pl.when(e + 1 < n_exp)
                def _():
                    for cp in wcopy(e + 1, 1 - s):
                        cp.start(priority=1)

                for cp in wcopy(e, s):
                    cp.wait()
                return carry

            lax.fori_loop(0, n_exp, body, jnp.int32(0))

        if do_copy:

            @pl.when(me == src)
            def _():
                wait_bytes(0)

            @pl.when(me == dst)
            def _():
                wait_bytes(1)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    out_shape = [jax.ShapeDtypeStruct(sh[n][0], sh[n][1]) for n in names]
    scratch = [
        pltpu.VMEM((2,) + sh["w1"][0], sh["w1"][1]),
        pltpu.VMEM((2,) + sh["w2"][0], sh["w2"][1]),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    call = pl.pallas_call(
        kernel,
        in_specs=[hbm] * 4,
        out_specs=[hbm] * 4,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=0, vmem_limit_bytes=60 * 2**20),
        name=f"l4probe_{mode}_c{chunks}",
    )

    def local(*ws):
        return tuple(o[None] for o in call(*(x[0] for x in ws)))

    f = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=tuple(P("d") for _ in names),
            out_specs=tuple(P("d") for _ in names),
            check_vma=False,
        )
    )
    # Random words made on the devices (no host copy of ~3 GB).
    args = []
    for i, n in enumerate(names):
        shp, dt = sh[n]
        gen = jax.jit(
            lambda key, _s=(ndev, n_exp) + shp, _d=dt: jax.random.bits(key, _s, _d),
            out_shardings=NamedSharding(mesh, P("d")),
        )
        args.append(gen(jax.random.key(i)))
    return f, args


def kernel_us(trace_dir, name):
    import dsv41_k32z_trace as TR

    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    dev = TR.load(files[-1])
    out = {}
    for d, rec in dev.items():
        durs = [(e - s) / 1e3 for s, e, n, _ in rec["ops"] if name in n]
        if durs:
            out[int(d)] = round(statistics.median(durs), 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="copy,stream,streamcopy")
    p.add_argument("--src", type=int, default=5)
    p.add_argument("--dst", type=int, default=4)
    p.add_argument("--n-exp", type=int, default=24)
    p.add_argument("--chunks", default="1,4")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--trace", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import jax
    from jax.sharding import Mesh

    devs = jax.devices()
    mesh = Mesh(np.asarray(devs), ("d",))
    res = {"devices": [str(d) for d in devs], "src": a.src, "dst": a.dst, "n_exp": a.n_exp, "points": []}
    sh = shapes()
    res["copy_bytes"] = int(sum(np.prod(s) * np.dtype(t).itemsize for s, t in sh.values()))
    res["stream_bytes_per_dev"] = int(a.n_exp * sum(np.prod(sh[n][0]) * 4 for n in ("w1", "w2")))
    for mode in a.modes.split(","):
        for chunks in [int(c) for c in a.chunks.split(",")] if mode != "stream" else [1]:
            rec = {"mode": mode, "chunks": chunks}
            try:
                f, args = build(mesh, mode, a.src, a.dst, a.n_exp, chunks)
                t0 = time.perf_counter()
                out = jax.block_until_ready(f(*args))
                rec["first_call_s"] = round(time.perf_counter() - t0, 2)
                if mode != "stream":
                    # The copy is exact: DST's hot buffers = SRC's expert 0.
                    ok = True
                    for k, n in enumerate(("w1", "w2", "w1s", "w2s")):
                        got = np.asarray(out[k][a.dst])
                        want = np.asarray(args[k][a.src, 0])
                        ok = ok and bool(np.array_equal(got, want))
                    rec["copy_exact"] = ok
                ts = []
                for _ in range(3):
                    t1 = time.perf_counter()
                    for _ in range(10):
                        r = f(*args)
                    jax.block_until_ready(r)
                    ts.append((time.perf_counter() - t1) / 10 * 1e6)
                rec["wall_us"] = [round(t, 1) for t in ts]
                td = os.path.join(a.trace, f"{mode}_c{chunks}")
                os.makedirs(td, exist_ok=True)
                with jax.profiler.trace(td):
                    for _ in range(a.calls):
                        r = f(*args)
                    jax.block_until_ready(r)
                rec["kernel_us_by_dev"] = kernel_us(td, f"l4probe_{mode}_c{chunks}")
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            print(json.dumps(rec), flush=True)
            res["points"].append(rec)
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
