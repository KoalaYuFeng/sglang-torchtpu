# SPDX-License-Identifier: Apache-2.0
"""K32Z grouped expert FFN on ONE TensorCore: the compute path of the fused EP
MoE kernel (sglang_tpu/moe/pallas/fused_moe_v2/kernel.py, K32Z weight format,
tree k32z-e2e 44f1da5db3) for one EP rank's 48 local experts, as a standalone
Pallas kernel with no dispatch and no return (Step 8, K8).

What is ported unchanged (functions copied from kernel.py): the K32Z decode
(k32z_build_table, k32z_block_readers, k32z_chunk_decoders), the FFN bodies
(expert_ffn_colscale, expert_ffn_colscale_rolled), the column-scale fill, the
intermediate requantization (_intermediate_rows), the activation row scales
(_tile_row_scales) and the wire epilogue (wire_row_and_scale: e4m3 rows with
one f32 scale per row).

What is the fused kernel's schedule, re-implemented on contiguous rows: the
visit list (experts with rows, in expert order); two weight slots, prefetch
distance 1 (an expert's four weight copies start at the head of the previous
visit, tables first, on DMA priority 1); a visit waits its two tables at its
head and each weight slab in the first block that reads it when the tile
height is <= STREAMED_ROWS (the rung covering the live rows), or both slabs at
the visit head for a taller first tile; 256-row tiles, the tail tile on the
smallest rung that covers its live rows; a tile's activation rows are issued
one tile ahead (the next expert's first tile before the current expert's last
FFN); the column scales are rebuilt every tile.

What differs from the fused kernel: the activation rows of an expert are
contiguous in `x` (the fused kernel gathers them per row or receives them by
push), and a tile's result goes to `out` (bf16, the wire row times its scale)
instead of the wire.

Contract:
  computation(x, x_scale, w1, w2, w1s, w2s, sizes, starts, plan) -> bf16 [M, 40, 128]
    x       float8_e4m3fn [M, 40, 128]   activation rows (row r = x[r].reshape(5120)),
                                         expert e's rows are x[starts[e] : starts[e] + sizes[e]]
    x_scale float32 [S, 128]             one scale per row, row r at [r // 128, r % 128];
                                         S >= ceil(M / 128) + 3
    w1      float4_e2m1fn [48, 5120, 4608]   gate|up codes (K32Z: e2m1 -> e5m2 + 4 * shift)
    w2      float4_e2m1fn [48, 2304, 5120]   down codes
    w1s     uint8 [48, 160, 576 * bits]  gate/up shift table (bits 2 or 4 per field)
    w2s     uint8 [48, 96, 2560]         down shift table + the column base bytes (rows 72..75)
    sizes   int32 [48]                   rows per local expert (>= 0)
    starts  int32 [48]                   first row of each expert in x / out
    plan    static str: "decode" | "mid" | "t1" | "t2" (the fused kernel's per-bucket
            K32Z build plan: rungs and looped bodies)
  out[r] = bf16(q * s) where (q, s) = the fused kernel's wire row and scale of row r's
  expert FFN (silu_and_mul_with_clamp, limit 10); rows outside every expert are not
  written (left as the output buffer's initial contents, zeros here).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HIDDEN = 5120
INTER = 2304
N_EXPERTS = 48
LANES = 128
LANE_BLOCKS = HIDDEN // LANES
FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
K32Z_WEIGHT_DTYPE = jnp.float8_e5m2
FP8_MAX = 448.0
PACK4 = 8
DN = (((1,), (0,)), ((), ()))
QCHUNK = 512
K32Z_GROUP = 32
K32Z_LANE_TILE = 128
K32Z_U8_SUBLANE_TILE = 32
K32Z_BASE_ROWS = 4
K32Z_DOWN_BITS = 4
K32Z_CHUNK = 256
K32Z_KBLOCK = 512
K32Z_SHIFT_WORD = 0x04040404
K32Z_GROUP_WORDS = K32Z_GROUP // 4
TILE_M = 256
STREAMED_ROWS = 128
WEIGHT_DMA_PRIORITY = 1
VMEM_FRACTION = 0.98
CLAMP_LIMIT = 10.0

# The fused kernel's K32Z build plans (host.k32z_build_plan at EP8, 48 local
# experts): rung heights below the 256-row tile, and (unroll, looped heights).
PLANS = {
    "decode": ((32, 64, 128), (2, (128, 256))),  # 8-64 rows/rank: K32Z_DECODE_ROLL (2, 128)
    "mid": ((32, 64, 128), (0, ())),  # 512 rows/rank
    "t1": ((64, 128), (0, ())),  # 1024 rows/rank: K32Z_LARGE_PLAN tier 128
    "t2": ((32, 128), (0, ())),  # 2048 rows/rank: K32Z_LARGE_PLAN tier 256
    # K8 R1 measurement plans (decode rungs, other roll rules): only the 256 body looped (K32Z_DECODE_ROLL
    # (2, 256)), and no looped body at all (K32Z_DECODE_ROLL None).
    "decode_r256": ((32, 64, 128), (2, (256,))),
    "decode_u": ((32, 64, 128), (0, ())),
    # K9: the decode plan with (gate/up, down) chunks per loop step other than (2, 2).
    "decode_u34": ((32, 64, 128), ((3, 4), (128, 256))),
    "decode_u65": ((32, 64, 128), ((6, 5), (128, 256))),
    "decode_u910": ((32, 64, 128), ((9, 10), (128, 256))),
}


def align_up(x, a):
    return -(-x // a) * a


def table_geometry(gate_up_bits):
    g1, g2 = HIDDEN // K32Z_GROUP, INTER // K32Z_GROUP
    return dict(
        g1=g1,
        g2=g2,
        rows1=align_up(g1, K32Z_U8_SUBLANE_TILE),
        cols1=2 * INTER * gate_up_bits // 8,
        rows2=align_up(g2 + K32Z_BASE_ROWS, K32Z_U8_SUBLANE_TILE),
        cols2=HIDDEN * K32Z_DOWN_BITS // 8,
        tab_rows=align_up(max(g1, g2), 8),
    )


# --------------------------------------------------------------------------- #
# Ported from fused_moe_v2/kernel.py (unchanged bodies).
# --------------------------------------------------------------------------- #
def silu_and_mul_with_clamp(gate, up, limit=CLAMP_LIMIT):
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    return jax.nn.silu(gate) * up


def _row_scale(amax):
    scale = amax.astype(jnp.float32) / FP8_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    return scale, sinv, usable


def _quant_apply(x, sinv_native, usable):
    return jnp.where(usable, x * sinv_native, jnp.zeros_like(x)).astype(FP8)


def _absmax_rows(x):
    return jnp.max(jnp.abs(x), axis=-1, keepdims=True)


def rowquant_fp8(x):
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale, sinv, usable = _row_scale(amax)
    return _quant_apply(x, sinv.astype(x.dtype), usable), scale


def wire_row_and_scale(acc2, mid_scale, *, w2s):
    down = acc2 * mid_scale
    down = down * w2s
    return rowquant_fp8(down.astype(jnp.bfloat16))


def _intermediate_rows(acc1, act_scale, inter, *, w1s):
    def half(lo, hi):
        scaled = acc1[:, lo:hi]
        scaled = scaled * act_scale
        scaled = scaled * w1s[:, lo:hi]
        return scaled

    mid_chunks, amax = [], None
    for c0 in range(0, inter, QCHUNK):
        c1 = min(c0 + QCHUNK, inter)
        gate = half(c0, c1)
        up = half(inter + c0, inter + c1)
        chunk = silu_and_mul_with_clamp(gate, up).astype(jnp.bfloat16)
        mid_chunks.append(chunk)
        chunk_amax = _absmax_rows(chunk)
        amax = chunk_amax if amax is None else jnp.maximum(amax, chunk_amax)
    mid_scale, mid_sinv, usable = _row_scale(amax)
    sinv_bf16 = mid_sinv.astype(jnp.bfloat16)
    mid_q = jnp.concatenate(
        [_quant_apply(chunk, sinv_bf16, usable) for chunk in mid_chunks], axis=-1
    )
    return mid_q, mid_scale


def _tile_row_scales(window, row_base, tile_m):
    lanes = LANES
    resid = lax.rem(row_base, jnp.int32(lanes))
    rot = pltpu.roll(window, lanes - resid, 1)
    lane = lax.broadcasted_iota(jnp.int32, rot[:-1].shape, 1)
    rows = jnp.where(lane < lanes - resid, rot[:-1], rot[1:])
    return rows.reshape(-1)[:tile_m][:, None]


def k32z_blocks(k):
    return tuple((k0, min(K32Z_KBLOCK, k - k0)) for k0 in range(0, k, K32Z_KBLOCK))


def k32z_shift_words(raw, field, bits):
    mask = (1 << bits) - 1
    return jnp.bitwise_and(jnp.right_shift(raw, bits * field), mask) * K32Z_SHIFT_WORD


def k32z_build_table(shift_vm, slot, n_start, groups, tab_vm, parity, bits):
    lanes = K32Z_LANE_TILE
    byte_tile, field = divmod(n_start // lanes, 8 // bits)
    raw = shift_vm[slot, :, pl.ds(byte_tile * lanes, lanes)].astype(jnp.int32)
    raw = raw[:groups]
    for t in range(K32Z_CHUNK // lanes):
        tab_vm[parity, t, pl.ds(0, groups), :] = k32z_shift_words(raw, field + t, bits)


def k32z_fill_colscale(w2s_vm, slot, col_vm, *, hidden, inter):
    lanes = K32Z_LANE_TILE
    g2 = inter // K32Z_GROUP
    first = g2 - g2 % K32Z_U8_SUBLANE_TILE
    rows = align_up(g2 + K32Z_BASE_ROWS, K32Z_U8_SUBLANE_TILE) - first
    r0 = g2 - first

    def scale_row(block, r):
        return pltpu.bitcast(jnp.left_shift(block[r : r + 1], 23), jnp.float32)

    for byte_tile in range(hidden * K32Z_DOWN_BITS // 8 // lanes):
        block = w2s_vm[slot, pl.ds(first, rows), pl.ds(byte_tile * lanes, lanes)].astype(
            jnp.int32
        )
        for half in range(2):
            tile = 2 * byte_tile + half
            cols = pl.ds(tile * lanes, lanes)
            col_vm[pl.ds(1, 1), cols] = scale_row(block, r0 + half)
            if tile * lanes < 2 * inter:
                col_vm[pl.ds(0, 1), cols] = scale_row(block, r0 + 2 + half)


def k32z_block_readers(
    w1_vm, w2_vm, w1s_vm, w2s_vm, tab_vm, slot, *, hidden, inter, gate_up_bits, wait_slab=None,
    build_tables=True,
):
    # build_tables=False (K9 9a, timing only): the readers skip the per-chunk table builds.
    lanes = K32Z_LANE_TILE
    tiles = K32Z_CHUNK // lanes
    blocks = (k32z_blocks(hidden), k32z_blocks(inter))
    groups = (hidden // K32Z_GROUP, inter // K32Z_GROUP)
    bits = (gate_up_bits, K32Z_DOWN_BITS)
    slabs = ((w1_vm, w1s_vm), (w2_vm, w2s_vm))

    def shift_words(parity, g0, n_groups):
        def load(t, g):
            return tab_vm[parity, t, pl.ds(g, K32Z_GROUP_WORDS, stride=0), :]

        return jnp.concatenate(
            [
                jnp.concatenate([load(t, g0 + i) for t in range(tiles)], axis=1)
                for i in range(n_groups)
            ],
            axis=0,
        )

    def reader(m):
        w_vm, shift_vm = slabs[m]

        def block(b, n_start, n_end):
            k0, rows = blocks[m][b]
            parity = (n_start // K32Z_CHUNK) % 2
            if b == 0:
                if wait_slab is not None and n_start == 0:
                    wait_slab(m + 1)
                if build_tables:
                    k32z_build_table(shift_vm, slot, n_start, groups[m], tab_vm, parity, bits[m])
            codes = w_vm[slot, pl.ds(k0 // PACK4, rows // PACK4), pl.ds(n_start, n_end - n_start)]
            e5 = pltpu.bitcast(codes, FP4).astype(K32Z_WEIGHT_DTYPE)
            words = pltpu.bitcast(e5, jnp.int32) + shift_words(
                parity, k0 // K32Z_GROUP, rows // K32Z_GROUP
            )
            return pltpu.bitcast(words, K32Z_WEIGHT_DTYPE)

        return block

    return reader(0), reader(1)


def expert_ffn_colscale(act_q, act_scale, w1_block, w2_block, w1s_col, *, hidden):
    inter = w1s_col.shape[-1] // 2

    def contract(lhs, rhs_block, k, n_out):
        chunks = []
        for n_start in range(0, n_out, K32Z_CHUNK):
            n_end = n_start + K32Z_CHUNK
            acc = None
            for b, (k0, rows) in enumerate(k32z_blocks(k)):
                part = lax.dot_general(
                    lhs[:, k0 : k0 + rows],
                    rhs_block(b, n_start, n_end),
                    DN,
                    preferred_element_type=jnp.float32,
                )
                acc = part if acc is None else acc + part
            chunks.append(acc)
        return jnp.concatenate(chunks, axis=-1)

    acc1 = contract(act_q, w1_block, hidden, 2 * inter)
    mid_q, mid_scale = _intermediate_rows(acc1, act_scale, inter, w1s=w1s_col)
    acc2 = contract(mid_q, w2_block, inter, hidden)
    return acc2, mid_scale


def _pow2_split(x, d):
    if d == 1:
        return x, 0
    return jnp.right_shift(x, d.bit_length() - 1), jnp.bitwise_and(x, d - 1)


def k32z_chunk_decoders(
    w1_vm, w2_vm, w1s_vm, w2s_vm, tab_vm, slot, *, hidden, inter, gate_up_bits, prebuilt=False
):
    # prebuilt=True (K9 "pt"): block 0 builds no table; the caller builds chunk n's table ahead with
    # k32z_chunk_tables (the same words into the same parity), returned as a third value.
    lanes = K32Z_LANE_TILE
    tiles = K32Z_CHUNK // lanes
    blocks = (k32z_blocks(hidden), k32z_blocks(inter))
    groups = (hidden // K32Z_GROUP, inter // K32Z_GROUP)
    bits = (gate_up_bits, K32Z_DOWN_BITS)
    slabs = ((w1_vm, w1s_vm), (w2_vm, w2s_vm))

    def shift_words(parity, g0, n_groups):
        def load(t, g):
            return tab_vm[parity, t, pl.ds(g, K32Z_GROUP_WORDS, stride=0), :]

        return jnp.concatenate(
            [
                jnp.concatenate([load(t, g0 + i) for t in range(tiles)], axis=1)
                for i in range(n_groups)
            ],
            axis=0,
        )

    def build_table(shift_vm, n, n_groups, parity, nbits):
        fields = 8 // nbits
        byte_tile, field = _pow2_split(n * tiles, fields)
        if tiles % fields == 0:
            field = 0
        start = pl.multiple_of(byte_tile * lanes, lanes)
        raw = shift_vm[slot, :, pl.ds(start, lanes)].astype(jnp.int32)
        raw = raw[:n_groups]
        for t in range(tiles):
            tab_vm[parity, t, pl.ds(0, n_groups), :] = k32z_shift_words(raw, field + t, nbits)

    def decoder(m):
        w_vm, shift_vm = slabs[m]

        def block(b, n, parity):
            k0, rows = blocks[m][b]
            if b == 0 and not prebuilt:
                build_table(shift_vm, n, groups[m], parity, bits[m])
            n_start = pl.multiple_of(n * K32Z_CHUNK, K32Z_CHUNK)
            codes = w_vm[slot, pl.ds(k0 // PACK4, rows // PACK4), pl.ds(n_start, K32Z_CHUNK)]
            e5 = pltpu.bitcast(codes, FP4).astype(K32Z_WEIGHT_DTYPE)
            words = pltpu.bitcast(e5, jnp.int32) + shift_words(
                parity, k0 // K32Z_GROUP, rows // K32Z_GROUP
            )
            return pltpu.bitcast(words, K32Z_WEIGHT_DTYPE)

        return block

    def builder(m):
        shift_vm = slabs[m][1]

        def build(n, parity):
            build_table(shift_vm, n, groups[m], parity, bits[m])

        return build

    if prebuilt:
        return decoder(0), decoder(1), (builder(0), builder(1))
    return decoder(0), decoder(1)


def expert_ffn_colscale_rolled(
    act_q, act_scale, decode1, decode2, w1s_col, acc_vm, *, hidden, unroll=1, wait_slab=None,
    tables=None,
):
    """tables=(build1, build2) (K9 "pt"): the decoders build no table; each matmul's loop is software-pipelined
    on the tables: chunks 0..unroll-1's tables are built before the loop, and right after chunk n's last decode
    block (its table half retired) chunk n + unroll's table is built into the same half (clamped to the last
    chunk, so the final step rebuilds a retired half with the last chunk's words). Same words, same decode, same
    dot order: bitwise the unpipelined body."""
    rows = act_q.shape[0]
    inter = w1s_col.shape[-1] // 2
    chunk = K32Z_CHUNK
    # K9: unroll may be (gate/up, down) chunks a step; each must divide its matmul's chunk count (18 / 20).
    u_gu, u_dn = (unroll, unroll) if isinstance(unroll, int) else unroll

    def contract(lhs, decode, k, n_out, build=None, unroll=1):
        n_chunks = n_out // chunk
        steps, rem = divmod(n_chunks, unroll)
        if rem:
            raise ValueError(f"{n_chunks} chunks are not whole steps of {unroll}")
        if build is not None and unroll % 2:
            raise ValueError("pt (prebuilt tables) needs an even unroll (one table half per chunk parity)")
        blocks = k32z_blocks(k)
        if build is not None:
            for u in range(unroll):
                build(jnp.int32(u), u % 2)

        def step(j, carry):
            for u in range(unroll):
                n = j * unroll + u
                acc = None
                for b, (k0, k_rows) in enumerate(blocks):
                    part = lax.dot_general(
                        lhs[:, k0 : k0 + k_rows],
                        decode(b, n, u % 2),
                        DN,
                        preferred_element_type=jnp.float32,
                    )
                    acc = part if acc is None else acc + part
                if build is not None:
                    build(jnp.minimum(n + unroll, n_chunks - 1), u % 2)
                n_start = pl.multiple_of(n * chunk, chunk)
                acc_vm[pl.ds(0, rows), pl.ds(n_start, chunk)] = acc
            return carry

        lax.fori_loop(0, steps, step, jnp.int32(0))
        return acc_vm[pl.ds(0, rows), pl.ds(0, n_out)]

    build1, build2 = tables if tables is not None else (None, None)
    if wait_slab is not None:
        wait_slab(1)
    acc1 = contract(act_q, decode1, hidden, 2 * inter, build1, u_gu)
    mid_q, mid_scale = _intermediate_rows(acc1, act_scale, inter, w1s=w1s_col)
    if wait_slab is not None:
        wait_slab(2)
    acc2 = contract(mid_q, decode2, inter, hidden, build2, u_dn)
    return acc2, mid_scale


# --------------------------------------------------------------------------- #
# The grouped kernel: the fused kernel's per-visit schedule on contiguous rows.
# --------------------------------------------------------------------------- #
KNOBS = ("wire", "fa", "fo", "ds", "hc", "nc", "nt", "uw", "nr128", "nr256", "pt", "u34", "u65", "u910")


def parse_knobs(knobs):
    kn = frozenset(k for k in (knobs or "").split(",") if k and k != "base")
    bad = kn - set(KNOBS)
    if bad:
        raise ValueError(f"unknown knobs {sorted(bad)}")
    return kn


def max_tiles(m_rows):
    """Upper bound on the visit tiles of a call (sum over experts of ceil(rows / TILE_M))."""
    return -(-m_rows // TILE_M) + N_EXPERTS


@functools.lru_cache(maxsize=None)
def build(m_rows, scale_rows, plan="decode", gate_up_bits=4, mode="full", w_split=1, knobs=""):
    """`mode` (K8 measurement only; "full" is the baseline): "dma" runs the weight stream, the activation and
    output DMAs and no FFN (the slabs are waited on each expert's first tile); "comp" runs the FFN on whatever
    the weight slots hold, with no weight DMA and no weight wait. `w_split` (K8 R2) issues each weight slab as
    that many DMAs along its packed-K rows, all on the slab's semaphore (1: the baseline's one DMA per slab).

    `knobs` (K9 9a attribution; comma list, "" = the K8 baseline gv2c):
      wire   the fused kernel's epilogue: e4m3 wire rows staged [h, 40, 128] and the row scales staged [1, h],
             both stored under static-parity branches; two output DMAs per tile (the rows; one [1, 256] scale
             row per tile); computation() rebuilds bf16(q * s) outside the kernel (= the baseline's product)
      fa     flat activation slab: lhs_vm [2, 256, 5120], x [M, 5120] (no [h, 40, 128] -> [h, 5120] reshape)
      fo     flat output staging: out_vm [2, 256, 5120] (no [h, 5120] -> [h, 40, 128] reshape)
      ds     dynamic-parity output stores (out_vm[par] = ..., no static-parity branches)
      hc     column scales filled once per visit at its head (not per tile)
      nc     TIMING ONLY: no column-scale fill (col_vm keeps whatever it held)
      nt     TIMING ONLY: no shift-table builds (tab_vm keeps whatever it held)
      uw     every visit waits both weight slabs at its head (no conditional waits inside the bodies)
      nr128 / nr256  TIMING ONLY (uniform full tiles): one body of that height, no rung dispatch
      pt     looped bodies software-pipelined on the shift tables (expert_ffn_colscale_rolled tables=; bitwise)
    """
    if mode not in ("full", "dma", "comp"):
        raise ValueError(mode)
    kn = parse_knobs(knobs)
    wire, fa, fo, ds = "wire" in kn, "fa" in kn, "fo" in kn, "ds" in kn
    hc, nc, nt, uw, pt = "hc" in kn, "nc" in kn, "nt" in kn, "uw" in kn, "pt" in kn
    nr = 128 if "nr128" in kn else (256 if "nr256" in kn else None)
    rungs, (unroll, roll_heights) = PLANS[plan]
    for k, u in (("u34", (3, 4)), ("u65", (6, 5)), ("u910", (9, 10))):
        if k in kn and unroll:
            unroll = u  # K9: the plan's looped bodies with (gate/up, down) chunks per step u
    heights = (nr,) if nr else tuple(r for r in rungs if r < TILE_M) + (TILE_M,)
    streamed = 0 if uw else max((r for r in heights if r <= STREAMED_ROWS), default=0)
    head_waits = TILE_M > streamed
    geo = table_geometry(gate_up_bits)
    win = align_up(TILE_M, LANES) // LANES + 1
    rolls = bool(unroll) and any(h in roll_heights for h in heights)
    hidden, inter = HIDDEN, INTER
    t_max = max_tiles(m_rows)
    fill_in_tile = not (hc or nc)

    def kernel(sizes_sm, starts_sm, visit_sm, nvis_sm, *refs):
        it = iter(refs)
        x_hbm, xs_hbm, w1_hbm, w2_hbm, w1s_hbm, w2s_hbm = (next(it) for _ in range(6))
        out_hbm = next(it)
        scl_hbm = next(it) if wire else None
        lhs_vm, ls_vm, w1_vm, w2_vm, w1s_vm, w2s_vm, tab_vm, col_vm = (
            next(it) for _ in range(8)
        )
        acc_vm = next(it) if rolls else None
        out_vm = next(it)
        oscl_vm = next(it) if wire else None
        lhs_sems, w1_sems, w2_sems, w1s_sems, w2s_sems, out_sems = (next(it) for _ in range(6))
        osc_sems = next(it) if wire else None
        w1_hbm = w1_hbm.bitcast(jnp.uint32)
        w2_hbm = w2_hbm.bitcast(jnp.uint32)
        n_visit = nvis_sm[0]

        def w1_copy(e, s):
            return pltpu.make_async_copy(w1_hbm.at[e], w1_vm.at[s], w1_sems.at[s])

        def w2_copy(e, s):
            return pltpu.make_async_copy(w2_hbm.at[e], w2_vm.at[s], w2_sems.at[s])

        def w1s_copy(e, s):
            return pltpu.make_async_copy(w1s_hbm.at[e], w1s_vm.at[s], w1s_sems.at[s])

        def w2s_copy(e, s):
            return pltpu.make_async_copy(w2s_hbm.at[e], w2s_vm.at[s], w2s_sems.at[s])

        def slab_parts(hbm, vm, sems, e, s, rows):
            if w_split == 1:
                return [pltpu.make_async_copy(hbm.at[e], vm.at[s], sems.at[s])]
            n = rows // w_split
            return [
                pltpu.make_async_copy(
                    hbm.at[e, pl.ds(i * n, n)], vm.at[s, pl.ds(i * n, n)], sems.at[s]
                )
                for i in range(w_split)
            ]

        def weight_starts(e, s):
            if mode == "comp":
                return
            w1s_copy(e, s).start(priority=WEIGHT_DMA_PRIORITY)
            w2s_copy(e, s).start(priority=WEIGHT_DMA_PRIORITY)
            for c in slab_parts(w1_hbm, w1_vm, w1_sems, e, s, hidden // PACK4):
                c.start(priority=WEIGHT_DMA_PRIORITY)
            for c in slab_parts(w2_hbm, w2_vm, w2_sems, e, s, inter // PACK4):
                c.start(priority=WEIGHT_DMA_PRIORITY)

        def lhs_copy(row0, nrows, par):
            if fa:
                # A flat [rows, 5120] slab takes only tile-aligned DMA sizes: always a whole tile (x carries
                # TILE_M padding rows; rows past the live ones are never read by the rung that runs).
                # Tile-aligned offsets too: asserted 128-row aligned (true for the uniform M_e = 128 k configs
                # fa is timed on; any other row layout is unsupported).
                return pltpu.make_async_copy(
                    x_hbm.at[pl.ds(pl.multiple_of(row0, LANES), TILE_M)], lhs_vm.at[par], lhs_sems.at[par]
                )
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(row0, nrows)], lhs_vm.at[par, pl.ds(0, nrows)], lhs_sems.at[par]
            )

        def ls_copy(row0, par):
            return pltpu.make_async_copy(
                xs_hbm.at[pl.ds(row0 // LANES, win)], ls_vm.at[par], lhs_sems.at[par]
            )

        def lhs_issue(row0, nrows, par):
            lhs_copy(row0, nrows, par).start()
            ls_copy(row0, par).start()

        def lhs_wait(row0, nrows, par):
            lhs_copy(row0, nrows, par).wait()
            ls_copy(row0, par).wait()

        def out_copy(row0, nrows, par):
            if fo:
                # Flat staging: whole-tile DMAs only (out carries TILE_M padding rows). A tile with fewer live
                # rows also writes the rows after them (the next expert's, which its own later DMA rewrites;
                # the two DMAs can race, so fo is bitwise-checked only where every tile is full).
                return pltpu.make_async_copy(
                    out_vm.at[par], out_hbm.at[pl.ds(pl.multiple_of(row0, LANES), TILE_M)], out_sems.at[par]
                )
            return pltpu.make_async_copy(
                out_vm.at[par, pl.ds(0, nrows)], out_hbm.at[pl.ds(row0, nrows)], out_sems.at[par]
            )

        def scl_copy(tile_i, par):
            return pltpu.make_async_copy(oscl_vm.at[par], scl_hbm.at[tile_i], osc_sems.at[par])

        def out_wait(nrows, par):
            out_copy(0, nrows, par).wait()
            if wire:
                scl_copy(0, par).wait()

        # Prologue: visit 0's weights (prefetch distance 1) and first tile.
        @pl.when(n_visit > 0)
        def _():
            e0 = visit_sm[0]
            weight_starts(e0, 0)
            lhs_issue(starts_sm[e0], jnp.minimum(sizes_sm[e0], TILE_M), 0)

        def visit_step(visit_i, carry):
            e = visit_sm[visit_i]
            slot = lax.rem(visit_i, jnp.int32(2))
            if mode != "comp":
                w1s_copy(e, slot).wait()
                w2s_copy(e, slot).wait()

            @pl.when(visit_i + 1 < n_visit)
            def _():
                weight_starts(visit_sm[visit_i + 1], jnp.int32(1) - slot)

            rows = sizes_sm[e]
            base = starts_sm[e]
            if head_waits and mode != "comp":

                @pl.when(rows > streamed)
                def _():
                    w1_copy(e, slot).wait()
                    w2_copy(e, slot).wait()

            if hc and mode != "dma":
                # K9 hc: the visit's column scales once, at its head (its w2s slot was waited above).
                k32z_fill_colscale(w2s_vm, slot, col_vm, hidden=hidden, inter=inter)

            n_tiles = (rows + TILE_M - 1) // TILE_M

            def tile_body(t, carry):
                tile_count, pend0, pend1 = carry
                par = lax.rem(tile_count, jnp.int32(2))
                row0 = base + t * TILE_M
                live = jnp.minimum(rows - t * TILE_M, TILE_M)
                lhs_wait(row0, live, par)
                other = jnp.int32(1) - par

                @pl.when(t + 1 < n_tiles)
                def _():
                    lhs_issue(row0 + TILE_M, jnp.minimum(rows - (t + 1) * TILE_M, TILE_M), other)

                @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
                def _():
                    nxt = visit_sm[visit_i + 1]
                    lhs_issue(starts_sm[nxt], jnp.minimum(sizes_sm[nxt], TILE_M), other)

                # out_vm[par] is free once its previous store has landed.
                pend = jnp.where(par == 0, pend0, pend1)

                @pl.when(pend > 0)
                def _():
                    out_wait(pend, par)

                first_tile = t == 0

                def compute_tile(height):
                    act = lhs_vm[par, pl.ds(0, height)]
                    if not fa:
                        act = act.reshape(height, hidden)
                    act_scales = _tile_row_scales(ls_vm[par], row0, height)
                    if fill_in_tile:
                        k32z_fill_colscale(w2s_vm, slot, col_vm, hidden=hidden, inter=inter)
                    if height <= streamed and mode != "comp":

                        def wait_slab(which):
                            @pl.when(first_tile)
                            def _():
                                (w1_copy if which == 1 else w2_copy)(e, slot).wait()

                    else:
                        wait_slab = None
                    w1s_col = col_vm[pl.ds(0, 1), pl.ds(0, 2 * inter)]
                    if rolls and height in roll_heights:
                        decs = k32z_chunk_decoders(
                            w1_vm, w2_vm, w1s_vm, w2s_vm, tab_vm, slot,
                            hidden=hidden, inter=inter, gate_up_bits=gate_up_bits, prebuilt=pt,
                        )
                        acc2, mid_scale = expert_ffn_colscale_rolled(
                            act, act_scales, decs[0], decs[1], w1s_col, acc_vm,
                            hidden=hidden, unroll=unroll, wait_slab=wait_slab,
                            tables=decs[2] if pt else None,
                        )
                    else:
                        w1_block, w2_block = k32z_block_readers(
                            w1_vm, w2_vm, w1s_vm, w2s_vm, tab_vm, slot,
                            hidden=hidden, inter=inter, gate_up_bits=gate_up_bits,
                            wait_slab=wait_slab, build_tables=not nt,
                        )
                        acc2, mid_scale = expert_ffn_colscale(
                            act, act_scales, w1_block, w2_block, w1s_col, hidden=hidden
                        )
                    w2s_row = col_vm[pl.ds(1, 1), pl.ds(0, hidden)]
                    q, s = wire_row_and_scale(acc2, mid_scale, w2s=w2s_row)
                    if wire:
                        rows_st = q if fo else q.reshape(height, LANE_BLOCKS, LANES)
                        scl_st = s.reshape(1, height)
                    else:
                        rows_st = (q.astype(jnp.float32) * s).astype(jnp.bfloat16)
                        rows_st = rows_st if fo else rows_st.reshape(height, LANE_BLOCKS, LANES)
                        scl_st = None

                    def store(p):
                        if height == TILE_M:
                            out_vm[p] = rows_st
                            if wire:
                                oscl_vm[p] = scl_st
                        else:
                            out_vm[p, pl.ds(0, height)] = rows_st
                            if wire:
                                oscl_vm[p, :, pl.ds(0, height)] = scl_st

                    if ds:
                        store(par)
                    else:

                        @pl.when(par == 0)
                        def _():
                            store(0)

                        @pl.when(par == 1)
                        def _():
                            store(1)

                if mode == "dma":
                    # No FFN: the slabs a streamed visit owes are waited on its first tile.
                    owes = first_tile
                    if head_waits:
                        owes = jnp.logical_and(owes, rows <= streamed)

                    @pl.when(owes)
                    def _():
                        w1_copy(e, slot).wait()
                        w2_copy(e, slot).wait()

                else:
                    for i, h in enumerate(heights):
                        below = heights[i - 1] if i else 0

                        @pl.when(jnp.logical_and(live > below, live <= h))
                        def _(h=h):
                            compute_tile(h)

                out_copy(row0, live, par).start()
                if wire:
                    scl_copy(tile_count, par).start()
                pend0 = jnp.where(par == 0, live, pend0)
                pend1 = jnp.where(par == 1, live, pend1)
                return tile_count + 1, pend0, pend1

            return lax.fori_loop(0, n_tiles, tile_body, carry)

        z = jnp.int32(0)
        _, pend0, pend1 = lax.fori_loop(0, n_visit, visit_step, (z, z, z))

        @pl.when(pend0 > 0)
        def _():
            out_wait(pend0, 0)

        @pl.when(pend1 > 0)
        def _():
            out_wait(pend1, 1)

    any_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    row_shape = (HIDDEN,) if fa else (LANE_BLOCKS, LANES)
    out_row_shape = (HIDDEN,) if fo else (LANE_BLOCKS, LANES)
    out_dtype = FP8 if wire else jnp.bfloat16
    scratch = [
        pltpu.VMEM((2, TILE_M) + row_shape, FP8),  # lhs_vm
        pltpu.VMEM((2, win, LANES), jnp.float32),  # ls_vm
        pltpu.VMEM((2, hidden // PACK4, 2 * inter), jnp.uint32),  # w1_vm
        pltpu.VMEM((2, inter // PACK4, hidden), jnp.uint32),  # w2_vm
        pltpu.VMEM((2, geo["rows1"], geo["cols1"]), jnp.uint8),  # w1s_vm
        pltpu.VMEM((2, geo["rows2"], geo["cols2"]), jnp.uint8),  # w2s_vm
        pltpu.VMEM((2, K32Z_CHUNK // K32Z_LANE_TILE, geo["tab_rows"], K32Z_LANE_TILE), jnp.int32),
        pltpu.VMEM((8, max(2 * inter, hidden)), jnp.float32),  # col_vm
    ]
    if rolls:
        scratch.append(pltpu.VMEM((TILE_M, max(2 * inter, hidden)), jnp.float32))  # acc_vm
    scratch.append(pltpu.VMEM((2, TILE_M) + out_row_shape, out_dtype))  # out_vm
    if wire:
        scratch.append(pltpu.VMEM((2, 1, TILE_M), jnp.float32))  # oscl_vm
    scratch += [
        pltpu.SemaphoreType.DMA((2,)),  # lhs_sems
        pltpu.SemaphoreType.DMA((2,)),  # w1_sems
        pltpu.SemaphoreType.DMA((2,)),  # w2_sems
        pltpu.SemaphoreType.DMA((2,)),  # w1s_sems
        pltpu.SemaphoreType.DMA((2,)),  # w2s_sems
        pltpu.SemaphoreType.DMA((2,)),  # out_sems
    ]
    if wire:
        scratch.append(pltpu.SemaphoreType.DMA((2,)))  # osc_sems
    out_shape = jax.ShapeDtypeStruct((m_rows + (TILE_M if fo else 0),) + out_row_shape, out_dtype)
    if wire:
        out_shape = (out_shape, jax.ShapeDtypeStruct((t_max, 1, TILE_M), jnp.float32))
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[any_spec] * 6,
            out_specs=(any_spec, any_spec) if wire else any_spec,
            scratch_shapes=tuple(scratch),
            grid=(),
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * VMEM_FRACTION),
            disable_bounds_checks=True,
        ),
        name=f"k8_k32z_grouped_ffn_{plan}_gu{gate_up_bits}"
        + ("" if mode == "full" else f"_{mode}")
        + ("" if w_split == 1 else f"_ws{w_split}")
        + ("" if not kn else "_k_" + "_".join(sorted(kn))),
    )
    return call


def visit_list(sizes):
    """(visit [48] i32: experts with rows, in expert order, then padding; count [1] i32)."""
    e = jnp.arange(N_EXPERTS, dtype=jnp.int32)
    has = sizes > 0
    key = jnp.where(has, e, N_EXPERTS + e)
    order = jnp.argsort(key).astype(jnp.int32)
    return order, jnp.sum(has.astype(jnp.int32)).reshape(1)


def computation(x, x_scale, w1, w2, w1s, w2s, sizes, starts, plan="decode", mode="full", w_split=1, knobs=""):
    kn = parse_knobs(knobs)
    gate_up_bits = w1s.shape[-1] * 8 // (2 * INTER)
    m = x.shape[0]
    call = build(m, x_scale.shape[0], plan, gate_up_bits, mode, w_split, ",".join(sorted(kn)))
    visit, nvis = visit_list(sizes)
    xin = x
    if "fa" in kn:  # flat rows plus one tile of padding rows (whole-tile activation DMAs)
        xin = jnp.concatenate([x.reshape(m, HIDDEN), jnp.zeros((TILE_M, HIDDEN), x.dtype)])
    res = call(sizes, starts, visit, nvis, xin, x_scale, w1, w2, w1s, w2s)
    if "wire" not in kn:
        return res[:m].reshape(m, LANE_BLOCKS, LANES)
    # K9 wire: rebuild out[r] = bf16(q[r] * s[r]) outside the kernel. Visits run in expert order, so expert e's
    # tiles start at tile_base[e] (exclusive cumsum of its tile counts) and row i of e sits in tile
    # tile_base[e] + i // TILE_M, lane i % TILE_M of the scale rows.
    q, scl = res
    q = q[:m]
    ntile = (sizes + TILE_M - 1) // TILE_M
    tile_base = jnp.cumsum(ntile) - ntile
    r = jnp.arange(m, dtype=jnp.int32)
    e = jnp.minimum(jnp.searchsorted(jnp.cumsum(sizes), r, side="right"), N_EXPERTS - 1).astype(jnp.int32)
    i = r - starts[e]
    s_row = scl[tile_base[e] + i // TILE_M, 0, i % TILE_M]
    out = (q.reshape(m, HIDDEN).astype(jnp.float32) * s_row[:, None]).astype(jnp.bfloat16)
    return out.reshape(m, LANE_BLOCKS, LANES)
