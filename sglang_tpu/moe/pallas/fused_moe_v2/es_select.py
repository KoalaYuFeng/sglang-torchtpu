"""Step 12 P3: the early-ship selector as one small Pallas kernel.

The same X and pick as host.early_ship_tail_us / host.early_ship_pick (the
Step 11 S1 model, read there), computed from the same replicated plan
(routing.run_rows [E, ep] true rows, routing.expert_rows_aligned [E]) in one
kernel launch instead of XLA fusions. host.K32Z_ES_PALLAS selects it in a
mode 2 build (default None: the XLA pick, unchanged).

Layout: per shard s (unrolled), a [g, g] tile with the visited expert h on
lanes and the expert g on sublanes; columns come from rows by a masked
diagonal sum (exact: one nonzero term). The visit-order sums (start, the
shard's compute end) are f32 sums of the same terms as the XLA version, in
the lane reduction's order. The link finishes need, per link l and release
i, the rows released at or after i: a 0 / 1 [n, n] matrix times the per-link
rows [n, L], done on the MXU in bf16 with each row count split into three
bytes (every product and partial sum an integer below 2^24, so exact).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_moe_v2 import host as _host

_LANES = 128


def _link_dests(ep):
    """links[l][s]: the destination shards whose push from s crosses link l."""
    inc = _host.early_ship_link_incidence(ep)
    return tuple(
        tuple(tuple(d for d in range(ep) if inc[l][s][d]) for s in range(ep))
        for l in range(len(inc))
    )


def _kernel(rows_ref, rr_ref, flag_ref, x_ref, rel_ref, w_ref, *, ep, g, tile_m, links):
    f32 = np.float32  # scalar literals (a kernel captures no array constants)
    n = ep * g
    a_us, b_us, w_us = f32(_host.K32Z_ES_A_US), f32(_host.K32Z_ES_B_US), f32(_host.K32Z_ES_W_US)
    sub = lax.broadcasted_iota(jnp.int32, (g, g), 0)  # g (sublanes)
    lane = lax.broadcasted_iota(jnp.int32, (g, g), 1)  # h (lanes)
    eye = sub == lane
    d_lane = lax.broadcasted_iota(jnp.int32, (g, ep), 1)
    l_lane = lax.broadcasted_iota(jnp.int32, (g, _LANES), 1)
    comp_end = None
    for s in range(ep):
        r_row = rows_ref[s : s + 1, :].astype(f32)  # [1, g]
        r_col = jnp.sum(jnp.where(eye, r_row, f32(0)), axis=1, keepdims=True)  # [g, 1]
        live_row, live_col = r_row > 0, r_col > 0
        dur_row = jnp.where(live_row, jnp.maximum(w_us, a_us * r_row), f32(0))
        dur_col = jnp.where(live_col, jnp.maximum(w_us, a_us * r_col), f32(0))
        # before[g, h]: shard s visits h before g (rows descending, then index).
        before = jnp.logical_and(
            jnp.logical_or(
                r_row > r_col, jnp.logical_and(r_row == r_col, lane < sub)
            ),
            live_row,
        )
        start = jnp.sum(jnp.where(before, dur_row, f32(0)), axis=1, keepdims=True)
        pos_col = jnp.sum(jnp.where(before, f32(1), f32(0)), axis=1, keepdims=True)
        pos_row = jnp.sum(jnp.where(eye, pos_col, f32(0)), axis=0, keepdims=True)
        first_row = dur_row * jnp.minimum(r_row, f32(tile_m)) / jnp.maximum(r_row, f32(1))
        nxt = jnp.logical_and(pos_row == pos_col + f32(1), live_row)
        rel = start + dur_col + jnp.sum(
            jnp.where(nxt, first_row, f32(0)), axis=1, keepdims=True
        )
        rel_ref[s * g : (s + 1) * g, :] = rel
        end = jnp.sum(dur_row, axis=1, keepdims=True)  # [1, 1]
        comp_end = end if comp_end is None else jnp.maximum(comp_end, end)
        # The rows each of this shard's experts sends over each link.
        rr_s = rr_ref[s * g : (s + 1) * g, :].astype(f32)  # [g, ep]
        w_s = jnp.zeros((g, _LANES), jnp.float32)
        for l, per_src in enumerate(links):
            dests = per_src[s]
            if not dests:
                continue
            m = d_lane == dests[0]
            for d in dests[1:]:
                m = jnp.logical_or(m, d_lane == d)
            col = jnp.sum(jnp.where(m, rr_s, f32(0)), axis=1, keepdims=True)
            w_s = jnp.where(l_lane == l, col, w_s)
        w_ref[s * g : (s + 1) * g, :] = w_s
    rel_col = rel_ref[...]  # [n, 1]
    eye_n = lax.broadcasted_iota(jnp.int32, (n, n), 0) == lax.broadcasted_iota(
        jnp.int32, (n, n), 1
    )
    rel_row = jnp.sum(jnp.where(eye_n, rel_col, f32(0)), axis=0, keepdims=True)
    # later[i, j]: j released at or after i.
    later = jnp.where(rel_row >= rel_col, f32(1), f32(0)).astype(jnp.bfloat16)
    w = w_ref[...]
    wi = w.astype(jnp.int32)
    suffix = None
    for shift in (16, 8, 0):
        part = ((wi >> shift) & 255).astype(f32).astype(jnp.bfloat16)
        p = jnp.dot(later, part, preferred_element_type=f32).astype(jnp.int32)
        suffix = p if suffix is None else suffix * 256 + p
    fin = jnp.where(w > 0, rel_col + b_us * suffix.astype(f32), -jnp.inf)
    top = jnp.max(jnp.max(fin, axis=1, keepdims=True), axis=0, keepdims=True)
    x = top - comp_end  # [1, 1]
    x_ref[...] = x
    on = x > f32(_host.K32Z_ES_THETA_US)
    lane0 = lax.broadcasted_iota(jnp.int32, (1, ep), 1) == 0
    flag_ref[...] = jnp.where(jnp.logical_and(lane0, on), 1, 0).astype(jnp.int32)


def early_ship_select_pallas(
    run_rows, expert_rows_aligned, *, ep, g_local, tile_m, interpret=False
):
    """(flag row [1, ep] i32 = host.early_ship_flag_row(host.early_ship_pick(..)),
    X [1, 1] f32 = host.early_ship_tail_us(..)) in one Pallas kernel."""
    n = ep * g_local
    links = _link_dests(ep)
    assert len(links) <= _LANES
    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    flag, x = pl.pallas_call(
        functools.partial(_kernel, ep=ep, g=g_local, tile_m=tile_m, links=links),
        in_specs=[vmem, vmem],
        out_specs=[vmem, vmem],
        out_shape=[
            jax.ShapeDtypeStruct((1, ep), jnp.int32),
            jax.ShapeDtypeStruct((1, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((n, 1), jnp.float32),
            pltpu.VMEM((n, _LANES), jnp.float32),
        ],
        name=f"moe_v2_es_select_e{n}_ep{ep}",
        interpret=interpret,
    )(
        expert_rows_aligned.reshape(ep, g_local).astype(jnp.int32),
        run_rows.astype(jnp.int32),
    )
    return flag, x
