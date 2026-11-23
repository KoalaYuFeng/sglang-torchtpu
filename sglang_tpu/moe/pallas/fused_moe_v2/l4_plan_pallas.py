"""Step 12 P3 (prototype, not wired into the layer): l4_split.plan's decision in one small Pallas kernel.

The same integer arithmetic as l4_split.plan (int32 everywhere; the per-rank loads are integer sums below 2^24,
taken in f32 lane reductions, so exact), from the same plan tables, for one rank `me`. Returns the plan's dict
(the same keys and values). Argmax = the first index of the maximum, as jnp.argmax.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.moe.pallas.fused_moe_v2.host import pow2_shift
from sglang_tpu.moe.pallas.fused_moe_v2.l4_split import ROLE_DONOR, ROLE_HELPER, ROLE_NONE

VECS = ("dl", "dt", "hl", "ht", "hu", "rs_h", "src_row", "dst_row", "last_tile", "src_sub", "dst_sub")
SCALARS = ("role", "is_donor", "is_helper", "donor", "helper", "h", "hg", "a", "m", "s_own")


def _kernel(me_ref, rows_ref, rra_ref, rr_ref, rs_ref, rb_ref, out_ref, *, ep, g, tile_m, shift, tshift,
            partners, kappa, min_rows):
    i32, f32 = jnp.int32, jnp.float32
    n = ep * g
    rows = rows_ref[...]  # [ep, g]
    s_sub = lax.broadcasted_iota(i32, (ep, 1), 0)
    g_lane = lax.broadcasted_iota(i32, (1, g), 1)
    load = jnp.sum(
        jnp.where(rows > 0, jnp.maximum(rows, kappa), 0).astype(f32), axis=1, keepdims=True
    ).astype(i32)  # [ep, 1]
    lmax = jnp.max(load, axis=0, keepdims=True)
    donor = jnp.min(jnp.where(load == lmax, s_sub, ep), axis=0, keepdims=True)  # [1, 1]
    helper = jnp.zeros((1, 1), i32)
    for s, p in enumerate(partners):
        helper = jnp.where(donor == s, p, helper)
    load_d = jnp.sum(jnp.where(s_sub == donor, load, 0).astype(f32), axis=0, keepdims=True).astype(i32)
    load_h = jnp.sum(jnp.where(s_sub == helper, load, 0).astype(f32), axis=0, keepdims=True).astype(i32)
    drow = jnp.sum(jnp.where(s_sub == donor, rows, 0).astype(f32), axis=0, keepdims=True).astype(i32)  # [1, g]
    n_h = jnp.max(drow, axis=1, keepdims=True)
    h = jnp.min(jnp.where(drow == n_h, g_lane, g), axis=1, keepdims=True)
    hg = donor * g + h
    m_star = jnp.right_shift(load_d - load_h, 1)
    a = jnp.right_shift(n_h - m_star + (tile_m - 1), tshift) * tile_m
    a = jnp.minimum(jnp.maximum(a, tile_m), n_h - 2 * tile_m)
    m = n_h - a
    active = (m >= min_rows) & (a >= tile_m) & (helper != donor)
    me = me_ref[0]
    is_donor = active & (donor == me)
    is_helper = active & (helper == me)
    role = jnp.where(is_donor, ROLE_DONOR, jnp.where(is_helper, ROLE_HELPER, ROLE_NONE))
    e_sub = lax.broadcasted_iota(i32, (n, ep), 0)

    def row_hg(ref):  # [1, ep]: the table's row hg
        return jnp.sum(jnp.where(e_sub == hg, ref[...], 0).astype(f32), axis=0, keepdims=True).astype(i32)

    rs_h, rra_h, rr_h = row_hg(rs_ref), row_hg(rra_ref), row_hg(rr_ref)
    dl = jnp.clip(a - rs_h, 0, rra_h)
    dt = jnp.clip(a - rs_h, 0, rr_h)
    hl = rra_h - dl
    ht = rr_h - dt
    start = jnp.maximum(rs_h, a)
    u0 = jnp.right_shift(start, shift)
    hu = jnp.where(hl > 0, jnp.right_shift(rs_h + rra_h - 1, shift) - u0 + 1, 0)
    s_own = jnp.sum(
        jnp.sum(jnp.where(s_sub == me, rows, 0).astype(f32), axis=1, keepdims=True), axis=0, keepdims=True
    ).astype(i32)  # [1, 1]
    # push_rows[d] = recv_base[d, donor, h]: column hg of rb [ep, n], moved to lanes.
    col = jnp.sum(
        jnp.where(lax.broadcasted_iota(i32, (ep, n), 1) == hg, rb_ref[...], 0).astype(f32), axis=1, keepdims=True
    )  # [ep, 1]
    eye = lax.broadcasted_iota(i32, (ep, ep), 0) == lax.broadcasted_iota(i32, (ep, ep), 1)
    push_rows = jnp.sum(jnp.where(eye, col, 0.0), axis=0, keepdims=True).astype(i32)  # [1, ep]
    vecs = dict(
        dl=dl, dt=dt, hl=hl, ht=ht, hu=hu, rs_h=rs_h,
        src_row=s_own + start - a,
        dst_row=push_rows + dl,
        last_tile=jnp.where(hl > 0, jnp.right_shift(rs_h + rra_h - 1 - a, tshift), -1),
        src_sub=u0 - jnp.right_shift(a, shift),
        dst_sub=jnp.right_shift(push_rows, shift) + 2 * hg + u0 - jnp.right_shift(rs_h, shift),
    )
    scal = dict(role=role, is_donor=is_donor.astype(i32), is_helper=is_helper.astype(i32), donor=donor,
                helper=helper, h=h, hg=hg, a=a, m=m, s_own=s_own)
    lane = lax.broadcasted_iota(i32, (1, ep), 1)
    for k, name in enumerate(VECS):
        out_ref[k : k + 1, :] = vecs[name].astype(i32)
    srow = jnp.zeros((1, ep), i32)
    for j, name in enumerate(SCALARS[:ep]):
        srow = jnp.where(lane == j, scal[name], srow)
    out_ref[len(VECS) : len(VECS) + 1, :] = srow
    srow2 = jnp.zeros((1, ep), i32)
    for j, name in enumerate(SCALARS[ep:]):
        srow2 = jnp.where(lane == j, scal[name], srow2)
    out_ref[len(VECS) + 1 : len(VECS) + 2, :] = srow2


def plan_pallas(routing, me, *, partners, e_total, ep, tile_m, mirror_m, kappa, min_rows, interpret=False):
    """l4_split.plan's dict from one Pallas kernel (ep >= 8 lanes hold the scalars)."""
    g = e_total // ep
    assert ep >= 8 and len(SCALARS) <= 2 * ep
    i32 = jnp.int32
    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)
    out = pl.pallas_call(
        functools.partial(
            _kernel, ep=ep, g=g, tile_m=tile_m, shift=pow2_shift(mirror_m, "mirror"),
            tshift=pow2_shift(tile_m, "tile_m"), partners=tuple(partners), kappa=kappa, min_rows=min_rows,
        ),
        in_specs=[smem, vmem, vmem, vmem, vmem, vmem],
        out_specs=vmem,
        out_shape=jax.ShapeDtypeStruct((len(VECS) + 2, ep), i32),
        name=f"moe_v2_l4_plan_e{e_total}_ep{ep}",
        interpret=interpret,
    )(
        jnp.asarray(me, i32).reshape(1),
        routing.expert_rows_aligned.astype(i32).reshape(ep, g),
        routing.run_rows_aligned.astype(i32),
        routing.run_rows.astype(i32),
        routing.run_start_aligned.astype(i32),
        routing.recv_base.astype(i32).reshape(ep, ep * g),
    )
    res = {name: out[k] for k, name in enumerate(VECS)}
    for j, name in enumerate(SCALARS):
        v = out[len(VECS) + j // ep, j % ep]
        res[name] = v.astype(bool) if name in ("is_donor", "is_helper") else v
    return res
