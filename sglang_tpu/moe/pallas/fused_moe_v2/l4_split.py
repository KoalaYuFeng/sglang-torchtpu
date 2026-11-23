# SPDX-License-Identifier: Apache-2.0
"""Step 10 L4: the hot expert's rows split with the sibling TensorCore.

All-gather builds only (every rank holds every token and every rank's routing
counts), decided at run time from the routing plan, the same answer on every
rank:

- load per rank = sum over its hit experts of max(kappa, aligned rows);
- the busiest rank (the donor) and the other TensorCore of its chip (the
  helper, `sibling_partners`);
- h = the donor's most-rows local expert, n_h its aligned rows; the donor
  keeps rows [0, a) of h and the helper computes rows [a, n_h), with
  a = round_up(n_h - (load_donor - load_helper) / 2, tile_m), so every tail
  tile is one of the donor's own tiles (same rows, same last partial tile)
  and a is also a scale-mirror unit boundary;
- no split when the helper's share is under `min_rows`.

The donor's tables shrink h to its head. The helper runs the tail as a 49th
local expert (index g_local): its slab rows follow its own experts, so the
layer's existing scatters fill the tail's token-gather rows and activation
scales once `slab_row` maps h's tail pairs there; the kernel (l4=True) streams
its weights from the hot buffers the donor's remote DMA fills, commits its
tiles to the contribution slab after its own experts', and ships each of h's
runs' tail share to the arrival rows and mirror sublanes the donor would have
used. The combine is untouched; each row is computed whole by one core.
"""

import jax.numpy as jnp

from sglang_tpu.moe.pallas.fused_moe_v2.host import (
    COUNT_RECV_ALIGNED_ROWS,
    COUNT_RECV_MIRROR,
    COUNT_RECV_ROWS,
    COUNT_SELF_MIRROR,
    COUNT_SEND_ALIGNED_ROWS,
    COUNT_SEND_MIRROR,
    COUNT_SEND_ROWS,
    COUNT_VISITS,
    ROWBLK_SHIFT,
    expert_visit_list,
    pow2_shift,
)

ROLE_NONE, ROLE_DONOR, ROLE_HELPER = 0, 1, 2
# l4_table rows 0 .. ep - 1, one per destination d of the helper's tail share
# of the donor's run (h, d): contribution row it starts at, arrival row,
# true rows (remote), aligned rows (own destination), mirror unit offset from
# the tail's contribution mirror base, arrival mirror sublane, units, and the
# tail tile its share ends in (-1: none; mode 4 ships it two tiles later).
L4_SRC_ROW, L4_DST_ROW, L4_TRUE, L4_ALIGNED, L4_SRC_SUB, L4_DST_SUB, L4_NSUB, L4_LAST_TILE = range(8)
# l4_table row ep: this rank's role, its sibling in the split, the donor's hot
# local expert, the split row, the tail's rows, the helper's own-destination
# tail rows (added to its self-hop drain).
L4_ROLE, L4_PARTNER, L4_H, L4_A, L4_M, L4_SELF_ROWS = range(6)
L4_COLS = 8


def sibling_partners(mesh):
    """Mesh index -> mesh index of the other TensorCore on its chip, or None
    when the devices do not pair up (static: device coordinates)."""
    devs = list(mesh.devices.reshape(-1))
    keys = []
    for d in devs:
        coords = getattr(d, "coords", None)
        core = getattr(d, "core_on_chip", None)
        if coords is None or core is None:
            return None
        keys.append(tuple(int(c) for c in coords))
    partners = []
    for i, k in enumerate(keys):
        js = [j for j, k2 in enumerate(keys) if k2 == k and j != i]
        if len(js) != 1:
            return None
        partners.append(js[0])
    return tuple(partners)


def plan(routing, me, *, partners, e_total, ep, tile_m, mirror_m, kappa, min_rows):
    """The split, and this rank's share of it, as a dict of traced values."""
    i32 = jnp.int32
    g = e_total // ep
    shift = pow2_shift(mirror_m, "the mirror's lane count")
    tshift = pow2_shift(tile_m, "tile_m")
    rra = routing.run_rows_aligned.astype(i32)  # [E, ep]
    rr = routing.run_rows.astype(i32)
    rs = routing.run_start_aligned.astype(i32)
    recv_base = routing.recv_base.astype(i32)  # [d, s, g] rows
    n_e = routing.expert_rows_aligned.astype(i32)  # [E]
    rows_rg = n_e.reshape(ep, g)
    load = jnp.where(rows_rg > 0, jnp.maximum(rows_rg, kappa), 0).sum(axis=1)
    donor = jnp.argmax(load).astype(i32)
    helper = jnp.asarray(partners, i32)[donor]
    h = jnp.argmax(rows_rg[donor]).astype(i32)
    hg = donor * g + h
    n_h = n_e[hg]
    m_star = (load[donor] - load[helper]) // 2
    a = ((n_h - m_star + tile_m - 1) // tile_m) * tile_m
    a = jnp.minimum(jnp.maximum(a, tile_m), n_h - 2 * tile_m)
    m = n_h - a
    active = (m >= min_rows) & (a >= tile_m) & (helper != donor)
    is_donor = active & (me == donor)
    is_helper = active & (me == helper)
    role = jnp.where(is_donor, ROLE_DONOR, jnp.where(is_helper, ROLE_HELPER, ROLE_NONE))
    # h's runs, one per destination, and their split at a (donor coordinates).
    rs_h, rra_h, rr_h = rs[hg], rra[hg], rr[hg]
    dl = jnp.clip(a - rs_h, 0, rra_h)  # donor's aligned rows of run d
    dt = jnp.clip(a - rs_h, 0, rr_h)  # donor's true rows
    hl = rra_h - dl  # helper's aligned rows
    ht = rr_h - dt  # helper's true rows
    start = jnp.maximum(rs_h, a)  # helper's first row of run d
    u0 = jnp.right_shift(start, shift)
    hu = jnp.where(
        hl > 0, jnp.right_shift(rs_h + rra_h - 1, shift) - u0 + 1, 0
    )  # helper's mirror units of run d
    s_own = rows_rg[me].sum()  # this rank's own slab rows
    push_rows = recv_base[:, donor, h]  # arrival row each run starts at
    return dict(
        role=role.astype(i32),
        is_donor=is_donor,
        is_helper=is_helper,
        donor=donor,
        helper=helper,
        h=h,
        hg=hg,
        a=a,
        m=m,
        dl=dl,
        dt=dt,
        hl=hl,
        ht=ht,
        hu=hu,
        rs_h=rs_h,
        s_own=s_own,
        src_row=s_own + start - a,
        dst_row=push_rows + dl,
        last_tile=jnp.where(hl > 0, jnp.right_shift(rs_h + rra_h - 1 - a, tshift), -1),
        src_sub=u0 - jnp.right_shift(a, shift),
        dst_sub=jnp.right_shift(push_rows, shift)
        + 2 * hg
        + u0
        - jnp.right_shift(rs_h, shift),
    )


def slab_row(p, slab_row_local, expert_ids, *, ep, per_owner):
    """The helper's slab rows: h's pairs whose donor row is in the tail map
    to s_own + row - a; every other rank's and pair's row is unchanged.
    `expert_ids` is the gathered [T * topk] expert of each pair, in the
    plan's flat (token, k) order, owners in blocks of `per_owner` pairs."""
    i32 = jnp.int32
    is_h = (expert_ids == p["hg"]).astype(i32)
    inc = jnp.cumsum(is_h)  # inclusive, over every owner
    per = inc.reshape(ep, per_owner)
    # h's pairs in the owners before d: the inclusive count at owner d - 1's end.
    owner_start = jnp.concatenate([jnp.zeros((1,), i32), per[:-1, -1]])
    rank = (per - is_h.reshape(ep, per_owner)) - owner_start[:, None]
    donor_row = (rank + p["rs_h"][:, None]).reshape(-1)
    take = p["is_helper"] & (is_h > 0) & (donor_row >= p["a"])
    return jnp.where(take, p["s_own"] + donor_row - p["a"], slab_row_local)


def donor_rows(p, expert_rows):
    """The donor's own expert rows with h cut to its head [0, a)."""
    sel = (jnp.arange(expert_rows.shape[0]) == p["h"]) & p["is_donor"]
    return jnp.where(sel, p["a"], expert_rows).astype(jnp.int32)


def slabs(p, expert_rows, slab_base):
    """(rows, base) [G + 1]: the helper's tail as local expert G."""
    tail_rows = jnp.where(p["is_helper"], p["m"], 0)
    tail_base = jnp.where(p["is_helper"], p["s_own"], 0)
    return (
        jnp.concatenate([expert_rows, tail_rows[None]]).astype(jnp.int32),
        jnp.concatenate([slab_base, tail_base[None]]).astype(jnp.int32),
    )


def visit_list(p, expert_rows, tail_pos=None):
    """[G + 1]: the own visit list (most rows first), with the helper's tail
    at visit `tail_pos` (None: after its own experts)."""
    g = expert_rows.shape[0]
    visit, n_own = expert_visit_list(expert_rows, g)
    n_own = n_own[0]
    pos = n_own if tail_pos is None else jnp.minimum(jnp.int32(tail_pos), n_own)
    idx = jnp.arange(g + 1, dtype=jnp.int32)
    padded = jnp.concatenate([visit, jnp.zeros((1,), jnp.int32)])
    # The helper's own visits from `pos` on move one step later.
    shifted = jnp.concatenate([jnp.zeros((1,), jnp.int32), visit])
    own = jnp.where(p["is_helper"] & (idx > pos), shifted, padded)
    tail = p["is_helper"] & (idx == pos)
    return jnp.where(tail, jnp.int32(g), own).astype(jnp.int32)


def kernel_tables(p, tables, *, ep):
    """The five kernel transport tables (commit start, commit length and
    contribution offset in blocks, push destination in blocks, true rows) with
    the donor's h cut at a and a row G for the helper's tail: one run at d = 0
    of m rows at contribution row s_own (its own experts' total), so each tail
    tile commits once; the tail's own pushes come from l4_table."""
    commit_start, commit_len, contrib_off, push_dst, true_rows = tables
    g = commit_len.shape[0]
    hrow = ((jnp.arange(g) == p["h"]) & p["is_donor"])[:, None]
    commit_len = jnp.where(hrow, jnp.right_shift(p["dl"], ROWBLK_SHIFT)[None, :], commit_len)
    true_rows = jnp.where(hrow, p["dt"][None, :], true_rows)
    d0 = (jnp.arange(ep) == 0) & p["is_helper"]
    zero = jnp.zeros((1, ep), jnp.int32)

    def grow(t, row):
        return jnp.concatenate([t.astype(jnp.int32), row.astype(jnp.int32)], axis=0)

    return (
        grow(commit_start, zero),
        grow(commit_len, jnp.where(d0, jnp.right_shift(p["m"], ROWBLK_SHIFT), 0)[None]),
        grow(contrib_off, jnp.where(d0, jnp.right_shift(p["s_own"], ROWBLK_SHIFT), 0)[None]),
        grow(push_dst, zero),
        grow(true_rows, zero),
    )


def counts(p, count_table, *, ep):
    """The kernel's count table with the split's transport moved between the
    donor and the helper; every other destination receives the same bytes."""
    i32 = jnp.int32
    lanes = jnp.arange(ep)
    not_d = (lanes != p["donor"]).astype(i32)
    not_h = (lanes != p["helper"]).astype(i32)
    dn, hp = p["is_donor"], p["is_helper"]

    def moved(x):
        """(send delta, own-destination share) of a per-dest quantity."""
        return (
            jnp.where(dn, -(x * not_d).sum(), jnp.where(hp, (x * not_h).sum(), 0)),
            jnp.where(dn, x[p["donor"]], jnp.where(hp, -x[p["helper"]], 0)),
        )

    send_rows, recv_rows = moved(p["ht"])
    send_al, recv_al = moved(p["hl"])
    send_mir, recv_mir = moved(p["hu"])
    delta = jnp.zeros((count_table.shape[0],), i32)
    delta = delta.at[COUNT_SEND_ROWS].set(send_rows)
    delta = delta.at[COUNT_RECV_ROWS].set(recv_rows)
    delta = delta.at[COUNT_SEND_ALIGNED_ROWS].set(send_al)
    delta = delta.at[COUNT_RECV_ALIGNED_ROWS].set(recv_al)
    delta = delta.at[COUNT_VISITS].set(jnp.where(hp, 1, 0))
    delta = delta.at[COUNT_SEND_MIRROR].set(send_mir)
    delta = delta.at[COUNT_RECV_MIRROR].set(recv_mir)
    delta = delta.at[COUNT_SELF_MIRROR].set(-recv_mir)
    return count_table.astype(i32).at[:, 0].add(delta)


def l4_table(p, *, ep):
    """[ep + 1, L4_COLS] i32, the kernel's L4 scalar table."""
    i32 = jnp.int32
    hp = p["is_helper"]
    per_dest = jnp.stack(
        [
            p["src_row"],
            p["dst_row"],
            p["ht"],
            p["hl"],
            p["src_sub"],
            p["dst_sub"],
            p["hu"],
            p["last_tile"],
        ],
        axis=1,
    )
    per_dest = jnp.where(hp, per_dest, 0).astype(i32)
    last = jnp.stack(
        [
            p["role"],
            jnp.where(p["is_donor"], p["helper"], p["donor"]),
            p["h"],
            p["a"],
            p["m"],
            jnp.where(hp, p["hl"][p["helper"]], 0),
            jnp.int32(0),
            jnp.int32(0),
        ]
    ).astype(i32)[None]
    return jnp.concatenate([per_dest, last], axis=0)


__all__ = [
    "L4_COLS",
    "ROLE_DONOR",
    "ROLE_HELPER",
    "ROLE_NONE",
    "counts",
    "donor_rows",
    "kernel_tables",
    "l4_table",
    "plan",
    "sibling_partners",
    "slab_row",
    "slabs",
    "visit_list",
]
