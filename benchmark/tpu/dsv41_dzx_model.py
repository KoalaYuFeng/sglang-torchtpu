#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 7 K2d: NumPy host model of the per-rank dispatch dedup (DZX) tables.

Written from the design (results/step7/k2/DESIGN.md §3), not from the device
code, so the L0 tests and the K2e pod run can compare the device tables to
it bitwise and replay a whole layer's data movement:

* every rank's sender tables (owner flags, the owned-first sort, each pair's
  run position idx, the payload's pair and count words);
* the visit order (plan_kernel._tables': most aligned rows first, ties by
  index, then light_prepush / visit_order), the deduplicated runs (not a
  SparseCore position, destination under the SRC cap), the rows each run
  lands, the dispatch segment counts and the receive counts;
* every destination's x tables (landed | dup base << 16), slab rows, SRC;
* a replay: the dispatch fills each destination's slab with the landed rows,
  the kernel's tile loader (landed stretches + one row copy per duplicate)
  fills every tile, and every true row of every tile must hold the token the
  default (D) layout puts there -- the same per-expert token multiset, each
  pair's return row at its run position -- with every owner landed before
  the visit that copies it (a SparseCore position or an earlier visit);
* counts: remote rows on the wire per rank pair (D vs DZX), duplicate row
  copies and tile copies per rank, bytes per directed ICI link under the
  K0 flow model (8 rank-pair flows per directed link, 5,120 B rows).

    python3 dsv41_dzx_model.py --tokens 512,1024 --routings uniform,skew,hot

prints those counts (and checks every invariant) for the K2e report.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

ROWBLK = 2
SHIFT = 16
ROW_BYTES = 5120  # one FP8 token row of hidden 5120
FLOWS_PER_LINK = 8  # K0: rank-pair flows per directed link (x-first routing)


def aligned(n):
    return (np.asarray(n) + ROWBLK - 1) // ROWBLK * ROWBLK


def routing_block(t, topk, max_block=256):
    """host.routing_block: the widest power-of-two block (at most 256)
    dividing the shard's t * topk routed pairs."""
    b = max_block
    while (t * topk) % b:
        b //= 2
    return b


# --------------------------------------------------------------------------- #
# Routings
# --------------------------------------------------------------------------- #
def routing(kind, seed, *, ep=8, g=48, t_local=512, topk=6, empty=0):
    """Global mesh-order expert ids [ep (source), t_local, topk], distinct per
    token. uniform: every expert alike; skew: Zipf-like popularity 0.8 with
    `empty` experts per rank getting nothing; hot: Zipf 1.2."""
    rng = np.random.default_rng(seed)
    e_total = ep * g
    if kind == "uniform":
        p = np.full(e_total, 1.0 / e_total)
    else:
        a = 0.8 if kind == "skew" else 1.2
        w = 1.0 / np.arange(1, e_total + 1) ** a
        rng.shuffle(w)
        for d in range(ep):
            w[d * g + rng.choice(g, empty, replace=False)] = 0.0
        p = w / w.sum()
    ids = np.empty((ep, t_local, topk), np.int64)
    for s in range(ep):
        for t in range(t_local):
            ids[s, t] = rng.choice(e_total, topk, replace=False, p=p)
    return ids


def pair_counts(ids, ep, g):
    """r3 [s, d, e]: pairs source s routes to destination d's expert e."""
    r3 = np.zeros((ep, ep, g), np.int64)
    for s in range(ep):
        valid = ids[s] < ep * g
        np.add.at(r3[s], (ids[s][valid] // g, ids[s][valid] % g), 1)
    return r3


# --------------------------------------------------------------------------- #
# Visit order (plan_kernel._tables)
# --------------------------------------------------------------------------- #
def visit_ranks(r3, n_pre, *, light_prepush=2, visit_order=0):
    """(rank [d, e] = visit position of local expert e at destination d,
    n_visit [d]). Positions past n_visit hold the experts without rows."""
    ep, _, g = r3.shape
    ea = aligned(r3).sum(axis=0)
    ranks = np.zeros((ep, g), np.int64)
    n_visits = np.zeros(ep, np.int64)
    for d in range(ep):
        order = sorted(range(g), key=lambda e: (-ea[d, e], e))
        rank = np.empty(g, np.int64)
        rank[order] = np.arange(g)
        n_visit = int((ea[d] > 0).sum())
        n_lp = min(n_visit, n_pre)
        m = n_visit - n_lp
        light = (rank >= m) & (rank < n_visit)
        if visit_order == 1:
            tc = rank < m
            by_id = np.cumsum(tc) - tc
            rank = np.where(light, n_visit - 1 - rank, np.where(tc, by_id + n_lp, rank))
        elif light_prepush:
            rest = rank
            if light_prepush == 2:
                half = m >> 1
                rest = np.where(rank & 1, half - 1 - (rank >> 1), half + (rank >> 1))
            rank = np.where(
                light, n_visit - 1 - rank, np.where(rank < m, rest + n_lp, rank)
            )
        assert sorted(rank) == list(range(g))
        ranks[d], n_visits[d] = rank, n_visit
    return ranks, n_visits


# --------------------------------------------------------------------------- #
# Sender (one source rank)
# --------------------------------------------------------------------------- #
def owner_flags(ids, me, *, g, e_total, local=False, vp=None):
    """[t, K]: True where the slot's row is sent. Owner on destination d =
    the least id (vp: least visit position) among the token's slots on d;
    own-rank slots always unless `local`; unrouted slots count as owned."""
    t, K = ids.shape
    owned = np.ones((t, K), bool)
    key = ids if vp is None else vp
    for i in range(t):
        for k in range(K):
            if ids[i, k] >= e_total:
                continue
            d = ids[i, k] // g
            if d == me and not local:
                continue
            peers = [j for j in range(K) if ids[i, j] < e_total and ids[i, j] // g == d]
            owned[i, k] = key[i, k] == min(key[i, j] for j in peers)
    return owned


def sender(ids, me, *, g, e_total, block, dedup=True, local=False, vp=None):
    """One source rank's tables (dict): key [t*K], tok [t*K] (the token of
    each sorted row; rows past the last routed pair take t - 1), seg [E]
    (each expert's first sorted row), n / own [E], idx [t, K] (position in
    its (destination, expert) run; unrouted pairs: rank among unrouted pairs
    in their routing block), pair words, count words."""
    t, K = ids.shape
    valid = ids < e_total
    if dedup:
        owned = owner_flags(ids, me, g=g, e_total=e_total, local=local, vp=vp)
        dup = valid & ~owned
        key = np.where(valid, 2 * ids + dup, 2 * e_total)
        nb = 2 * e_total
    else:
        key = np.where(valid, ids, e_total)
        nb = e_total
    flat = key.reshape(-1)
    order = np.argsort(flat, kind="stable")  # tokens ascending within a key
    cnt = np.bincount(flat, minlength=nb + 1)[:nb]
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    n_valid = int(cnt.sum())
    tok = np.where(np.arange(t * K) < n_valid, order // K, t - 1)
    sorted_pos = np.empty(t * K, np.int64)
    sorted_pos[order] = np.arange(t * K)
    if dedup:
        seg, n, own = start[0::2], cnt[0::2] + cnt[1::2], cnt[0::2]
    else:
        seg, n, own = start, cnt, cnt
    eflat = ids.reshape(-1)
    idx = np.zeros(t * K, np.int64)
    vflat = valid.reshape(-1)
    idx[vflat] = sorted_pos[vflat] - seg[eflat[vflat]]
    # Unrouted pairs: rank among the block's unrouted pairs (the device's
    # block lookup finds no bin, so only the within-block rank remains).
    for b0 in range(0, t * K, block):
        r = 0
        for j in range(b0, b0 + block):
            if not vflat[j]:
                idx[j] = r
                r += 1
    idx = idx.reshape(t, K)
    words = (n | (own << SHIFT)) if dedup else n
    return dict(
        key=flat,
        tok=tok,
        seg=seg,
        n=n,
        own=own,
        idx=idx,
        sort_counts=cnt,
        pair_words=(ids | (idx << SHIFT)).reshape(-1),
        count_words=words,
    )


# --------------------------------------------------------------------------- #
# The whole layer
# --------------------------------------------------------------------------- #
def build(
    ids, *, g, n_pre, tile_m, cap=None, dedup=True, local=False,
    owner_visit=False, light_prepush=2, visit_order=0, block=None, x_npre=None,
):  # fmt: skip
    """Every rank's tables for one call. ids [ep, t, K] mesh-order ids.

    dedup=False models x_tables alone (D's layout, idx = token-order rank).
    The order is visit_order=1's under dedup with the lowest-id owner, the
    bundle's (light_prepush) under owner_visit. `x_npre` (default n_pre) is
    the SparseCore set the dedup leaves whole; n_pre the dispatch's (g under
    disp_off). `block` (default: the device's routing_block) is the routing
    block the unrouted pairs' run positions are ranked within."""
    ep, t, K = ids.shape
    block = routing_block(t, K) if block is None else block
    e_total = ep * g
    cap = ep * t if cap is None else cap
    x_npre = n_pre if x_npre is None else x_npre
    if dedup and not owner_visit:
        visit_order = 1
    r3 = pair_counts(ids, ep, g)
    rank, n_visit = visit_ranks(
        r3, n_pre, light_prepush=light_prepush, visit_order=visit_order
    )
    senders = []
    for s in range(ep):
        vp = None
        if dedup and owner_visit:
            v = ids[s]
            vp = np.where(v < e_total, rank.reshape(-1)[np.minimum(v, e_total - 1)], 0)
        senders.append(
            sender(
                ids[s],
                s,
                g=g,
                e_total=e_total,
                block=block,
                dedup=dedup,
                local=local,
                vp=vp,
            )
        )
    o3 = np.stack([sd["own"].reshape(ep, g) for sd in senders])
    if dedup:
        xsc = rank < x_npre  # [d, e]
        dsum = ((r3 - o3) * ~xsc[None]).sum(axis=(0, 2))  # [d]
        whole = xsc | (dsum > cap)[:, None]
        landed = np.where(whole[None], r3, o3)
    else:
        xsc = rank < x_npre
        dsum = np.zeros(ep, np.int64)
        landed = r3.copy()
    visit = np.argsort(rank, axis=1)  # [d, v] expert at position v
    ranks_out = []
    for me in range(ep):
        r_dme = r3[:, me, :]  # [s, e]
        l_dme = landed[:, me, :]
        al = aligned(r_dme)
        run_start = np.cumsum(al, axis=0) - al  # [s, e]
        expert_rows = al.sum(axis=0)
        slab_base = np.cumsum(expert_rows) - expert_rows
        run_base = slab_base[None, :] + run_start  # [s, e]
        dups = r_dme - l_dme
        flat = dups.T.reshape(-1)  # (e, s) order, expert-major
        dbase = (np.cumsum(flat) - flat).reshape(g, ep).T  # [s, e]
        xg = l_dme | (dbase << SHIFT)
        # Sender view of me: dispatch counts by (visit position, destination).
        cnt = np.zeros((g, ep), np.int64)
        for d in range(ep):
            for v in range(n_visit[d]):
                cnt[v, d] = landed[me, d, visit[d, v]]
        recv = l_dme.sum(axis=0)
        remote = cnt * (np.arange(ep)[None, :] != me)
        n_remote = int(remote[n_pre:].sum())
        my_vpos = rank[me]
        rows_at_v = np.zeros(g, np.int64)
        for v in range(n_visit[me]):
            rows_at_v[v] = recv[visit[me, v]]
        vv = np.arange(g)
        pre, odd = vv < n_pre, (vv & 1) == 1
        counts_sc = np.array(
            [
                rows_at_v[pre & ~odd].sum(),
                rows_at_v[pre & odd].sum(),
                remote.sum(axis=1)[pre & ~odd].sum(),
                remote.sum(axis=1)[pre & odd].sum(),
                0,
                0,
                0,
                0,
            ]
        )
        meta = np.concatenate([np.zeros(g + 1, np.int64), recv, [n_remote, cnt.sum()]])
        # Receiver: slab rows and SRC.
        stride_off = -1
        slab_row = np.full((ep, t, K), stride_off, np.int64)
        src = np.zeros(cap, np.int64)
        src_set = np.zeros(cap, bool)
        for s in range(ep):
            sid, sidx = ids[s], senders[s]["idx"]
            on = (sid >= me * g) & (sid < (me + 1) * g)
            el = np.where(on, sid - me * g, 0)
            new_row = run_base[s, el] + sidx
            slab_row[s] = np.where(on, new_row, stride_off)
            if not dedup:
                continue
            for i in range(t):
                ks = np.nonzero(on[i])[0]
                if not len(ks):
                    continue
                key = sid[i, ks] if not owner_visit else my_vpos[el[i, ks]]
                k_own = ks[np.argmin(key)]
                for k in ks:
                    e = el[i, k]
                    if sidx[i, k] >= l_dme[s, e]:
                        w = dbase[s, e] + sidx[i, k] - l_dme[s, e]
                        assert not src_set[w]
                        src[w], src_set[w] = new_row[i, k_own], True
        assert src_set.sum() == dups.sum()
        ranks_out.append(
            dict(
                run_base=run_base,
                run_start=run_start,
                expert_rows=expert_rows,
                slab_base=slab_base,
                landed=l_dme,
                dbase=dbase,
                x_g=xg,
                x_k=xg.T,
                cnt=cnt,
                recv=recv,
                meta=meta,
                counts_sc=counts_sc,
                slab_row=slab_row,
                src=src,
                n_dup=int(dups.sum()),
            )
        )
    return dict(
        ids=ids,
        g=g,
        ep=ep,
        t=t,
        K=K,
        n_pre=n_pre,
        x_npre=x_npre,
        tile_m=tile_m,
        cap=cap,
        dedup=dedup,
        local=local,
        owner_visit=owner_visit,
        r3=r3,
        o3=o3,
        landed=landed,
        xsc=xsc,
        overflow=dsum > cap,
        rank=rank,
        n_visit=n_visit,
        visit=visit,
        senders=senders,
        ranks=ranks_out,
    )


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def tile_copies(rk, e, t0, live, tile_m, ep):
    """The kernel's tile loader (x_issue_tile) for tile rows [t0, t0 + live)
    of expert e: [(kind, src slab row, tile row, rows)], kind 'stretch' or
    'dup'."""
    base = rk["slab_base"][e]
    n_dup = int((rk["run_true"][:, e] - rk["landed"][:, e]).sum())
    lo, hi = t0, t0 + live
    if n_dup == 0:
        return [("stretch", base + lo, 0, live)]
    out = []

    def stretch(a, b):
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append(("stretch", base + a, a - lo, b - a))

    prev = 0
    for s in range(ep):
        run = rk["run_start"][s, e]
        own_end = run + rk["landed"][s, e]
        stretch(prev, own_end)
        prev = run + rk["run_true"][s, e]
        for r in range(max(own_end, lo), min(prev, hi)):
            w = rk["dbase"][s, e] + r - own_end
            out.append(("dup", rk["src"][w], r - lo, 1))
    stretch(prev, hi)
    return out


def replay(case):
    """Dispatch, tile loads and return positions of every rank; asserts
    every invariant and returns counts."""
    ep, g, t, K = case["ep"], case["g"], case["t"], case["K"]
    tile_m, e_total = case["tile_m"], ep * g
    senders, landed, r3 = case["senders"], case["landed"], case["r3"]
    stats = dict(tile_copies=0, dup_copies=0, tiles=0, tiles_with_dups=0)
    for me in range(ep):
        rk = case["ranks"][me]
        rk["run_true"] = r3[:, me, :]
        rows_alloc = int(rk["expert_rows"].sum())
        slab = np.full((rows_alloc, 2), -1, np.int64)  # (source, token)
        written = np.zeros(rows_alloc, bool)
        # Dispatch: each run's first `landed` sorted rows of its source.
        for s in range(ep):
            sd = senders[s]
            for e in range(g):
                n_land = landed[s, me, e]
                ge = me * g + e
                for j in range(n_land):
                    row = rk["run_base"][s, e] + j
                    assert not written[row]
                    slab[row] = (s, sd["tok"][sd["seg"][ge] + j])
                    written[row] = True
        # Tile loads, in visit order: every true row holds its expected token.
        pos = case["rank"][me]
        done = np.zeros(g, bool)
        for v in range(case["n_visit"][me]):
            e = case["visit"][me, v]
            rows = rk["expert_rows"][e]
            for t0 in range(0, rows, tile_m):
                live = min(rows - t0, tile_m)
                tile = np.full((live, 2), -2, np.int64)
                covered = np.zeros(live, np.int64)
                copies = tile_copies(rk, e, t0, live, tile_m, ep)
                stats["tiles"] += 1
                stats["tile_copies"] += len(copies)
                n_d = sum(1 for c in copies if c[0] == "dup")
                stats["dup_copies"] += n_d
                stats["tiles_with_dups"] += n_d > 0
                for kind, src_row, trow, n in copies:
                    covered[trow : trow + n] += 1
                    if kind == "dup":
                        assert written[src_row], "duplicate copied from an unlanded row"
                        # Its owner: a SparseCore position or visited earlier.
                        oe = int(np.searchsorted(rk["slab_base"], src_row, "right") - 1)
                        assert case["xsc"][me, oe] or pos[oe] < pos[e] or done[oe], (
                            "owner visited after its duplicate"
                        )
                        assert landed_row(rk, oe, src_row)
                    tile[trow : trow + n] = slab[src_row : src_row + n]
                assert (covered == 1).all(), "tile copies do not partition the tile"
                for s in range(ep):
                    run = rk["run_start"][s, e]
                    sd = senders[s]
                    for j in range(r3[s, me, e]):
                        r = run + j - t0
                        if 0 <= r < live:
                            want = (s, sd["tok"][sd["seg"][me * g + e] + j])
                            assert tuple(tile[r]) == want, (me, e, s, j, tile[r], want)
            done[e] = True
        # Return positions: pair (s, t, k)'s run position indexes its own
        # token in its source's sorted rows, so (with the tile check above)
        # the result row it reads back holds token t of source s.
        for s in range(ep):
            sd = senders[s]
            sid, sidx = case["ids"][s], sd["idx"]
            for i in range(t):
                for k in range(K):
                    if sid[i, k] >= e_total or sid[i, k] // g != me:
                        continue
                    e = sid[i, k] % g
                    assert sidx[i, k] < r3[s, me, e]
                    assert sd["tok"][sd["seg"][sid[i, k]] + sidx[i, k]] == i
                    assert rk["slab_row"][s, i, k] == rk["run_base"][s, e] + sidx[i, k]
    return stats


def landed_row(rk, e, row):
    """Whether slab row `row` (in expert e's region) is a landed row."""
    rel = row - rk["slab_base"][e]
    for s in range(rk["run_start"].shape[0]):
        run = rk["run_start"][s, e]
        if run <= rel < run + rk["landed"][s, e]:
            return True
    return False


# --------------------------------------------------------------------------- #
# Counts for the report
# --------------------------------------------------------------------------- #
def link_counts(case):
    """Remote dispatch rows per rank pair (mean over ordered pairs s != d),
    D vs this case, and bytes per directed link (K0 flow model: dispatch +
    return, 8 flows per directed link)."""
    ep = case["ep"]
    off = ~np.eye(ep, dtype=bool)
    d_rows = case["r3"].sum(axis=2)[off].mean()
    x_rows = case["landed"].sum(axis=2)[off].mean()
    per_link = lambda rows: FLOWS_PER_LINK * rows * ROW_BYTES  # noqa: E731
    return dict(
        rows_per_pair_D=float(d_rows),
        rows_per_pair_X=float(x_rows),
        ratio=float(x_rows / d_rows),
        dispatch_MB_per_link_D=per_link(d_rows) / 1e6,
        dispatch_MB_per_link_X=per_link(x_rows) / 1e6,
        total_MB_per_link_D=2 * per_link(d_rows) / 1e6,
        total_MB_per_link_X=(per_link(x_rows) + per_link(d_rows)) / 1e6,
        busiest_send_ratio=float(
            max(case["landed"][s][off[s]].sum() for s in range(ep))
            / max(case["r3"][s][off[s]].sum() for s in range(ep))
        ),
    )


def dma_counts(case):
    """Per rank (max and mean): duplicate rows gathered, remote dispatch
    segments the TensorCore issues (non-empty runs at positions >= n_pre)."""
    ep, g, n_pre = case["ep"], case["g"], case["n_pre"]
    dups = [rk["n_dup"] for rk in case["ranks"]]
    segs = []
    for me in range(ep):
        cnt = case["ranks"][me]["cnt"]
        remote = np.arange(ep)[None, :] != me
        segs.append(int(((cnt[n_pre:] > 0) & remote).sum()))
    return dict(
        dup_rows_max=max(dups),
        dup_rows_mean=float(np.mean(dups)),
        tc_remote_segments_max=max(segs),
        overflow_ranks=int(case["overflow"].sum()),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokens", default="512,1024")
    p.add_argument("--routings", default="uniform,skew,hot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile-m", type=int, default=256)
    p.add_argument("--n-pre", type=int, default=2)
    p.add_argument("--no-replay", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rows = []
    for t in (int(x) for x in args.tokens.split(",")):
        for kind in args.routings.split(","):
            ids = routing(
                kind, args.seed, t_local=t, empty=3 if kind != "uniform" else 0
            )
            for mode in ("D", "DZt", "DZX", "DZX_local", "DZX_visit"):
                case = build(
                    ids,
                    g=48,
                    n_pre=args.n_pre,
                    tile_m=args.tile_m,
                    dedup=mode.startswith("DZX"),
                    local=mode == "DZX_local",
                    owner_visit=mode == "DZX_visit",
                )
                rec = dict(tokens=t, routing=kind, mode=mode)
                if mode != "D":
                    rec.update(link_counts(case))
                    rec.update(dma_counts(case))
                    if not args.no_replay:
                        rec.update(replay(case))
                rows.append(rec)
                print(json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
