# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the K32Z expert loader (sglang_tpu/dsv4_mxfp4_k32z.py).

The loader turns DeepSeek-V4.1's checkpoint block-32 MXFP4 experts into the
fused EP kernel's K32Z operands. These tests hold it to:

* a lossless re-layout: decoding the operands the way the kernel addresses
  them gives every checkpoint weight exactly, and k32z_to_mxfp4 gives back the
  checkpoint codes (-0 as +0) and exponents;
* the layout timed in moe-research Step 6 round 4 (k32f.py, expert-bench
  99d278a608: k32f_prep + canon_neg0 + compact_scales), rebuilt here in numpy,
  byte for byte;
* shapes, dtypes, the shift ranges the fields hold and the refusals;
* the K32z decode contract on the operands (FP4 -> e5m2 + 4d);
* the switch: k512 (default) keeps the gmm preparation, k32z defers it.

Real experts: set DSV41_WEIGHTS to a DeepSeek-V4.1-Flash checkpoint directory
(e.g. /weights/DeepSeek-V4.1-Flash) to also check layer-3 experts and scan
every MoE layer's shift range; skipped otherwise. K32Z_TIME_EXPERTS sets the
expert count of the timed conversion (default 8; 48 is one rank's layer).

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q -s <this file>
"""

import json
import os
import re
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import dsv4_mxfp4_k32z as k32z

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

SERVED = (5120, 2304)  # V4.1 Flash hidden, moe_intermediate_size
SMALL = (1536, 768)
E2M1 = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def _round_up(a, b):
    return -(-a // b) * b


def _experts(num_experts, hidden, inter, seed, *, spread=(3, 13)):
    """Checkpoint-format experts (numpy): codes uint8 w13 [E, 2I, H/2], w2
    [E, H, I/2] (all 16 codes, -0 included) and E8M0 exponents e13
    [E, 2I, H/32], e2 [E, H, I/32]: a random base per column plus a shift of at
    most spread[0] (gate/up) / spread[1] (down), each extreme present."""
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, (num_experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (num_experts, hidden, inter // 2), dtype=np.uint8)

    def exps(n, g, top):
        base = rng.integers(100, 126, (num_experts, n, 1))
        d = rng.integers(0, top + 1, (num_experts, n, g))
        d[:, :, 0] = 0  # the column minimum is the base
        d[:, :, -1] = np.where(rng.random((num_experts, n)) < 0.5, top, d[:, :, -1])
        return (base + d).astype(np.uint8)

    return (
        w13,
        exps(2 * inter, hidden // 32, spread[0]),
        w2,
        exps(hidden, inter // 32, spread[1]),
    )


def _f32_scale(e):
    """E8M0 bytes as SGLang's loader leaves the f32 scale parameter."""
    return (
        torch.from_numpy(np.ascontiguousarray(e))
        .view(torch.float8_e8m0fnu)
        .to(torch.float32)
    )


def _convert(w13, e13, w2, e2, scale="f32", **kw):
    as_scale = {
        "f32": _f32_scale,
        "u8": lambda e: torch.from_numpy(e.copy()),
        "e8m0": lambda e: torch.from_numpy(e.copy()).view(torch.float8_e8m0fnu),
    }[scale]
    return k32z.convert_mxfp4_experts_to_k32z(
        torch.from_numpy(w13).view(torch.int8),
        as_scale(e13),
        torch.from_numpy(w2).view(torch.int8),
        as_scale(e2),
        **kw,
    )


def _canon(q):
    lo, hi = q & 0x0F, q & 0xF0
    return (np.where(lo == 0x08, 0, lo) | np.where(hi == 0x80, 0, hi)).astype(np.uint8)


def _reference_operands(w13, e13, w2, e2):
    """Step 6's K32Z operands, numpy (k32f.py at expert-bench 99d278a608:
    _shift_tables / k32f_prep, canon_neg0, compact_scales)."""
    num_experts, n1, kh = w13.shape
    hidden, inter = 2 * kh, n1 // 2
    n2, g1, g2 = hidden, hidden // 32, inter // 32
    g1p, g2p = _round_up(g1, 32), _round_up(g2, 32)
    assert g2p >= g2 + 4

    def shift_tables(e8, gp):
        b = e8.min(-1)
        sp = np.zeros((num_experts, gp, e8.shape[1]), np.uint8)
        sp[:, : e8.shape[2]] = (e8 - b[..., None]).astype(np.uint8).transpose(0, 2, 1)
        return sp, b

    q1 = _canon(w13.transpose(0, 2, 1))
    q2 = _canon(w2.transpose(0, 2, 1))
    s1, b1 = shift_tables(e13, g1p)
    s2, b2 = shift_tables(e2, g2p)
    s1c = np.zeros((num_experts, g1p, n1 // 4), np.uint8)
    for t in range(n1 // 128):
        c = slice((t // 4) * 128, (t // 4 + 1) * 128)
        s1c[:, :, c] |= (s1[:, :, t * 128 : (t + 1) * 128] << (2 * (t % 4))).astype(
            np.uint8
        )
    s2c = np.zeros((num_experts, g2p, n2 // 2), np.uint8)
    for t in range(n2 // 128):
        c = slice((t // 2) * 128, (t // 2 + 1) * 128)
        s2c[:, :g2, c] |= (s2[:, :g2, t * 128 : (t + 1) * 128] << (4 * (t % 2))).astype(
            np.uint8
        )
        s2c[:, g2 + t % 2, c] = b2[:, t * 128 : (t + 1) * 128]
    for t in range(n1 // 128):
        c = slice((t // 2) * 128, (t // 2 + 1) * 128)
        s2c[:, g2 + 2 + t % 2, c] = b1[:, t * 128 : (t + 1) * 128]
    return q1, s1c, q2, s2c


def _kernel_view(ops, e, which, hidden, inter, gate_up_bits=2):
    """Matmul ``which`` of expert ``e`` as the kernel addresses the operands:
    (codes [K, N], shifts [K/32, N], bases [N]), numpy int64."""
    s2 = ops.w2_shift[e].numpy().astype(np.int64)
    g2 = inter // 32
    if which == 1:
        q, tab, bits, n, g = (
            ops.w13[e].numpy(),
            ops.w13_shift[e].numpy(),
            gate_up_bits,
            2 * inter,
            hidden // 32,
        )
        base_rows = s2[g2 + 2 : g2 + 4]
    else:
        q, tab, bits, n, g = ops.w2[e].numpy(), ops.w2_shift[e].numpy(), 4, hidden, g2
        base_rows = s2[g2 : g2 + 2]
    codes = np.stack([q & 15, q >> 4], 1).reshape(2 * q.shape[0], n).astype(np.int64)
    per, cols = 8 // bits, np.arange(n)
    tiles, lanes = cols // 128, cols % 128
    byte_col = (tiles // per) * 128 + lanes
    d = (tab[:g, byte_col].astype(np.int64) >> (bits * (tiles % per))) & (
        (1 << bits) - 1
    )
    b = base_rows[tiles % 2, (tiles // 2) * 128 + lanes]
    return codes, d, b


def _ckpt_values(w, e):
    """Checkpoint [N, K/2] codes + [N, K/32] exponents -> f64 [K, N]."""
    codes = np.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1)
    return (
        E2M1[codes] * np.exp2(np.repeat(e.astype(np.float64), 32, axis=1) - 127.0)
    ).T


# --------------------------------------------------------------------------- #
# Lossless re-layout.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "geometry,num_experts", [(SMALL, 3), (SERVED, 2)], ids=["small", "served"]
)
@pytest.mark.parametrize("scale", ["f32", "u8", "e8m0"])
def test_lossless_round_trip(geometry, num_experts, scale):
    hidden, inter = geometry
    w13, e13, w2, e2 = _experts(num_experts, hidden, inter, seed=hidden + len(scale))
    ops = _convert(w13, e13, w2, e2, scale=scale)
    assert ops.max_shift == (3, 13)

    back = [t.numpy() for t in k32z.k32z_to_mxfp4(ops)]
    np.testing.assert_array_equal(back[0], _canon(w13))
    np.testing.assert_array_equal(back[1], e13)
    np.testing.assert_array_equal(back[2], _canon(w2))
    np.testing.assert_array_equal(back[3], e2)

    # Every weight, decoded from the operands as the kernel addresses them
    # (codes x 2^(b + d - 127)), equals the checkpoint weight.
    for e in range(num_experts):
        for which, w, ex in ((1, w13, e13), (2, w2, e2)):
            codes, d, b = _kernel_view(ops, e, which, hidden, inter)
            got = E2M1[codes] * np.exp2(np.repeat(d, 32, axis=0) + b[None, :] - 127.0)
            np.testing.assert_array_equal(got, _ckpt_values(w[e], ex[e]))


@pytest.mark.parametrize("geometry", [SMALL, SERVED], ids=["small", "served"])
def test_matches_step6_reference_layout(geometry):
    w13, e13, w2, e2 = _experts(3, *geometry, seed=7)
    want = _reference_operands(w13, e13, w2, e2)
    for chunk in (1, 2, 8):
        ops = _convert(w13, e13, w2, e2, experts_per_chunk=chunk)
        for got, ref in zip(ops[:4], want):
            np.testing.assert_array_equal(got.numpy(), ref)


def test_shapes_dtypes_and_padding():
    hidden, inter = SERVED
    ops = _convert(*_experts(2, hidden, inter, seed=3))
    geo = k32z.k32z_geometry(hidden, inter)
    assert (geo.g1, geo.g1p, geo.g2, geo.g2p) == (160, 160, 72, 96)
    shapes = [tuple(t.shape) for t in ops[:4]]
    assert shapes == [(2, 2560, 4608), (2, 160, 1152), (2, 1152, 5120), (2, 96, 2560)]
    assert all(t.dtype == torch.uint8 and t.is_contiguous() for t in ops[:4])
    s2 = ops.w2_shift.numpy()
    assert not s2[:, 76:].any(), "rows past the base rows are zero"
    assert not s2[:, 74:76, inter:].any(), "gate/up base rows hold I byte columns"
    assert (s2[:, 72:76, :inter] > 0).all(), "bases are >= 1"
    # Down fields are 4 bits, so the shift table bytes of rows < G2 carry two.
    assert int(s2[:, :72].max()) > 15


def test_k32z_decode_contract_on_operands():
    """FP4 -> e5m2 + 4d (one u32 add per 4 bytes in the kernel) times the column
    scale: nonzero weights exact, zero codes with d >= 1 -> 2^(d - 15) x 2^(b - 127)."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    hidden, inter = SMALL
    w13, e13, w2, e2 = _experts(2, hidden, inter, seed=11)
    ops = _convert(w13, e13, w2, e2)
    e5 = ml_dtypes.float8_e5m2
    fp4_to_e5 = E2M1.astype(np.float32).astype(e5).view(np.uint8).astype(np.int64)
    e5_value = np.arange(256, dtype=np.uint8).view(e5).astype(np.float64)
    for which, w, ex in ((1, w13, e13), (2, w2, e2)):
        codes, d, b = _kernel_view(ops, 1, which, hidden, inter)
        byte = fp4_to_e5[codes] + 4 * np.repeat(d, 32, axis=0)
        # No carry into the sign bit (or the next byte), and below inf.
        assert int((byte & 0x7F).max()) <= 0x7B and int(byte.max()) <= 0xFB
        assert np.array_equal(byte & 0x80, fp4_to_e5[codes] & 0x80)
        got = e5_value[byte] * np.exp2(b[None, :] - 127.0)
        want = _ckpt_values(w[1], ex[1])
        nonzero = E2M1[codes] != 0
        np.testing.assert_array_equal(got[nonzero], want[nonzero])
        dd = np.repeat(d, 32, axis=0)[~nonzero]
        expect = np.where(dd == 0, 0.0, np.exp2(dd - 15.0)) * np.exp2(
            np.broadcast_to(b, codes.shape)[~nonzero] - 127.0
        )
        np.testing.assert_array_equal(got[~nonzero], expect)


# --------------------------------------------------------------------------- #
# Ranges and refusals.
# --------------------------------------------------------------------------- #
def test_shift_ranges_and_refusals():
    hidden, inter = SMALL
    w13, e13, w2, e2 = _experts(2, hidden, inter, seed=5)
    assert _convert(w13, e13, w2, e2).max_shift == (3, 13)

    over13 = e13.copy()
    over13[0, 5, 3] = over13[0, 5].min() + 4
    with pytest.raises(ValueError, match="gate/up shift 4 exceeds the 3"):
        _convert(w13, over13, w2, e2)
    # 4-bit gate/up fields hold it (an option the kernel would have to read).
    ops = _convert(w13, over13, w2, e2, gate_up_bits=4)
    assert ops.max_shift == (4, 13) and ops.w13_shift.shape[-1] == inter
    np.testing.assert_array_equal(
        k32z.k32z_to_mxfp4(ops, gate_up_bits=4)[1].numpy(), over13
    )

    over2 = e2.copy()
    over2[1, 7, 2] = over2[1, 7].min() + 14
    with pytest.raises(ValueError, match="down shift 14 exceeds the 13"):
        _convert(w13, e13, w2, over2)

    nan = e2.copy()
    nan[0, 0, 0] = 255
    with pytest.raises(ValueError, match="0xFF"):
        _convert(w13, e13, w2, nan, scale="u8")
    with pytest.raises(ValueError, match="not an E8M0 power of two"):
        _convert(w13, e13, w2, nan, scale="f32")
    zero_base = e2.copy()
    zero_base[0, 3, :] = 0  # 2^-127, the f32 subnormal: read, but no column scale
    assert int(k32z.e8m0_exponents(_f32_scale(zero_base))[0, 3].max()) == 0
    with pytest.raises(ValueError, match=r"base exponent outside \[1, 254\]"):
        _convert(w13, e13, w2, zero_base)
    not_pow2 = _f32_scale(e13)
    not_pow2[0, 0, 0] *= 1.5
    with pytest.raises(ValueError, match="not an E8M0 power of two"):
        k32z.e8m0_exponents(not_pow2)
    with pytest.raises(ValueError, match="do not match"):
        k32z.convert_mxfp4_experts_to_k32z(
            torch.from_numpy(w13),
            _f32_scale(e13),
            torch.from_numpy(w2[:, :, 1:]),
            _f32_scale(e2),
        )
    with pytest.raises(ValueError, match="cannot lay out"):
        k32z.k32z_geometry(1024, 768)  # gate/up bases need I <= H/2 byte columns


def test_conversion_time():
    """Wall time of one conversion at the served geometry (printed with -s)."""
    num_experts = int(os.environ.get("K32Z_TIME_EXPERTS", "8"))
    hidden, inter = SERVED
    w13, e13, w2, e2 = _experts(num_experts, hidden, inter, seed=13)
    args = (
        torch.from_numpy(w13).view(torch.int8),
        _f32_scale(e13),
        torch.from_numpy(w2).view(torch.int8),
        _f32_scale(e2),
    )
    t0 = time.perf_counter()
    ops = k32z.convert_mxfp4_experts_to_k32z(*args)
    seconds = time.perf_counter() - t0
    print(
        f"\n[k32z] {num_experts} experts ({hidden}x{inter}, CPU torch "
        f"{torch.get_num_threads()} threads): {seconds:.3f} s"
    )
    assert ops.w13.shape[0] == num_experts


# --------------------------------------------------------------------------- #
# The layer step and the switch.
# --------------------------------------------------------------------------- #
def _layer(hidden, inter, num_experts=2, seed=17):
    from torch.nn import Module, Parameter

    w13, e13, w2, e2 = _experts(num_experts, hidden, inter, seed=seed)
    layer = Module()
    layer.w13_weight = Parameter(
        torch.from_numpy(w13).view(torch.int8).clone(), requires_grad=False
    )
    layer.w2_weight = Parameter(
        torch.from_numpy(w2).view(torch.int8).clone(), requires_grad=False
    )
    layer.w13_weight_scale_inv = Parameter(_f32_scale(e13), requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(_f32_scale(e2), requires_grad=False)
    return layer, (w13, e13, w2, e2)


def test_layer_preparation_replaces_parameters():
    hidden, inter = SMALL
    layer, ckpt = _layer(hidden, inter)
    assert k32z.checkpoint_layout_ok(layer, hidden, inter)
    w13_param = layer.w13_weight
    old_storage = layer.w13_weight_scale_inv.data.untyped_storage()
    ops = k32z.prepare_tpu_dsv4_mxfp4_k32z_weights(layer)
    assert layer.w13_weight is w13_param, "the Parameter object is kept"
    assert layer.w13_weight_scale_inv is None and layer.w2_weight_scale_inv is None
    assert old_storage.nbytes() == 0, "the replaced buffers are freed"
    assert getattr(layer, k32z.FORMAT_ATTR) == k32z.K32Z and not k32z.is_deferred(layer)
    want = _reference_operands(*ckpt)
    got = (
        layer.w13_weight,
        layer.w13_weight_shift,
        layer.w2_weight,
        layer.w2_weight_shift,
    )
    for t, ref in zip(got, want):
        np.testing.assert_array_equal(t.data.numpy(), ref)
    assert layer.k32z_max_shift == ops.max_shift


def _prepare_fp4(layer):
    from sglang_tpu.fp8_moe import prepare_tpu_fp8_moe_weights

    prepare_tpu_fp8_moe_weights(
        layer,
        block_quant=True,
        weight_block_size=None,
        use_mxfp8=False,
        is_fp4_expert=True,
    )


def test_switch_selects_the_preparation(monkeypatch):
    hidden, inter = 1024, 512
    monkeypatch.delenv("SGLANG_TPU_DSV41_EXPERT_FORMAT", raising=False)
    assert k32z.selected_expert_format() == k32z.K512
    default, _ = _layer(hidden, inter)
    _prepare_fp4(default)  # the gmm preparation, as without the switch
    assert default.w13_weight_scale.dtype == torch.float32
    assert not k32z.is_deferred(default)

    monkeypatch.setenv("SGLANG_TPU_DSV41_EXPERT_FORMAT", "k32z")
    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "0")
    with pytest.raises(ValueError, match="needs SGLANG_TPU_DSV41_FUSED_EP=1"):
        _prepare_fp4(_layer(hidden, inter)[0])
    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "1")
    k32z.check_expert_format_env()
    deferred, _ = _layer(hidden, inter)
    _prepare_fp4(deferred)
    assert k32z.is_deferred(deferred) and k32z.checkpoint_layout_ok(
        deferred, hidden, inter
    )

    # A deferred layer that keeps gmm_v2 gets exactly the default layout.
    holder = torch.nn.ModuleList([deferred])
    assert k32z.finish_deferred_as_k512(holder) == 1
    assert not k32z.is_deferred(deferred)
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        a, b = getattr(deferred, name).data, getattr(default, name).data
        assert a.dtype == b.dtype and torch.equal(
            a.view(torch.uint8), b.view(torch.uint8)
        ), name
    assert k32z.finish_deferred_as_k512(holder) == 0

    monkeypatch.setenv("SGLANG_TPU_DSV41_EXPERT_FORMAT", "k32")
    with pytest.raises(ValueError, match="must be one of"):
        k32z.selected_expert_format()


# --------------------------------------------------------------------------- #
# Real checkpoint (DSV41_WEIGHTS).
# --------------------------------------------------------------------------- #
_WEIGHTS = os.environ.get("DSV41_WEIGHTS")
real = pytest.mark.skipif(
    not (_WEIGHTS and (Path(_WEIGHTS) / "model.safetensors.index.json").exists()),
    reason="DSV41_WEIGHTS does not point at a DeepSeek-V4.1-Flash checkpoint",
)


def _st_read(path, names):
    """{name: uint8 ndarray} for checkpoint tensors (I8 / F8_E8M0 / U8)."""
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    by_file = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    out = {}
    for fname, group in by_file.items():
        with open(Path(path) / fname, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            base = f.tell()
            for n in group:
                h = header[n]
                assert h["dtype"] in ("I8", "F8_E8M0", "U8"), (n, h["dtype"])
                a, b = h["data_offsets"]
                f.seek(base + a)
                out[n] = np.frombuffer(f.read(b - a), np.uint8).reshape(h["shape"])
    return out


def _real_experts(path, layer, experts, parts=("weight", "scale")):
    pre = f"layers.{layer}.ffn.experts."
    names = [
        f"{pre}{e}.{w}.{p}" for e in experts for w in ("w1", "w2", "w3") for p in parts
    ]
    t = _st_read(path, names)

    def get(e, w, p):
        return t[f"{pre}{e}.{w}.{p}"]

    out = {}
    for p in parts:
        out[f"w13_{p}"] = np.stack(
            [np.concatenate([get(e, "w1", p), get(e, "w3", p)]) for e in experts]
        )
        out[f"w2_{p}"] = np.stack([get(e, "w2", p) for e in experts])
    return out


@real
def test_real_layer3_experts_round_trip():
    experts = (0, 1, 128, 255)
    t = _real_experts(_WEIGHTS, 3, experts)
    w13, e13, w2, e2 = t["w13_weight"], t["w13_scale"], t["w2_weight"], t["w2_scale"]
    ops = _convert(w13, e13, w2, e2)
    print(
        f"\n[k32z] layer 3 experts {experts}: max shift (gate/up, down) {ops.max_shift}"
    )
    back = [x.numpy() for x in k32z.k32z_to_mxfp4(ops)]
    np.testing.assert_array_equal(back[0], _canon(w13))
    np.testing.assert_array_equal(back[1], e13)
    np.testing.assert_array_equal(back[2], _canon(w2))
    np.testing.assert_array_equal(back[3], e2)
    for got, ref in zip(ops[:4], _reference_operands(w13, e13, w2, e2)):
        np.testing.assert_array_equal(got.numpy(), ref)
    hidden, inter = 2 * w13.shape[2], w13.shape[1] // 2
    for which, w, ex in ((1, w13, e13), (2, w2, e2)):
        codes, d, b = _kernel_view(ops, 0, which, hidden, inter)
        got = E2M1[codes] * np.exp2(np.repeat(d, 32, axis=0) + b[None, :] - 127.0)
        np.testing.assert_array_equal(got, _ckpt_values(w[0], ex[0]))


@real
def test_real_checkpoint_shift_ranges():
    """Every routed-expert MoE layer's largest K32Z shifts (scales only) fit the
    served fields: 4-bit gate/up and down, within e5m2 (<= 13). (The 2026-11-22
    scan of V4.1 Flash found gate/up shifts up to 6, so 2-bit gate/up fields,
    <= 3, hold only some layers; the count is printed.)"""
    index = json.loads((Path(_WEIGHTS) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    found = {}
    for name in index:
        m = re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.scale", name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    worst = {}
    for layer in sorted(found):
        t = _real_experts(_WEIGHTS, layer, sorted(found[layer]), parts=("scale",))
        d13 = t["w13_scale"].astype(np.int32)
        d2 = t["w2_scale"].astype(np.int32)
        worst[layer] = (
            int((d13 - d13.min(-1, keepdims=True)).max()),
            int((d2 - d2.min(-1, keepdims=True)).max()),
        )
        print(f"[k32z] layer {layer}: max shift (gate/up, down) {worst[layer]}")
    assert found, "no routed experts in the index"
    fits2 = sorted(layer for layer, (g, _) in worst.items() if g <= 3)
    print(
        f"[k32z] {len(fits2)} of {len(worst)} layers fit 2-bit gate/up fields: {fits2}"
    )
    served = k32z.k32z_geometry(SERVED[0], SERVED[1], gate_up_bits=4)
    assert all(
        g <= served.max_shift(1) and d <= served.max_shift(2) for g, d in worst.values()
    ), worst
