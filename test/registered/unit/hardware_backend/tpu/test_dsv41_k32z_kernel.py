# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fused_ep_moe_v2's K32Z weight format (Step 7 N2a).

K32Z keeps DeepSeek-V4.1's block-32 MXFP4 experts lossless: e2m1 codes plus
per-column shift tables (sglang_tpu/dsv4_mxfp4_k32z.py builds them). The
kernel converts a block of codes to e5m2, adds 4 * d to every byte and applies
one power of two per column after the K sum. These tests hold it to:

* the table geometry the loader lays out, the VMEM buffers and the build's
  refusals;
* the decode, bit for bit: the kernel's own table build, column-scale fill and
  block readers, run in the Pallas TPU interpreter, against an independent
  numpy decode of the checkpoint (code -> e5m2 byte + 4d, d from the
  checkpoint exponents);
* the expert FFN (expert_ffn_colscale + the wire epilogue) in the interpreter
  against the same math on the numpy-decoded weights and against an FP64
  replica with exact K32 weights at the kernel's rounding points;
* the ablation switch: an empty set traces to the very program of no
  argument, a non-empty one renames it; the layer lowers for TPU (Mosaic) on
  this host;
* the push dispatch (Step 7 N2b): the push kernel binds the same decode and
  FFN (run through its own names), its K32Z and ablation builds refuse what
  they cannot take, and the push layer lowers for TPU with K32Z, on the push
  defaults and on #2102's tuned bundle, at the served shapes too (those need
  eight CPU devices: run this file in its own process); make_fused_ep_moe's
  dispatch switch (push, auto per traced shape, the default call unchanged).

Nothing here needs a TPU; the interpreter has no stride-0 load, so the readers
take their ``interpret`` broadcast (the TPU path is checked on the chip).

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import contextlib
import os
import re
import sys
from functools import partial
from types import SimpleNamespace

# CPU JAX with virtual devices for the EP meshes; no effect once JAX is up.
if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=8"
        ).strip()

import numpy as np
import pytest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


# The TPU unit lane has jax and ml_dtypes; the CPU lane does not.
register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
ml_dtypes = pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")

import dsv41_fused_ep_reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sglang_tpu import dsv4_mxfp4_k32z as k32z_loader  # noqa: E402
from sglang_tpu.models.deepseek_v41 import fused_ep as fe  # noqa: E402

SERVED = (5120, 2304)  # V4.1 Flash hidden, moe_intermediate_size
# Reduced: three gate/up K blocks, a down K tail of 256, six chunks each.
SMALL = (1536, 768)
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
SHIFT_WORD = 0x04040404


@pytest.fixture(scope="module")
def v2():
    try:
        package = ref.import_fused_moe_v2(isolated=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fused_moe_v2 does not import here: {exc!r}")
    from sglang_tpu.moe.pallas.fused_moe_v2 import host, kernel, layer
    from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import kernel as push_kernel

    return SimpleNamespace(
        package=package, host=host, kernel=kernel, layer=layer, push_kernel=push_kernel
    )


def _bindings(v2, via):
    """The K32Z compute functions as the v2 ("v2") or the push ("push") kernel
    module binds them: the push kernel calls v2's, under its own names."""
    if via == "v2":
        return v2.kernel
    names = (
        "k32z_block_readers",
        "k32z_fill_colscale",
        "expert_ffn_colscale",
        "wire_row_and_scale",
    )
    return SimpleNamespace(
        k32z_blocks=v2.kernel.k32z_blocks,
        **{n: getattr(v2.push_kernel, n) for n in names},
    )


class _FakeChip:
    """What the VMEM accounting reads off a v7 device record (64 MiB VMEM,
    the 32-byte sublane rule)."""

    num_lanes = 128
    generation = 7

    def __init__(self, vmem_capacity_bytes=64 * 2**20):
        self.vmem_capacity_bytes = vmem_capacity_bytes

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


@contextlib.contextmanager
def _tpu7x_target():
    """Lower Pallas TPU kernels for TPU7x on this host (as
    sglang_tpu.pallas_export.tpu_info_for_cpu_export does for torch_tpu)."""
    from jax._src import tpu_info as core_tpu_info
    from jax._src.pallas.mosaic import tpu_info

    names = ("get_tpu_info", "get_device_kind", "is_tpu_device")
    modules = (tpu_info, core_tpu_info)
    saved_public = pltpu.get_tpu_info
    saved = [{n: getattr(m, n) for n in names} for m in modules]
    kind = "TPU7x"
    info = tpu_info.get_tpu_info_for_chip(
        tpu_info.chip_version_from_device_kind(kind), 1
    )
    pltpu.get_tpu_info = lambda: info
    for m in modules:
        m.get_tpu_info = lambda: info
        m.get_device_kind = lambda: kind
        m.is_tpu_device = lambda: True
    try:
        yield info
    finally:
        pltpu.get_tpu_info = saved_public
        for m, old in zip(modules, saved):
            for n, v in old.items():
                setattr(m, n, v)


# --------------------------------------------------------------------------- #
# Checkpoint experts and an independent numpy decode.
# --------------------------------------------------------------------------- #
def _experts(num_experts, hidden, inter, seed, *, top=(3, 13)):
    """Checkpoint-format experts: codes w13 [E, 2I, H/2], w2 [E, H, I/2]
    (all 16 codes) and E8M0 exponents e13 [E, 2I, H/32], e2 [E, H, I/32]; a
    random base per column plus a shift of at most top[0] (gate/up) / top[1]
    (down), each extreme present."""
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, (num_experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (num_experts, hidden, inter // 2), dtype=np.uint8)

    def exps(n, g, t):
        base = rng.integers(110, 124, (num_experts, n, 1))
        d = rng.integers(0, t + 1, (num_experts, n, g))
        d[:, :, 0] = 0
        d[:, :, -1] = np.where(rng.random((num_experts, n)) < 0.5, t, d[:, :, -1])
        return (base + d).astype(np.uint8)

    return (
        w13,
        exps(2 * inter, hidden // 32, top[0]),
        w2,
        exps(hidden, inter // 32, top[1]),
    )


def _convert(w13, e13, w2, e2, gate_up_bits=2):
    return k32z_loader.convert_mxfp4_experts_to_k32z(
        torch.from_numpy(w13).view(torch.int8),
        torch.from_numpy(e13.copy()),
        torch.from_numpy(w2).view(torch.int8),
        torch.from_numpy(e2.copy()),
        gate_up_bits=gate_up_bits,
    )


def _codes_kn(w):
    """Checkpoint codes [N, K/2] -> [K, N] (low nibble = even K)."""
    return np.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1).T


def _shift_base(e):
    """Exponents [N, K/32] -> (d [K/32, N], b [N]) from the checkpoint alone."""
    e = e.astype(np.int64)
    b = e.min(-1)
    return (e - b[:, None]).T, b


def _k32z_e5m2_bytes(w, e):
    """The K32Z decode in numpy: e2m1 code -> e5m2 byte (-0 as +0) + 4d.
    Returns uint8 [K, N]."""
    codes = _codes_kn(w)
    lut = np.asarray(E2M1, np.float32).astype(ml_dtypes.float8_e5m2).view(np.uint8)
    lut[8] = 0  # -0 -> +0, as the loader canonicalizes
    d, _ = _shift_base(e)
    return (lut[codes] + 4 * np.repeat(d, 32, axis=0)).astype(np.uint8)


def _k32_exact(w, e):
    """Exact checkpoint values f64 [K, N]."""
    codes = _codes_kn(w)
    return E2M1[codes] * np.exp2(np.repeat(e.astype(np.float64).T, 32, axis=0) - 127)


def _pack_u32(q):
    """u8 [K/2, N] K-packed codes -> u32 [K/8, N], word r = bytes of rows
    4r..4r+3 little endian (the kernel's packed-weight view)."""
    r, n = q.shape
    return (
        np.ascontiguousarray(np.swapaxes(q.reshape(r // 4, 4, n), 1, 2))
        .view("<u4")
        .reshape(r // 4, n)
    )


# --------------------------------------------------------------------------- #
# Geometry, buffers, rungs, refusals.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("geometry", [SERVED, SMALL], ids=["served", "small"])
def test_table_geometry_matches_the_loader(v2, geometry, bits):
    geo = v2.host.k32z_table_geometry(*geometry, bits)
    lg = k32z_loader.k32z_geometry(*geometry, gate_up_bits=bits)
    assert (geo.rows1, geo.cols1) == (lg.g1p, lg.w13_shift_cols)
    assert (geo.rows2, geo.cols2) == (lg.g2p, lg.w2_shift_cols)
    assert (geo.g1, geo.g2) == (lg.g1, lg.g2)
    if geometry == SERVED:
        assert (geo.rows1, geo.cols1, geo.rows2, geo.cols2) == (
            160,
            1152 * bits // 2,
            96,
            2560,
        )
        assert geo.tab_rows == 160


@pytest.mark.parametrize(
    "hidden,inter,bits,what",
    [
        (5120, 2304, 3, "bits wide"),
        (5000, 2304, 2, "multiples of 256"),
        (5120, 2300, 2, "multiples of 256"),
        (2048, 2304, 2, "byte columns"),
    ],
)
def test_table_geometry_refusals(v2, hidden, inter, bits, what):
    with pytest.raises(ValueError, match=what):
        v2.host.k32z_table_geometry(hidden, inter, bits)


def test_scratch_and_tile_body_buffers(v2):
    host = v2.host
    k32z = host.WeightFormat.K32Z
    arrays = host.vmem_scratch_arrays(48, 256, *SERVED, nbuf=2, weight_format=k32z)
    got = {name: (shape, jnp.dtype(dt).name) for name, shape, dt in arrays}
    assert [name for name, _, _ in arrays] == [
        "lhs_vm",
        "w1_vm",
        "w2_vm",
        "w1s_vm",
        "w2s_vm",
        "tab_vm",
        "col_vm",
        "ls_vm",
        "out_vm",
        "oscl_vm",
    ]
    assert got["w1_vm"] == ((2, 640, 4608), "uint32")
    assert got["w2_vm"] == ((2, 288, 5120), "uint32")
    assert got["w1s_vm"] == ((2, 160, 1152), "uint8")
    assert got["w2s_vm"] == ((2, 96, 2560), "uint8")
    assert got["tab_vm"] == ((2, 2, 160, 128), "int32")
    assert got["col_vm"] == ((8, 5120), "float32")
    # A 256-row tile's scale sublane is split over two 128-lane rows.
    assert got["oscl_vm"] == ((2, 2, 128), "float32")
    assert got["lhs_vm"] == ((2, 256, 40, 128), "float8_e4m3fn")
    assert got["out_vm"] == ((2, 256, 40, 128), "float8_e4m3fn")
    wide = host.vmem_scratch_arrays(
        48, 256, *SERVED, nbuf=2, weight_format=k32z, k32z_gate_up_bits=4
    )
    assert dict((n, s) for n, s, _ in wide)["w1s_vm"] == (2, 160, 2304)
    body = {
        name: (shape, jnp.dtype(dt).name)
        for name, shape, dt in host.vmem_tile_body_arrays(
            256, *SERVED, weight_format=k32z
        )
    }
    assert body["decoded_weight_block"] == ((2, 512, 256), "float8_e5m2")
    assert body["shift_words"] == ((2, 128, 256), "int32")
    assert "widened_weight_block" not in body
    # FP4 is untouched.
    fp4 = host.vmem_scratch_arrays(
        48, 128, 5120, 2560, nbuf=2, weight_format=host.WeightFormat.FP4
    )
    assert "tab_vm" not in [n for n, _, _ in fp4]


def test_k32z_keeps_its_weight_prefetch(v2):
    """The served build at 256-row tiles keeps prefetch distance 1 (nbuf 2)
    under a 64 MiB VMEM, at either gate/up field width."""
    host = v2.host
    chip = _FakeChip()
    for bits in (2, 4):
        slots = host.fit_weight_slots(
            48,
            256,
            *SERVED,
            weight_format=host.WeightFormat.K32Z,
            info=chip,
            k32z_gate_up_bits=bits,
        )
        assert (slots.distance, slots.nbuf) == (1, 2), (bits, slots)
        assert slots.fits
    fp4 = host.fit_weight_slots(
        48, 128, 5120, 2560, weight_format=host.WeightFormat.FP4, info=chip
    )
    assert (fp4.distance, fp4.nbuf) == (1, 2)


def test_mirror_split(v2):
    """Up to 128-row tiles the scale mirrors keep their [sublanes, tile_m]
    shape (the FP4 program is unchanged); taller tiles split a sublane over
    128-lane rows of a 3-D mirror whose sublane axis is untiled."""
    host = v2.host
    for tile_m in (32, 64, 128):
        assert host.mirror_split(tile_m) == 1
        assert host.mirror_shape(102, tile_m) == (102, tile_m)
    assert host.mirror_split(256) == 2
    assert host.mirror_shape(102, 256) == (102, 2, 128)
    fp4 = host.vmem_scratch_arrays(
        48, 128, 5120, 2560, nbuf=2, weight_format=host.WeightFormat.FP4
    )
    assert dict((n, s) for n, s, _ in fp4)["oscl_vm"] == (2, 1, 128)


def test_rungs_and_formats(v2):
    host = v2.host
    assert host.k32z_tile_height_rungs(256) == (32, 64, 128, 256)
    assert host.k32z_tile_height_rungs(128) == (32, 64, 128)
    # 128 is a rung wherever a taller tile exists: the slab waits move there.
    for tile_m in (160, 192, 224, 256):
        assert host.K32Z_STREAMED_ROWS in host.k32z_tile_height_rungs(tile_m)
    # float4 weights are FP4 unless K32Z is named.
    assert host.weight_format_of_dtype(jnp.float4_e2m1fn) == host.WeightFormat.FP4
    form = host.weight_form("k32z")
    assert (
        form.weight_dtype == jnp.float4_e2m1fn and form.scale_layout == "column_shift"
    )
    assert form.wire_dtype == host.FP8 and form.quantized_activations


@pytest.mark.parametrize(
    "kw,what",
    [
        (dict(k32z_gate_up_bits=3), "bits wide"),
        (dict(hidden=5000), "multiples of 256"),
        (dict(capacity=240), "rungs"),
        (dict(logical_intermediate_size=2048), "logical intermediate"),
        (dict(ablation=frozenset({"bogus"})), "unknown ablation"),
        (dict(ablation="w_off"), "string"),
    ],
)
def test_build_refusals(v2, kw, what):
    args = dict(
        g_local=48,
        capacity=256,
        hidden=5120,
        inter=2304,
        ep=8,
        weight_format=v2.host.WeightFormat.K32Z,
        ragged_rows_alloc=4096,
        act_fn="silu_and_mul_with_clamp",
    )
    args.update(kw)
    with pytest.raises(ValueError, match=what):
        v2.kernel._build_fused_ep_moe_kernel(**args)


def test_ablation_names(v2):
    kernel = v2.kernel
    assert kernel.check_ablation(None) == frozenset()
    assert kernel.check_ablation(["w_off", "ffn_off"]) == {"w_off", "ffn_off"}
    assert kernel.ablation_tag(frozenset()) == ""
    assert kernel.ablation_tag(frozenset({"w_off", "ffn_off"})) == "_ffn_off_w_off"
    with pytest.raises(ValueError, match="unknown ablation"):
        kernel.check_ablation({"comp"})


# --------------------------------------------------------------------------- #
# The table math, against the checkpoint's own shifts.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 4])
def test_shift_words_match_the_checkpoint_shifts(v2, bits):
    """kernel.k32z_shift_words on each chunk's byte column of the loader's
    table, for every chunk and column tile the kernel builds, equals d * 4 in
    every byte with d taken from the checkpoint exponents."""
    hidden, inter = SERVED
    w13, e13, w2, e2 = _experts(2, hidden, inter, seed=3)
    ops = _convert(w13, e13, w2, e2, gate_up_bits=bits)
    for e in range(2):
        for which, table, exps, groups, n, b in (
            (1, ops.w13_shift, e13, hidden // 32, 2 * inter, bits),
            (2, ops.w2_shift, e2, inter // 32, hidden, 4),
        ):
            d, _ = _shift_base(exps[e])
            tab = table[e].numpy().astype(np.int32)
            for n0 in range(0, n, 256):
                byte_tile, field = divmod(n0 // 128, 8 // b)
                raw = jnp.asarray(tab[:groups, byte_tile * 128 : (byte_tile + 1) * 128])
                for t in range(2):
                    got = np.asarray(v2.kernel.k32z_shift_words(raw, field + t, b))
                    cols = slice(n0 + 128 * t, n0 + 128 * (t + 1))
                    np.testing.assert_array_equal(
                        got,
                        (d[:, cols] * SHIFT_WORD).astype(np.int32),
                        err_msg=f"{which} {n0} {t}",
                    )


# --------------------------------------------------------------------------- #
# The kernel's decode and FFN, in the Pallas TPU interpreter.
# --------------------------------------------------------------------------- #
def _operands(ops, e, hidden, inter, bits):
    """One expert's kernel operands with a leading slot axis of 1."""
    return (
        jnp.asarray(_pack_u32(ops.w13[e].numpy())[None]),
        jnp.asarray(_pack_u32(ops.w2[e].numpy())[None]),
        jnp.asarray(ops.w13_shift[e].numpy()[None]),
        jnp.asarray(ops.w2_shift[e].numpy()[None]),
    )


def _scratch(v2, hidden, inter, bits):
    geo = v2.host.k32z_table_geometry(hidden, inter, bits)
    return [
        pltpu.VMEM((2, 2, geo.tab_rows, 128), jnp.int32),
        pltpu.VMEM((8, max(2 * inter, hidden)), jnp.float32),
    ]


def _decode_call(v2, hidden, inter, bits, via="v2"):
    kernel = _bindings(v2, via)

    def body(w1, w2, s1, s2, d1, d2, col, tab_vm, col_vm):
        slot = jnp.int32(0)
        kernel.k32z_fill_colscale(s2, slot, col_vm, hidden=hidden, inter=inter)
        col[...] = col_vm[...]
        readers = kernel.k32z_block_readers(
            w1,
            w2,
            s1,
            s2,
            tab_vm,
            slot,
            hidden=hidden,
            inter=inter,
            gate_up_bits=bits,
            interpret=True,
        )
        for read, out, k, n in (
            (readers[0], d1, hidden, 2 * inter),
            (readers[1], d2, inter, hidden),
        ):
            for n0 in range(0, n, 256):
                for b, (k0, rows) in enumerate(kernel.k32z_blocks(k)):
                    out[pl.ds(k0, rows), pl.ds(n0, 256)] = read(b, n0, n0 + 256)

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((hidden, 2 * inter), jnp.float8_e5m2),
            jax.ShapeDtypeStruct((inter, hidden), jnp.float8_e5m2),
            jax.ShapeDtypeStruct((8, max(2 * inter, hidden)), jnp.float32),
        ),
        scratch_shapes=_scratch(v2, hidden, inter, bits),
        interpret=pltpu.InterpretParams(),
    )


@pytest.mark.parametrize("via", ["v2", "push"])
@pytest.mark.parametrize("bits", [2, 4])
def test_kernel_decode_is_k32z_bit_for_bit(v2, bits, via):
    """The kernel's table build, column-scale fill and block readers (the
    interpreter's stand-in for the stride-0 load aside) decode every block
    to code -> e5m2 + 4d, and the column scales to 2^(b - 127); through the
    push kernel's bindings too."""
    hidden, inter = SMALL
    w13, e13, w2, e2 = _experts(2, hidden, inter, seed=11)
    ops = _convert(w13, e13, w2, e2, gate_up_bits=bits)
    call = jax.jit(_decode_call(v2, hidden, inter, bits, via))
    for e in range(2):
        d1, d2, col = call(*_operands(ops, e, hidden, inter, bits))
        np.testing.assert_array_equal(
            np.asarray(d1).view(np.uint8), _k32z_e5m2_bytes(w13[e], e13[e])
        )
        np.testing.assert_array_equal(
            np.asarray(d2).view(np.uint8), _k32z_e5m2_bytes(w2[e], e2[e])
        )
        _, b1 = _shift_base(e13[e])
        _, b2 = _shift_base(e2[e])
        col = np.asarray(col)
        np.testing.assert_array_equal(
            col[0, : 2 * inter], np.exp2(b1 - 127.0).astype(np.float32)
        )
        np.testing.assert_array_equal(
            col[1, :hidden], np.exp2(b2 - 127.0).astype(np.float32)
        )


def _ffn_call(v2, m, hidden, inter, bits, via="v2"):
    kernel, host = _bindings(v2, via), v2.host

    def body(xq, xs, w1, w2, s1, s2, q_out, s_out, tab_vm, col_vm):
        slot = jnp.int32(0)
        kernel.k32z_fill_colscale(s2, slot, col_vm, hidden=hidden, inter=inter)
        w1_block, w2_block = kernel.k32z_block_readers(
            w1,
            w2,
            s1,
            s2,
            tab_vm,
            slot,
            hidden=hidden,
            inter=inter,
            gate_up_bits=bits,
            interpret=True,
        )
        acc2, mid_scale = kernel.expert_ffn_colscale(
            xq[...],
            xs[...],
            w1_block,
            w2_block,
            col_vm[pl.ds(0, 1), pl.ds(0, 2 * inter)],
            hidden=hidden,
            act_fn="silu_and_mul_with_clamp",
        )
        q, s = kernel.wire_row_and_scale(
            acc2,
            mid_scale,
            w2s=col_vm[pl.ds(1, 1), pl.ds(0, hidden)],
            wire_dtype=host.FP8,
        )
        q_out[...] = q
        s_out[...] = jnp.broadcast_to(s, s_out.shape)

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, hidden), host.FP8),
            jax.ShapeDtypeStruct((m, 128), jnp.float32),
        ),
        scratch_shapes=_scratch(v2, hidden, inter, bits),
        interpret=pltpu.InterpretParams(),
    )


def _bf16(a):
    return (
        np.asarray(a, np.float64)
        .astype(np.float32)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float64)
    )


def _fp8(a):
    return (
        np.asarray(a, np.float64)
        .astype(np.float32)
        .astype(ml_dtypes.float8_e4m3fn)
        .astype(np.float64)
    )


def _rowquant(x):
    """rowquant_fp8 on bf16 values (as f64): (q values, scale)."""
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = (amax.astype(np.float32) / np.float32(448.0)).astype(np.float64)
    usable = np.isfinite(scale) & (scale > 0)
    scale = np.where(usable, scale, 0.0)
    sinv = np.where(
        usable,
        (np.float32(1.0) / np.where(usable, scale, 1.0).astype(np.float32)).astype(
            np.float64
        ),
        0.0,
    )
    return np.where(usable, _fp8(_bf16(x * _bf16(sinv))), 0.0), scale


def _replica(xq, xs, w13, w2, inter):
    """FP64 expert at the kernel's rounding points: wire rows (q * scale)."""
    acc1 = (xq @ w13) * xs
    gate = np.minimum(acc1[:, :inter], 10.0)
    up = np.clip(acc1[:, inter:], -10.0, 10.0)
    mid = _bf16(gate / (1.0 + np.exp(-gate)) * up)
    mid_q, mid_s = _rowquant(mid)
    q, s = _rowquant(_bf16((mid_q @ w2) * mid_s))
    return q * s


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("via", ["v2", "push"])
@pytest.mark.parametrize("m", [32, 160])
def test_kernel_expert_ffn(v2, m, via):
    """expert_ffn_colscale on the kernel's readers equals the same math on the
    numpy-decoded K32Z weights, and tracks an FP64 replica with exact K32
    weights at the kernel's rounding points; the push kernel's bindings too."""
    kernel, host = v2.kernel, v2.host
    hidden, inter = SMALL
    w13, e13, w2, e2 = _experts(1, hidden, inter, seed=23)
    ops = _convert(w13, e13, w2, e2)
    rng = np.random.default_rng(m)
    x = rng.standard_normal((m, hidden)).astype(np.float32)
    xq, xs = jax.jit(kernel.rowquant_fp8)(jnp.asarray(x, jnp.bfloat16))
    q, s = jax.jit(_ffn_call(v2, m, hidden, inter, 2, via))(
        xq, xs, *_operands(ops, 0, hidden, inter, 2)
    )
    got = np.asarray(q.astype(jnp.float32), np.float64) * np.asarray(s)[:, :1]

    # The same jnp math on the numpy-decoded weights.
    dec1 = _k32z_e5m2_bytes(w13[0], e13[0]).view(ml_dtypes.float8_e5m2)
    dec2 = _k32z_e5m2_bytes(w2[0], e2[0]).view(ml_dtypes.float8_e5m2)
    _, b1 = _shift_base(e13[0])
    _, b2 = _shift_base(e2[0])
    c1 = jnp.asarray(np.exp2(b1 - 127.0).astype(np.float32)[None])
    c2 = jnp.asarray(np.exp2(b2 - 127.0).astype(np.float32)[None])

    def reader(dec, k):
        blocks = kernel.k32z_blocks(k)

        def block(b, n0, n1):
            k0, rows = blocks[b]
            return jnp.asarray(dec[k0 : k0 + rows, n0:n1])

        return block

    def same_math(xq, xs):
        acc2, mid = kernel.expert_ffn_colscale(
            xq,
            xs,
            reader(dec1, hidden),
            reader(dec2, inter),
            c1,
            hidden=hidden,
            act_fn="silu_and_mul_with_clamp",
        )
        return kernel.wire_row_and_scale(acc2, mid, w2s=c2, wire_dtype=host.FP8)

    q_ref, s_ref = jax.jit(same_math)(xq, xs)
    want = np.asarray(q_ref.astype(jnp.float32), np.float64) * np.asarray(s_ref)
    # Interpreter vs the same ops outside it: XLA may fuse them differently,
    # so FP8 rounding flips are possible but must stay rare.
    print(
        f"[k32z ffn m={m} {via}] interpreter vs same math rel L2 {_rel(got, want):.3e}"
    )
    assert _rel(got, want) < 2e-3, _rel(got, want)
    assert np.all(np.isfinite(got))

    xq64 = np.asarray(xq.astype(jnp.float32), np.float64)
    xs64 = np.asarray(xs, np.float64)
    exact = _replica(
        xq64, xs64, _k32_exact(w13[0], e13[0]), _k32_exact(w2[0], e2[0]), inter
    )
    z = _replica(
        xq64,
        xs64,
        dec1.astype(np.float64) * np.asarray(c1, np.float64),
        dec2.astype(np.float64) * np.asarray(c2, np.float64),
        inter,
    )
    # Recorded bands (FP8 rounding points): kernel vs its own contract in
    # FP64, and vs the exact checkpoint weights.
    print(
        f"[k32z ffn m={m}] vs FP64 K32Z {_rel(got, z):.4f}, "
        f"vs FP64 exact K32 {_rel(got, exact):.4f}"
    )
    # On this host f32 accumulation rarely flips an FP8 rounding (measured:
    # 0 / <= 3e-4 at m = 32 / 160); the chip's FP8 matmul is checked in L1.
    assert _rel(got, z) < 1e-2, _rel(got, z)
    assert _rel(got, exact) < 1e-2, _rel(got, exact)


def _rolled_ffn_call(v2, m, hidden, inter, bits, unroll):
    """_ffn_call's body with the FFN's output chunks walked by a loop
    (k32z_chunk_decoders + expert_ffn_colscale_rolled, host.K32Z_FFN_ROLL)."""
    kernel, host = v2.kernel, v2.host

    def body(xq, xs, w1, w2, s1, s2, q_out, s_out, tab_vm, col_vm, acc_vm):
        slot = jnp.int32(0)
        kernel.k32z_fill_colscale(s2, slot, col_vm, hidden=hidden, inter=inter)
        decode1, decode2 = kernel.k32z_chunk_decoders(
            w1,
            w2,
            s1,
            s2,
            tab_vm,
            slot,
            hidden=hidden,
            inter=inter,
            gate_up_bits=bits,
            interpret=True,
        )
        acc2, mid_scale = kernel.expert_ffn_colscale_rolled(
            xq[...],
            xs[...],
            decode1,
            decode2,
            col_vm[pl.ds(0, 1), pl.ds(0, 2 * inter)],
            acc_vm,
            hidden=hidden,
            act_fn="silu_and_mul_with_clamp",
            unroll=unroll,
        )
        q, s = kernel.wire_row_and_scale(
            acc2,
            mid_scale,
            w2s=col_vm[pl.ds(1, 1), pl.ds(0, hidden)],
            wire_dtype=host.FP8,
        )
        q_out[...] = q
        s_out[...] = jnp.broadcast_to(s, s_out.shape)

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, hidden), host.FP8),
            jax.ShapeDtypeStruct((m, 128), jnp.float32),
        ),
        scratch_shapes=_scratch(v2, hidden, inter, bits)
        + [pltpu.VMEM((m, max(2 * inter, hidden)), jnp.float32)],
        interpret=pltpu.InterpretParams(),
    )


@pytest.mark.parametrize("unroll", [1, 2, 3, (3, 2), (2, 3)])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("m", [32, 160])
def test_rolled_ffn_is_the_unrolled_body_bit_for_bit(v2, m, bits, unroll):
    """The looped K32Z body (host.K32Z_FFN_ROLL) returns the unrolled body's
    wire rows and scales bit for bit: the same block products in the same
    order, with the chunk's column a loop value (traced table field at 2-bit
    gate/up). A (gate/up, down) pair (Step 9, host.K32Z_DECODE_STEP) walks
    each matmul at its own step; an odd step reuses a table half across
    steps."""
    hidden, inter = SMALL
    w13, e13, w2, e2 = _experts(1, hidden, inter, seed=29)
    ops = _convert(w13, e13, w2, e2, gate_up_bits=bits)
    rng = np.random.default_rng(m + bits)
    x = rng.standard_normal((m, hidden)).astype(np.float32)
    xq, xs = jax.jit(v2.kernel.rowquant_fp8)(jnp.asarray(x, jnp.bfloat16))
    operands = _operands(ops, 0, hidden, inter, bits)
    q0, s0 = jax.jit(_ffn_call(v2, m, hidden, inter, bits))(xq, xs, *operands)
    q1, s1 = jax.jit(_rolled_ffn_call(v2, m, hidden, inter, bits, unroll))(
        xq, xs, *operands
    )
    np.testing.assert_array_equal(
        np.asarray(q1).view(np.uint8), np.asarray(q0).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(s1).view(np.uint32), np.asarray(s0).view(np.uint32)
    )


def test_k32z_roll_plan(v2):
    """K32Z_FFN_ROLL None applies K32Z_DECODE_ROLL to decode builds only
    (ragged rows <= ep * g_local * first rung); an explicit roll wins and 0
    forces the unrolled bodies."""
    host = v2.host
    saved = (
        host.K32Z_FFN_ROLL,
        host.K32Z_FFN_ROLL_FROM,
        host.K32Z_DECODE_ROLL,
        host.K32Z_DECODE_STEP,
    )
    ep, g = 8, 48
    decode = host.ragged_stride_bound(8 * 64, 6, 384, 256)  # 64 rows per rank
    prefill = host.ragged_stride_bound(8 * 512, 6, 384, 256)
    try:
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = None, 0
        # The rule's own count (Step 10 made K32Z_DECODE_STEP (3, 4)).
        host.K32Z_DECODE_STEP = None
        host.K32Z_DECODE_ROLL = None
        assert host.k32z_roll_plan(decode, ep, g) == (0, 0)
        host.K32Z_DECODE_ROLL = (1, 64)
        assert host.k32z_roll_plan(decode, ep, g) == (1, 64)
        assert host.k32z_roll_plan(prefill, ep, g) == (0, 0)
        host.K32Z_FFN_ROLL = 0
        assert host.k32z_roll_plan(decode, ep, g) == (0, 0)
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = 2, 128
        assert host.k32z_roll_plan(prefill, ep, g) == (2, 128)
    finally:
        (
            host.K32Z_FFN_ROLL,
            host.K32Z_FFN_ROLL_FROM,
            host.K32Z_DECODE_ROLL,
            host.K32Z_DECODE_STEP,
        ) = saved


def test_k32z_decode_step(v2):
    """K32Z_DECODE_STEP (Step 9): a decode build's looped bodies walk
    (gate/up, down) chunks a step; an equal pair is the plain count (same
    name, same program), and every check names the matmul it fails."""
    host = v2.host
    saved = (
        host.K32Z_FFN_ROLL,
        host.K32Z_FFN_ROLL_FROM,
        host.K32Z_DECODE_ROLL,
        host.K32Z_DECODE_STEP,
    )
    ep, g = 8, 48
    decode = host.ragged_stride_bound(8 * 64, 6, 384, 256)
    prefill = host.ragged_stride_bound(8 * 512, 6, 384, 256)
    try:
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = None, 0
        host.K32Z_DECODE_ROLL = (2, 128)
        host.K32Z_DECODE_STEP = None
        assert host.k32z_roll_plan(decode, ep, g) == (2, 128)
        host.K32Z_DECODE_STEP = (3, 4)
        assert host.k32z_roll_plan(decode, ep, g) == ((3, 4), 128)
        assert host.k32z_roll_plan(prefill, ep, g) == (0, 0)
        host.K32Z_DECODE_STEP = (2, 2)
        assert host.k32z_roll_plan(decode, ep, g) == (2, 128)
    finally:
        (
            host.K32Z_FFN_ROLL,
            host.K32Z_FFN_ROLL_FROM,
            host.K32Z_DECODE_ROLL,
            host.K32Z_DECODE_STEP,
        ) = saved
    assert host.check_k32z_ffn_roll((3, 4), *SERVED) == (3, 4)
    assert host.check_k32z_ffn_roll((2, 2), *SERVED) == 2
    assert host.check_k32z_ffn_roll((9, 10), *SERVED) == (9, 10)
    for bad in ((0, 4), (3,), (3, 4, 5), (1.5, 2)):
        with pytest.raises(ValueError, match="chunk count"):
            host.check_k32z_ffn_roll(bad, *SERVED)
    with pytest.raises(ValueError, match="gate/up"):
        host.check_k32z_ffn_roll((4, 4), *SERVED)
    with pytest.raises(ValueError, match="down"):
        host.check_k32z_ffn_roll((3, 3), *SERVED)
    assert host.k32z_roll_steps(2) == (2, 2)
    assert host.k32z_roll_steps((3, 4)) == (3, 4)
    assert host.k32z_roll_tag(2) == "2"
    assert host.k32z_roll_tag((3, 4)) == "3x4"


def test_k32z_decode_step_default(v2):
    """Step 10 L1: K32Z_DECODE_STEP is (3, 4) by default. A decode build at
    V4.1's shape takes it; a shape whose chunk counts it does not divide
    takes K32Z_DECODE_ROLL's count; a prefill build loops nothing."""
    host = v2.host
    assert host.K32Z_DECODE_STEP == (3, 4)
    saved = host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM, host.K32Z_DECODE_ROLL
    ep, g = 8, 48
    decode = host.ragged_stride_bound(8 * 64, 6, 384, 256)
    prefill = host.ragged_stride_bound(8 * 512, 6, 384, 256)
    try:
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = None, 0
        host.K32Z_DECODE_ROLL = (2, 128)
        assert host.k32z_roll_plan(decode, ep, g, *SERVED) == ((3, 4), 128)
        assert host.k32z_roll_plan(decode, ep, g) == ((3, 4), 128)
        # Six output chunks on both matmuls: 4 does not divide them.
        assert host.k32z_roll_plan(decode, ep, g, 1536, 768) == (2, 128)
        assert host.k32z_roll_plan(prefill, ep, g, *SERVED) == (0, 0)
        plan = host.k32z_build_plan(decode, ep, g, *SERVED)
        assert plan["k32z_ffn_roll"] == (3, 4)
        assert host.k32z_build_plan(decode, ep, g, 1536, 768)["k32z_ffn_roll"] == 2
    finally:
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM, host.K32Z_DECODE_ROLL = saved
    assert not host.k32z_step_applies((3, 4), 1536, 768)
    assert host.k32z_step_applies((3, 4), *SERVED)
    assert host.k32z_step_applies((0, 4), *SERVED)  # refused later, by name


@pytest.mark.parametrize("dispatch", ["all_gather", "push"])
def test_k32z_decode_step_default_lowers(v2, dispatch):
    """Step 10 L1: a default decode build whose chunk counts (3, 4) divides
    (hidden 2048: 8 down chunks; inter 768: 6 gate/up chunks) lowers under
    the `_r3x4` name; at 1536 / 768 the default keeps the `_r2` body."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    names = _kernel_names if dispatch == "all_gather" else _push_names
    saved = host.K32Z_FFN_ROLL, host.K32Z_DECODE_STEP
    try:
        host.K32Z_FFN_ROLL = None
        (fits,) = names(
            _lower_layer(v2, weight_format=k32z, dispatch=dispatch, hidden=2048)
        )
        (small,) = names(_lower_layer(v2, weight_format=k32z, dispatch=dispatch))
        host.K32Z_DECODE_STEP = None
        (fits_off,) = names(
            _lower_layer(v2, weight_format=k32z, dispatch=dispatch, hidden=2048)
        )
    finally:
        host.K32Z_FFN_ROLL, host.K32Z_DECODE_STEP = saved
    assert "_r3x4f128" in fits, fits
    assert "_r2f128" in small, small
    assert fits_off == fits.replace("_r3x4f128", "_r2f128"), (fits, fits_off)


def test_rolled_ffn_knob(v2):
    """K32Z_FFN_ROLL's checks, its scratch and VMEM accounting, and the
    kernel name: 0 is the unrolled N2a body and changes nothing."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    assert host.check_k32z_ffn_roll(0, *SERVED) == 0
    assert host.check_k32z_ffn_roll(2, *SERVED) == 2
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="chunk count"):
            host.check_k32z_ffn_roll(bad, *SERVED)
    # 18 gate/up chunks, 20 down chunks at the served shape.
    with pytest.raises(ValueError, match="gate/up"):
        host.check_k32z_ffn_roll(4, *SERVED)
    with pytest.raises(ValueError, match="down"):
        host.check_k32z_ffn_roll(3, *SERVED)
    base = host.vmem_scratch_arrays(48, 256, *SERVED, nbuf=2, weight_format=k32z)
    rolled = host.vmem_scratch_arrays(
        48, 256, *SERVED, nbuf=2, weight_format=k32z, k32z_ffn_roll=2
    )
    assert [n for n, _, _ in rolled] == [n for n, _, _ in base][:7] + ["acc_vm"] + [
        n for n, _, _ in base
    ][7:]
    assert dict((n, (s, jnp.dtype(d).name)) for n, s, d in rolled)["acc_vm"] == (
        (256, 5120),
        "float32",
    )
    body = [
        n
        for n, _, _ in host.vmem_tile_body_arrays(
            256, *SERVED, weight_format=k32z, k32z_ffn_roll=2
        )
    ]
    assert "acc1" not in body and "acc2" in body
    # FP4 ignores the knob.
    fp4 = host.WeightFormat.FP4
    assert host.vmem_scratch_arrays(
        48, 128, *SERVED, nbuf=2, weight_format=fp4, k32z_ffn_roll=2
    ) == host.vmem_scratch_arrays(48, 128, *SERVED, nbuf=2, weight_format=fp4)


# --------------------------------------------------------------------------- #
# The layer lowers for TPU; the ablation key.
# --------------------------------------------------------------------------- #
def _lower_layer(
    v2,
    *,
    weight_format,
    ablation=None,
    stage="fused",
    hidden=1536,
    inter=768,
    ep=2,
    g_local=8,
    t_local=16,
    dispatch="all_gather",
    push_options=None,
    bits=2,
    logical=None,
):
    host = v2.host
    mesh = Mesh(np.asarray(jax.devices()[:ep]), ("d",))
    rows, repl = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    e_total = ep * g_local
    t = ep * t_local
    if weight_format == host.WeightFormat.K32Z:
        geo = host.k32z_table_geometry(hidden, inter, bits)
        s1 = jax.ShapeDtypeStruct(
            (e_total * geo.rows1, geo.cols1), jnp.uint8, sharding=rows
        )
        s2 = jax.ShapeDtypeStruct(
            (e_total * geo.rows2, geo.cols2), jnp.uint8, sharding=rows
        )
        cap = 256
    else:
        r1 = host.fp4_scale_rows(hidden, 512, padded=True)
        r2 = host.fp4_scale_rows(inter, 512, padded=True)
        s1 = jax.ShapeDtypeStruct((e_total * r1, 2 * inter), jnp.float32, sharding=rows)
        s2 = jax.ShapeDtypeStruct((e_total * r2, hidden), jnp.float32, sharding=rows)
        cap = 128
    w1 = jax.ShapeDtypeStruct(
        (e_total, hidden, 2 * inter), jnp.float4_e2m1fn, sharding=rows
    )
    w2 = jax.ShapeDtypeStruct(
        (e_total, inter, hidden), jnp.float4_e2m1fn, sharding=rows
    )
    x = jax.ShapeDtypeStruct((t, hidden), jnp.bfloat16, sharding=rows)
    g = jax.ShapeDtypeStruct((t, e_total), jnp.float32, sharding=rows)
    bias = jax.ShapeDtypeStruct((e_total,), jnp.float32, sharding=repl)
    kw = {} if ablation is None else {"ablation": ablation}
    if dispatch == "push":
        kw.update(dispatch="push", push_options=dict(push_options or {}))

    def layer(x, g, w1, w2, s1, s2, bias):
        return v2.package.fused_ep_moe_v2(
            x,
            w1,
            w2,
            s1,
            s2,
            g,
            topk=6,
            renormalize=True,
            mesh=mesh,
            capacity=cap,
            weight_format=weight_format,
            rhs_qb=512,
            act_fn="silu_and_mul_with_clamp",
            mesh_ep_ranks=tuple(range(ep)),
            sharded_plan=True,
            scoring_fn="sqrtsoftplus",
            score_bias=bias,
            routed_scaling_factor=1.5,
            logical_intermediate_size=logical,
            stage=stage,
            **kw,
        )

    with _tpu7x_target():
        return (
            jax.jit(layer)
            .trace(x, g, w1, w2, s1, s2, bias)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )


def _kernel_names(text):
    return sorted(set(re.findall(r"(fused_ep_moe_v2\w*)", text)))


def test_k32z_layer_lowers_and_empty_ablation_is_production(v2):
    k32z = v2.host.WeightFormat.K32Z
    base = _lower_layer(v2, weight_format=k32z)
    (name,) = _kernel_names(base)
    assert re.fullmatch(
        r"fused_ep_moe_v2_k32zw_silu_and_mul_with_clamp_g8_c256_nb\d(_r\d+f\d+)?"
        r"(_mu\d+)?",
        name,
    ), name
    assert _lower_layer(v2, weight_format=k32z, ablation=frozenset()) == base
    ablated = _lower_layer(
        v2, weight_format=k32z, ablation=frozenset({"ffn_off", "w_off"})
    )
    assert _kernel_names(ablated) == [name + "_ffn_off_w_off"]
    assert ablated != base


def test_fp4_empty_ablation_is_production(v2):
    fp4 = v2.host.WeightFormat.FP4
    base = _lower_layer(v2, weight_format=fp4, inter=1024)
    assert _lower_layer(v2, weight_format=fp4, inter=1024, ablation=frozenset()) == base


def test_layer_refusals(v2):
    host = v2.host
    k32z = host.WeightFormat.K32Z
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("d",))
    hidden, inter = SMALL
    geo = host.k32z_table_geometry(hidden, inter, 2)
    x = jnp.zeros((4, hidden), jnp.bfloat16)
    g = jnp.zeros((4, 4), jnp.float32)
    w1 = jnp.zeros((4, hidden, 2 * inter), jnp.float4_e2m1fn)
    w2 = jnp.zeros((4, inter, hidden), jnp.float4_e2m1fn)
    s1 = jnp.zeros((4 * geo.rows1, geo.cols1), jnp.uint8)
    s2 = jnp.zeros((4 * geo.rows2, geo.cols2), jnp.uint8)
    call = partial(
        v2.package.fused_ep_moe_v2,
        topk=2,
        renormalize=True,
        mesh=mesh,
        capacity=256,
        weight_format=k32z,
    )
    with pytest.raises(ValueError, match="disp_off"):
        call(x, w1, w2, s1, s2, g, ablation={"disp_off"})
    with pytest.raises(ValueError, match="flat K32Z shift table"):
        call(x, w1, w2, s1.astype(jnp.int8), s2, g)
    with pytest.raises(ValueError, match="flat K32Z shift table"):
        call(x, w1, w2, s1, s2[:-32], g)
    with pytest.raises(ValueError, match="bits wide"):
        call(
            x, w1, w2, jnp.zeros((4 * geo.rows1, 3 * geo.cols1 // 2), jnp.uint8), s2, g
        )
    with pytest.raises(ValueError, match="float4_e2m1fn expert weights"):
        call(x, w1.astype(jnp.float8_e4m3fn), w2, s1, s2, g)


def test_make_fused_ep_moe_expert_format():
    seen = []

    def fake(*args, **kw):
        seen.append(kw)
        return args[0]

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("ep",))
    args = (jnp.zeros((1, 8)), jnp.zeros((1, 4)), None, None, None, None, jnp.zeros(4))
    fe.make_fused_ep_moe(mesh, kernel_fn=fake, jit=False)(*args)
    fe.make_fused_ep_moe(
        mesh, kernel_fn=fake, jit=False, expert_format="k32z", ablation={"ret_off"}
    )(*args)
    assert (str(seen[0]["weight_format"]), seen[0]["capacity"]) == ("fp4", 128)
    assert "ablation" not in seen[0]
    assert (str(seen[1]["weight_format"]), seen[1]["capacity"]) == ("k32z", 256)
    assert seen[1]["ablation"] == frozenset({"ret_off"})
    with pytest.raises(ValueError, match="expert_format"):
        fe.make_fused_ep_moe(mesh, kernel_fn=fake, jit=False, expert_format="k64")


# --------------------------------------------------------------------------- #
# The push dispatch (D+Z, Step 7 N2b).
# --------------------------------------------------------------------------- #
def _tuned_push_options(g_local):
    """#2102's tuned push bundle (config D), as the dispatch x routing
    harness copies it from test_fused_ep_moe_push.py."""
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


def _push_names(text):
    return sorted(set(re.findall(r"(push_fused_moe\w*)", text)))


def test_make_fused_ep_moe_dispatch(v2):
    """make_fused_ep_moe's dispatch (the serving switch): the default calls the
    kernel with the same keywords as before push was wired in; push takes
    #2102's tuned bundle sized to the operands' local experts, with either
    expert format; auto decides on each traced shape's rows per EP rank, the
    bound inclusive; what cannot run is refused when the layer is built."""
    if len(jax.devices()) < 2:
        pytest.skip("needs two CPU devices: run this file in its own process")
    seen = []

    def fake(*args, **kw):
        seen.append(kw)
        return args[0]

    mesh = Mesh(np.asarray(jax.devices()[:2]), ("ep",))
    g_local = 4
    w = jnp.zeros((2 * g_local, 1, 1))
    bias = jnp.zeros(2 * g_local)

    def call(moe, rows_per_rank):
        t = 2 * rows_per_rank
        moe(jnp.zeros((t, 8)), jnp.zeros((t, 2 * g_local)), w, w, None, None, bias)
        return seen[-1]

    default = call(fe.make_fused_ep_moe(mesh, kernel_fn=fake, jit=False), 8)
    assert sorted(default) == [
        "act_fn",
        "capacity",
        "dispatch",
        "logical_intermediate_size",
        "mesh",
        "mesh_ep_ranks",
        "preprocess_topology",
        "rank",
        "renormalize",
        "rhs_qb",
        "routed_scaling_factor",
        "score_bias",
        "scoring_fn",
        "sharded_plan",
        "topk",
        "weight_format",
    ]
    assert default["dispatch"] == "all_gather"
    for fmt, capacity in (("k512", 128), ("k32z", 256)):
        push = fe.make_fused_ep_moe(
            mesh, kernel_fn=fake, jit=False, expert_format=fmt, dispatch="push"
        )
        kw = call(push, 2048)
        assert (kw["dispatch"], kw["capacity"]) == ("push", capacity)
        assert kw["push_options"] == _tuned_push_options(g_local)
    auto = fe.make_fused_ep_moe(
        mesh, kernel_fn=fake, jit=False, dispatch="auto", push_max_rows=16
    )
    assert [call(auto, rows)["dispatch"] for rows in (8, 16, 32)] == [
        "push",
        "push",
        "all_gather",
    ]
    assert "push_options" not in seen[-1]
    own = fe.make_fused_ep_moe(
        mesh, kernel_fn=fake, jit=False, dispatch="auto", push_options={}
    )
    assert call(own, 8)["push_options"] == {}
    for kw, match in (
        (dict(dispatch="pull"), "dispatch must be one of"),
        (dict(push_options={}), "push_options need dispatch"),
        (dict(dispatch="auto", push_max_rows=0), "push_max_rows must be >= 1"),
    ):
        with pytest.raises(ValueError, match=match):
            fe.make_fused_ep_moe(mesh, kernel_fn=fake, jit=False, **kw)


def test_push_binds_v2s_k32z_compute(v2):
    """The push kernel runs v2's K32Z decode, FFN, column scales and ablation
    names, not copies (the interpreter tests above run them through it)."""
    for name in (
        "k32z_block_readers",
        "k32z_fill_colscale",
        "expert_ffn_colscale",
        "wire_row_and_scale",
        "check_ablation",
        "ablation_tag",
    ):
        assert getattr(v2.push_kernel, name) is getattr(v2.kernel, name), name


@pytest.mark.parametrize(
    "kw,what",
    [
        (dict(k32z_gate_up_bits=3), "bits wide"),
        (dict(hidden=5000), "multiples of 256"),
        (dict(capacity=240), "rungs"),
        (dict(height_rungs_override=2), "height_rungs_override"),
        (dict(logical_intermediate_size=2048), "logical intermediate"),
        (dict(ablation=frozenset({"disp_off"})), "sc_prepush"),
        (dict(ablation=frozenset({"bogus"})), "unknown ablation"),
        (dict(ablation="w_off"), "string"),
    ],
)
def test_push_build_refusals(v2, kw, what):
    args = dict(
        g_local=48,
        capacity=256,
        hidden=5120,
        inter=2304,
        ep=8,
        weight_format=v2.host.WeightFormat.K32Z,
        ragged_rows_alloc=4096,
        act_fn="silu_and_mul_with_clamp",
        push_tokens=64,
        sc_prepush=2,
    )
    args.update(kw)
    with pytest.raises(ValueError, match=what):
        v2.push_kernel._build_push_moe_kernel(**args)


def test_push_layer_refusals(v2):
    host = v2.host
    k32z = host.WeightFormat.K32Z
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("d",))
    hidden, inter = SMALL
    geo = host.k32z_table_geometry(hidden, inter, 2)
    x = jnp.zeros((4, hidden), jnp.bfloat16)
    g = jnp.zeros((4, 4), jnp.float32)
    w1 = jnp.zeros((4, hidden, 2 * inter), jnp.float4_e2m1fn)
    w2 = jnp.zeros((4, inter, hidden), jnp.float4_e2m1fn)
    s1 = jnp.zeros((4 * geo.rows1, geo.cols1), jnp.uint8)
    s2 = jnp.zeros((4 * geo.rows2, geo.cols2), jnp.uint8)
    call = partial(
        v2.package.fused_ep_moe_v2,
        topk=2,
        renormalize=True,
        mesh=mesh,
        capacity=256,
        weight_format=k32z,
        dispatch="push",
    )
    with pytest.raises(ValueError, match="flat K32Z shift table"):
        call(x, w1, w2, s1.astype(jnp.int8), s2, g, push_options={})
    with pytest.raises(ValueError, match="flat K32Z shift table"):
        call(x, w1, w2, s1, s2[:-32], g, push_options={})
    with pytest.raises(ValueError, match="bits wide"):
        call(
            x,
            w1,
            w2,
            jnp.zeros((4 * geo.rows1, 3 * geo.cols1 // 2), jnp.uint8),
            s2,
            g,
            push_options={},
        )
    with pytest.raises(ValueError, match="float4_e2m1fn expert weights"):
        call(x, w1.astype(jnp.float8_e4m3fn), w2, s1, s2, g, push_options={})
    with pytest.raises(ValueError, match="height_rungs_override"):
        call(x, w1, w2, s1, s2, g, push_options={"height_rungs_override": 2})
    with pytest.raises(ValueError, match="unknown ablation"):
        call(x, w1, w2, s1, s2, g, push_options={}, ablation={"comp"})


@pytest.mark.parametrize("bundle", ["defaults", "tuned"])
def test_push_k32z_layer_lowers_and_empty_ablation_is_production(v2, bundle):
    """The push layer lowers for TPU with K32Z on the push defaults (tiles
    shipped from VMEM, 3-D mirrors on the wire) and on the tuned bundle (HBM
    staging: commits and two-hop pushes of 3-D mirrors). An empty ablation is
    the production program; a non-empty one renames it; disp_off pre-pushes
    every visit position on the SparseCore."""
    k32z = v2.host.WeightFormat.K32Z
    options = {} if bundle == "defaults" else _tuned_push_options(8)
    lower = partial(
        _lower_layer, v2, weight_format=k32z, dispatch="push", push_options=options
    )
    base = lower()
    (name,) = _push_names(base)
    tail = "" if bundle == "defaults" else "_notail_hbmst_da8_1b"
    assert re.fullmatch(
        rf"push_fused_moe_k32zw_silu_and_mul_with_clamp_g8_c256_nb\d_s2{tail}"
        r"(_r\d+f\d+)?(_mu\d+)?",
        name,
    ), name
    assert lower(ablation=frozenset()) == base
    ablated = lower(ablation=frozenset({"ffn_off", "w_off"}))
    assert _push_names(ablated) == [name + "_ffn_off_w_off"]
    assert ablated != base
    comp = lower(ablation=frozenset({"disp_off", "ret_off"}))
    assert _push_names(comp) == [name.replace("_s2", "_s8") + "_disp_off_ret_off"]


@pytest.mark.parametrize("dispatch", ["all_gather", "push"])
def test_k32z_rolled_layer_lowers(v2, dispatch):
    """host.K32Z_FFN_ROLL is read when the layer is traced: a nonzero roll
    lowers for TPU under its own kernel name; 0 is the unrolled program."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    names = _kernel_names if dispatch == "all_gather" else _push_names
    lower = partial(_lower_layer, v2, weight_format=k32z, dispatch=dispatch)
    saved = host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM, host.K32Z_MIRROR_UNIT
    try:
        # The roll's own names, on the tile-sublane mirror.
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = 0, 0
        host.K32Z_MIRROR_UNIT = 0
        base = lower()
        host.K32Z_FFN_ROLL = 2
        rolled = lower()
        # Only the rungs from 64 up loop; 32 keeps the unrolled body.
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM = 1, 64
        mixed = lower()
        # A threshold without a roll is no roll.
        host.K32Z_FFN_ROLL = 0
        unrolled = lower()
    finally:
        host.K32Z_FFN_ROLL, host.K32Z_FFN_ROLL_FROM, host.K32Z_MIRROR_UNIT = saved
    (name,) = names(base)
    assert names(rolled) == [name + "_r2"], names(rolled)
    assert rolled != base
    assert names(mixed) == [name + "_r1f64"], names(mixed)
    assert mixed not in (base, rolled)
    assert unrolled == base


@pytest.mark.parametrize(
    "dispatch,bundle",
    [("all_gather", None), ("push", "defaults"), ("push", "tuned")],
)
def test_k32z_mirror_unit_layer_lowers(v2, dispatch, bundle):
    """host.K32Z_MIRROR_UNIT = 128 lowers for TPU under its own kernel name
    (with and without the looped body); 0 is the program as before."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    names = _kernel_names if dispatch == "all_gather" else _push_names
    options = _tuned_push_options(8) if bundle == "tuned" else None
    lower = partial(
        _lower_layer, v2, weight_format=k32z, dispatch=dispatch, push_options=options
    )
    saved = host.K32Z_MIRROR_UNIT, host.K32Z_FFN_ROLL
    try:
        host.K32Z_MIRROR_UNIT, host.K32Z_FFN_ROLL = 0, 0
        base = lower()
        host.K32Z_MIRROR_UNIT = 128
        unit = lower()
        host.K32Z_FFN_ROLL = 2
        both = lower()
    finally:
        host.K32Z_MIRROR_UNIT, host.K32Z_FFN_ROLL = saved
    (name,) = names(base)
    assert names(unit) == [name + "_mu128"], names(unit)
    assert names(both) == [name + "_r2_mu128"], names(both)


@pytest.mark.parametrize("dispatch", ["all_gather", "push"])
def test_k32z_rung_ladder_knob(v2, dispatch):
    """host.K32Z_RUNGS other than the default ladder is traced into the build
    (k32z_rungs, named _q...); malformed ladders are refused."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    names = _kernel_names if dispatch == "all_gather" else _push_names
    lower = partial(_lower_layer, v2, weight_format=k32z, dispatch=dispatch)
    saved = host.K32Z_RUNGS
    try:
        base = lower()
        host.K32Z_RUNGS = (32, 64, 96, 128)
        q96 = lower()
    finally:
        host.K32Z_RUNGS = saved
    (name,) = names(base)
    assert names(q96) == [name + "_q32_64_96_128"], names(q96)
    assert host.k32z_tile_height_rungs(256, (32, 64, 96, 128)) == (32, 64, 96, 128, 256)
    # A ladder without 128 is a ladder (Step 7 R2): it streams up to its
    # tallest rung below 128, and only a ladder of tile_m alone streams none.
    assert host.k32z_tile_height_rungs(256, (32, 64)) == (32, 64, 256)
    assert host.k32z_streamed_rows((32, 64, 128, 256)) == host.K32Z_STREAMED_ROWS
    assert host.k32z_streamed_rows((32, 256)) == 32
    assert host.k32z_streamed_rows((256,)) == 0
    for bad in ((64, 32, 128), (32, 48, 128), (0, 32, 128)):
        with pytest.raises(ValueError, match="rungs"):
            host.k32z_tile_height_rungs(256, bad)


# Step 7 R2 (K1) knobs, read by host.k32z_build_plan at trace time.
_R2_KNOBS = {
    "K32Z_FFN_ROLL": None,
    "K32Z_FFN_ROLL_FROM": 0,
    "K32Z_ROLL_RUNGS": None,
    "K32Z_LARGE_PLAN": None,
    "PUSH_FROM_VMEM_ROWS": None,
    "DEFER_PUSH": False,
}


@contextlib.contextmanager
def _knobs(host, **values):
    """The R2 knobs at their defaults, then `values`; restored on exit."""
    names = set(_R2_KNOBS) | set(values)
    saved = {n: getattr(host, n) for n in names}
    try:
        for n, v in {**_R2_KNOBS, **values}.items():
            setattr(host, n, v)
        yield
    finally:
        for n, v in saved.items():
            setattr(host, n, v)


def test_k32z_build_plan(v2):
    """host.k32z_build_plan: at the defaults it is the R1 build (the
    builders' defaults plus k32z_roll_plan), each R2 knob reaches its builder
    argument, and K32Z_LARGE_PLAN applies to large builds only."""
    host = v2.host
    ep, g = 8, 48
    rows = {b: host.ragged_stride_bound(8 * b, 6, 384, 256) for b in (8, 64, 512, 1024)}
    rows2048 = host.ragged_stride_bound(8 * 2048, 6, 384, 256)
    kept = host.K32Z_LARGE_PLAN  # the module default: Step 7 R2's kept tiers

    def r1(r):
        roll, roll_from = host.k32z_roll_plan(r, ep, g)
        plan = dict(host.K32Z_PLAN_DEFAULTS)
        plan["k32z_ffn_roll"], plan["k32z_roll_from"] = roll, roll_from
        return plan

    with _knobs(host):
        for r in rows.values():
            assert host.k32z_build_plan(r, ep, g) == r1(r)
    with _knobs(
        host,
        K32Z_FFN_ROLL=2,
        K32Z_ROLL_RUNGS=(64, 32),
        PUSH_FROM_VMEM_ROWS=128,
        DEFER_PUSH=True,
    ):
        plan = host.k32z_build_plan(rows[1024], ep, g)
    assert plan["k32z_ffn_roll"] == 2 and plan["k32z_roll_rungs"] == (32, 64)
    assert plan["push_from_vmem_rows"] == 128 and plan["defer_push"]
    large = {"rungs": (32, 64), "roll": (2, (32,)), "defer_push": True}
    with _knobs(host, K32Z_LARGE_PLAN=large):
        # 1024 rows per rank is the first large bucket (> 128 rows/expert).
        assert host.k32z_build_plan(rows[512], ep, g) == r1(rows[512])
        plan = host.k32z_build_plan(rows[1024], ep, g)
    assert plan["k32z_rungs"] == (32, 64) and plan["k32z_ffn_roll"] == 2
    assert plan["k32z_roll_rungs"] == (32,) and plan["defer_push"]
    with _knobs(host, K32Z_LARGE_PLAN={"rungs": host.K32Z_DEFAULT_RUNGS}):
        assert host.k32z_build_plan(rows[1024], ep, g)["k32z_rungs"] is None
    # Tiers: the tallest tier a build exceeds (rows per expert) applies.
    tiers = ((128, {"defer_push": True}), (256, {"rungs": (32, 128)}))
    with _knobs(host, K32Z_LARGE_PLAN=tiers):
        assert host.k32z_build_plan(rows[512], ep, g) == r1(rows[512])
        mid = host.k32z_build_plan(rows[1024], ep, g)
        top = host.k32z_build_plan(rows2048, ep, g)
    assert mid["defer_push"] and mid["k32z_rungs"] is None
    assert top["k32z_rungs"] == (32, 128) and not top["defer_push"]
    with _knobs(host, K32Z_LARGE_PLAN={"bogus": 1}):
        with pytest.raises(ValueError, match="K32Z_LARGE_PLAN"):
            host.k32z_build_plan(rows[1024], ep, g)
    # The kept defaults: decode and 512 rows per rank build as in k32z-r1;
    # 1024 drops the 32 rung and takes the two-hop return with the deferred
    # push; 2048 drops the 64 rung and defers the push.
    with _knobs(host, K32Z_LARGE_PLAN=kept):
        for b in (8, 64, 512):
            assert host.k32z_build_plan(rows[b], ep, g) == r1(rows[b])
        mid = host.k32z_build_plan(rows[1024], ep, g)
        top = host.k32z_build_plan(rows2048, ep, g)
    assert (mid["k32z_rungs"], mid["push_from_vmem_rows"], mid["defer_push"]) == (
        (64, 128),
        128,
        True,
    )
    assert (top["k32z_rungs"], top["push_from_vmem_rows"], top["defer_push"]) == (
        (32, 128),
        None,
        True,
    )


def _with_after(name, after, tag):
    """`name` with `tag` inserted right after its `after` match."""
    m = re.search(after, name)
    assert m, (name, after)
    return name[: m.end()] + tag + name[m.end() :]


# (knobs, rows per rank, where the tag goes, tag). ep 2 x 8 local experts at
# 256-row tiles: 16 rows per rank is a decode build (the default looped
# rungs, _r2f64) shipping from VMEM under either bound, 256 ships from VMEM
# by tile_m but not by 128, 512 takes the two-hop path.
_R2_LOWERINGS = [
    (dict(K32Z_FFN_ROLL=2, K32Z_ROLL_RUNGS=(32, 64)), 512, r"_nb\d+", "_r2s32_64"),
    (dict(PUSH_FROM_VMEM_ROWS=128), 256, r"$", "_pv128"),
    (dict(DEFER_PUSH=True), 512, r"$", "_dp"),
    (dict(PUSH_FROM_VMEM_ROWS=128, DEFER_PUSH=True), 256, r"$", "_pv128_dp"),
]


def _any_nbuf(name):
    """`name` with its weight-slot count masked: a looped body's smaller
    tile-body VMEM can leave room for another slot."""
    return re.sub(r"_nb\d+", "_nb#", name)


@pytest.mark.parametrize("knobs,t_local,after,tag", _R2_LOWERINGS)
def test_k32z_r2_knobs_lower(v2, knobs, t_local, after, tag):
    """Each Step 7 R2 knob lowers for TPU under its own kernel name where it
    changes the build (the K32Z all-gather layer reads them at trace time)."""
    host = v2.host
    lower = partial(
        _lower_layer, v2, weight_format=host.WeightFormat.K32Z, t_local=t_local
    )
    with _knobs(host):
        base = lower()
    with _knobs(host, **knobs):
        got = lower()
    (name,) = _kernel_names(base)
    (got_name,) = _kernel_names(got)
    assert _any_nbuf(got_name) == _any_nbuf(_with_after(name, after, tag)), got_name
    assert got != base


@pytest.mark.parametrize(
    "knobs,t_local",
    [
        # Both paths already agree with a 128-row bound here.
        (dict(PUSH_FROM_VMEM_ROWS=128), 16),
        (dict(PUSH_FROM_VMEM_ROWS=128), 512),
        # Push-from-VMEM builds have no deferred drain.
        (dict(DEFER_PUSH=True), 16),
        # A roll set without a roll loops nothing (512: not a decode build,
        # so no K32Z_DECODE_ROLL either).
        (dict(K32Z_ROLL_RUNGS=(32,)), 512),
        # A large plan below the large bound.
        (dict(K32Z_LARGE_PLAN={"rungs": (32,), "defer_push": True}), 16),
    ],
)
def test_k32z_r2_knobs_that_change_nothing(v2, knobs, t_local):
    """Where an R2 knob does not change the build, the program is the
    default one byte for byte (same name, same text)."""
    host = v2.host
    lower = partial(
        _lower_layer, v2, weight_format=host.WeightFormat.K32Z, t_local=t_local
    )
    with _knobs(host):
        base = lower()
    with _knobs(host, **knobs):
        assert lower() == base


@pytest.mark.parametrize("dispatch", ["all_gather", "push"])
def test_r2_knobs_leave_k512_alone(v2, dispatch):
    """The R2 knobs are K32Z build choices: every K512 program lowers byte
    for byte as at their defaults, on both dispatches."""
    host = v2.host
    fp4 = host.WeightFormat.FP4
    options = _tuned_push_options(8) if dispatch == "push" else None
    lower = partial(
        _lower_layer,
        v2,
        weight_format=fp4,
        inter=1024,
        dispatch=dispatch,
        push_options=options,
        t_local=512,
    )
    with _knobs(host):
        base = lower()
    every = dict(
        K32Z_FFN_ROLL=2,
        K32Z_ROLL_RUNGS=(32,),
        K32Z_LARGE_PLAN={"rungs": (32,), "defer_push": True},
        PUSH_FROM_VMEM_ROWS=128,
        DEFER_PUSH=True,
    )
    with _knobs(host, **every):
        assert lower() == base


@pytest.mark.parametrize("rungs", [(32,), (32, 64), ()])
def test_k32z_ladders_without_128_lower(v2, rungs):
    """A ladder without 128 lowers (it streams up to its tallest rung below
    128, or nothing for tile_m alone) on the all-gather and push layers."""
    host = v2.host
    k32z = host.WeightFormat.K32Z
    saved = host.K32Z_RUNGS
    try:
        host.K32Z_RUNGS = rungs
        # No large plan: its tiers set their own ladders at 512 rows here.
        with _knobs(host):
            for dispatch, names in (
                ("all_gather", _kernel_names),
                ("push", _push_names),
            ):
                text = _lower_layer(
                    v2, weight_format=k32z, dispatch=dispatch, t_local=512
                )
                (name,) = names(text)
                assert name.endswith("_q" + "_".join(map(str, rungs))), name
    finally:
        host.K32Z_RUNGS = saved


@pytest.mark.parametrize("mirror_m", [256, 128])
def test_mirror_unit_positions(v2, mirror_m):
    """Below 256-row tiles a 128-lane mirror unit keeps every pair's mirror
    position inside the arrival mirror the kernel declares, one position per
    arrival row on each destination; the arrival rows do not move."""
    host = v2.host
    ep, t_local, g_local, topk, tile_m = 4, 64, 8, 6, 256
    e_total, T = ep * g_local, ep * t_local
    rng = np.random.default_rng(3)
    idx = jnp.asarray(rng.integers(0, e_total, (T, topk)).astype(np.int32))
    kw = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=tile_m,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, tile_m),
    )
    base = host.build_routing_tables(idx, **kw)
    got = host.build_routing_tables(idx, mirror_m=mirror_m, **kw)
    np.testing.assert_array_equal(
        np.asarray(got.arrival_row), np.asarray(base.arrival_row)
    )
    recv_rows = host.align_up(t_local * topk + (host.ROWBLK - 1) * e_total, host.ROWBLK)
    bound = host.arrival_mirror_rows(recv_rows, e_total, mirror_m) * mirror_m
    pos = np.asarray(got.arrival_row).reshape(ep, t_local * topk)
    mir = np.asarray(got.mirror_row).reshape(ep, t_local * topk)
    for d in range(ep):
        assert 0 <= mir[d].min() and mir[d].max() < bound
        assert len(set(mir[d])) == len(set(pos[d])) == len(set(zip(pos[d], mir[d])))
    assert host.mirror_lanes(256, 128) == 128 and host.mirror_lanes(128, 128) == 128
    assert host.mirror_lanes(256, 0) == 256
    unit = host.vmem_scratch_arrays(
        48, 256, *SERVED, nbuf=2, weight_format=host.WeightFormat.K32Z, mirror_unit=128
    )
    assert dict((n, s) for n, s, _ in unit)["oscl_vm"] == (2, 2, 1, 128)


def test_push_fp4_empty_ablation_is_production(v2):
    fp4 = v2.host.WeightFormat.FP4
    lower = partial(
        _lower_layer,
        v2,
        weight_format=fp4,
        inter=1024,
        dispatch="push",
        push_options=_tuned_push_options(8),
    )
    base = lower()
    (name,) = _push_names(base)
    assert name.startswith("push_fused_moe_fp4w_silu_and_mul_with_clamp_g8_c128_"), name
    assert lower(ablation=frozenset()) == base
    assert _push_names(lower(ablation=frozenset({"ret_off"}))) == [name + "_ret_off"]


@pytest.mark.parametrize("rows", [8, 2048], ids=["decode8", "prefill2048"])
def test_push_k32z_served_shapes_lower(v2, rows):
    """The served V4.1 Flash layer (EP8, 48 local experts of 5120 x 2304,
    4-bit gate/up fields) lowers for TPU on the tuned push bundle at a decode
    and a prefill bucket, as does K512 (inter 2560, logical 2304). Its
    256-row scale mirrors take v2's combine at the prefill bucket."""
    if len(jax.devices()) < 8:
        pytest.skip("needs eight CPU devices: run this file in its own process")
    host = v2.host
    served = dict(ep=8, g_local=48, t_local=rows, dispatch="push")
    options = _tuned_push_options(48)
    text = _lower_layer(
        v2,
        weight_format=host.WeightFormat.K32Z,
        hidden=5120,
        inter=2304,
        bits=4,
        push_options=options,
        **served,
    )
    roll, roll_from = host.k32z_roll_plan(
        host.ragged_stride_bound(8 * rows, 6, 384, 256), 8, 48
    )
    unit = host.mirror_lanes(256, host.K32Z_MIRROR_UNIT)
    assert _push_names(text) == [
        "push_fused_moe_k32zw_gu4_silu_and_mul_with_clamp_g48_c256_nb2_s2"
        "_notail_hbmst_da48_1b"
        + (f"_r{host.k32z_roll_tag(roll)}f{roll_from}" if roll else "")
        + (f"_mu{unit}" if unit < 256 else "")
    ]
    if rows == 2048:
        # 256-lane mirror sublanes take v2's combine; 128-lane units (the
        # default mirror unit) take the fused scale lookup again.
        fused = host.mirror_lanes(256, host.K32Z_MIRROR_UNIT) <= 128
        assert set(re.findall(r"moe_v2_combine\w*", text)) == {
            "moe_v2_combine_t2048_k6_b64"
            + (f"_p{v2.kernel.COMBINE_PARITIES}_fs" if fused else "")
        }
    if rows == 8:
        k512 = _lower_layer(
            v2,
            weight_format=host.WeightFormat.FP4,
            hidden=5120,
            inter=2560,
            logical=2304,
            push_options=options,
            **served,
        )
        assert _push_names(k512) == [
            "push_fused_moe_fp4w_silu_and_mul_with_clamp_g48_c128_nb2_s2"
            "_notail_hbmst_da48_1b"
        ]
