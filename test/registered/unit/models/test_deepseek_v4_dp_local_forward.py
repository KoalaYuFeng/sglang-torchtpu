"""DeepSeek V4's ``mlp.dp_local_forward`` seam in ``_run_moe_ffn_dp_sync``."""

import __future__

import ast
import contextlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import reduce_moe_output
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PATH = Path(sglang.__file__).resolve().parent / "srt" / "models" / "deepseek_v4.py"
METHOD = "_run_moe_ffn_dp_sync"
ORDINARY = ["gather", "AR", "slice"]  # DP gather, post-experts all-reduce, scatter


def _a2a_none():
    names = ("flashinfer", "pplx", "flashinfer_megamoe")
    return types.SimpleNamespace(
        **{f"is_{n}": (lambda: False) for n in names}, is_none=lambda: True
    )


def run_dp_sync(dp_local_forward=None, input_ids=None):
    """Execute the real method (from source, without weights) on a two-rank
    attention-DP + TP-MoE fake. Returns (output, trace, batch)."""
    trace = []

    def gather(output, value, batch):
        trace.append("gather")
        output.fill_(1)

    def scatter(output, value, batch):
        trace.append("slice")
        output.copy_(value[:2])

    def reduce_scatterv(value, *, output, sizes):
        trace.append("RSv")
        output.copy_(value[:2] * 2)

    parallel = types.SimpleNamespace(
        attn_dp_size=2,
        attn_tp_size=1,
        tp_size=2,
        tp_rank=0,
        dwdp_size=1,
        moe_dp_size=1,
        tp_group=types.SimpleNamespace(reduce_scatterv=reduce_scatterv),
    )
    node = next(
        n
        for n in ast.walk(ast.parse(PATH.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == METHOD
    )
    namespace = dict(
        torch=torch,
        get_forward=get_forward,
        get_parallel=lambda: parallel,
        get_moe_a2a_backend=_a2a_none,
        should_use_dp_reduce_scatterv=lambda: False,
        is_dp_gatherv_active=lambda: False,
        is_cp_active=lambda batch: False,
        envs=types.SimpleNamespace(
            SGLANG_DP_USE_REDUCE_SCATTER=types.SimpleNamespace(get=lambda: False)
        ),
        _SHARED_EXPERT_LOCAL=False,
        nullcontext=contextlib.nullcontext,
        get_global_dp_buffer=lambda group: torch.empty(4, 3),
        get_local_dp_buffer=lambda group: torch.empty(2, 3),
        get_dp_global_num_tokens=lambda: [2, 2],
        dp_gather_replicate=gather,
        dp_scatter=scatter,
    )
    code = compile(
        ast.Module(body=[node], type_ignores=[]),
        str(PATH),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)

    def mlp(value, batch, **kwargs):
        return reduce_moe_output(value)

    # DeepseekV2MoE.__init__ sets it to None; a backend may replace it.
    mlp.dp_local_forward = dp_local_forward
    layer = types.SimpleNamespace(mlp=mlp)
    batch = types.SimpleNamespace(
        dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True)
    )
    with (
        patch.object(moe_utils, "get_parallel", return_value=parallel),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=_a2a_none()),
        patch.object(
            moe_utils,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ),
        patch.object(moe_utils, "should_use_dp_reduce_scatterv", return_value=False),
        patch(
            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
            side_effect=lambda value: trace.append("AR") or value * 2,
        ),
        get_forward().scoped(mlp_reduce_scatter=False),
    ):
        output = namespace[METHOD](
            layer,
            torch.ones(2, 3),
            batch,
            input_ids=input_ids,
            input_ids_global=None,
        )
    return output, trace, batch


class TestDpLocalForward(CustomTestCase):
    def test_no_backend_keeps_the_ordinary_path(self):
        output, trace, _ = run_dp_sync()
        self.assertEqual(trace, ORDINARY)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_backend_output_is_returned_as_is(self):
        calls = []
        local = torch.full((2, 3), 7.0)
        input_ids = torch.tensor([3, 4])

        def dp_local_forward(hidden_states, forward_batch, *, input_ids):
            calls.append((hidden_states, forward_batch, input_ids))
            return local

        output, trace, batch = run_dp_sync(dp_local_forward, input_ids)
        self.assertIs(output, local)
        self.assertEqual(trace, [], "no gather, all-reduce, RSv or scatter")
        ((hidden_states, forward_batch, ids),) = calls
        torch.testing.assert_close(hidden_states, torch.ones(2, 3))
        self.assertIs(forward_batch, batch)
        self.assertIs(ids, input_ids)

    def test_none_selects_the_ordinary_path(self):
        calls = []

        def decline(hidden_states, forward_batch, *, input_ids):
            calls.append(hidden_states)
            return None

        output, trace, _ = run_dp_sync(decline)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace, ORDINARY)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))


if __name__ == "__main__":
    unittest.main()
