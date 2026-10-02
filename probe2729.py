# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device probe for PR #2729: floor-division sign handling.

Runs torch.div(..., rounding_mode="floor") and // through the compiled Spyre
path and compares against the CPU reference, for POSITIVE and NEGATIVE divisors.
Readback uses .cpu().float() (never .float().cpu()) to avoid the staggered-EA
readback trap.
"""

import os
import sys

import torch

import torch_spyre  # noqa: F401

DEV = "spyre:0"


def _rb(t):
    """Readback: device -> host, then widen. Order matters."""
    return t.cpu().float() if t.dtype in (torch.float16, torch.float32) else t.cpu()


def run(label, fn, x, y, dtype_note=""):
    cx = x.to(DEV) if isinstance(x, torch.Tensor) else x
    cy = y.to(DEV) if isinstance(y, torch.Tensor) else y
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()
    compiled = torch.compile(fn, backend="inductor")
    try:
        got_dev = compiled(cx, cy)
    except Exception as e:  # noqa: BLE001
        print(f"  {label:34s} RAISED {type(e).__name__}: {str(e)[:120]}")
        return None
    got = _rb(got_dev)
    exp = fn(x, y)
    exp_c = exp.float() if exp.dtype in (torch.float16, torch.float32) else exp
    bad = got != exp_c
    n = bad.sum().item() if bad.numel() else 0
    status = "OK  " if n == 0 else "FAIL"
    print(
        f"  {label:34s} {status} mismatch {n}/{got.numel():<6d}"
        f" dtype dev={got_dev.dtype} cpu={exp.dtype} {dtype_note}"
    )
    if n:
        # flatten FIRST, then take flat indices (nonzero on a 2D tensor returns
        # index *pairs*, which .flatten() would scramble into bogus positions).
        idx = bad.flatten().nonzero().flatten()[:5].tolist()
        for k in idx:
            xv = x.flatten()[k].item() if isinstance(x, torch.Tensor) else x
            yv = y.flatten()[k].item() if isinstance(y, torch.Tensor) else y
            print(
                f"      x={xv!s:>10} y={yv!s:>8} -> got {got.flatten()[k].item()!s:>8}"
                f"  exp {exp_c.flatten()[k].item()!s:>8}"
            )
    return n


def floor_div(a, b):
    return torch.div(a, b, rounding_mode="floor")


def py_floordiv(a, b):
    return a // b


def true_div(a, b):
    return torch.div(a, b)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"=== PR#2729 floor-div sign probe  [{which}] ===")

    if which in ("all", "pos"):
        print("\n-- POSITIVE divisors (the regime the PR's tests cover) --")
        x = torch.tensor([10, 20, 30, -10, 7, -7, 12], dtype=torch.int64)
        y = torch.tensor([3, 4, 5, 3, 2, 2, 3], dtype=torch.int64)
        run("int64 floor, y>0", floor_div, x, y)

    if which in ("all", "neg"):
        print("\n-- NEGATIVE divisors (uncovered by the PR's tests) --")
        x = torch.tensor([10, 20, 30, -10, 7, -7, 12], dtype=torch.int64)
        y = torch.tensor([-3, -4, -5, -3, -2, -2, -3], dtype=torch.int64)
        run("int64 floor, y<0", floor_div, x, y)

    if which in ("all", "negf32"):
        print("\n-- NEGATIVE divisors, fp32 tensor --")
        x = torch.tensor([-10.5, -20.3, 30.7, -5.2], dtype=torch.float32)
        y = torch.tensor([-2.0, -2.0, -2.0, -2.0], dtype=torch.float32)
        run("fp32 floor, y<0", floor_div, x, y)

    if which in ("all", "negscalar"):
        print("\n-- NEGATIVE scalar divisor (aten.div.Scalar_mode) --")
        x = torch.tensor([-10.5, -20.3, 30.7, -5.2], dtype=torch.float32)
        run("fp32 floor, scalar y=-2.0", floor_div, x, -2.0)

    if which in ("all", "pyop"):
        print("\n-- Python // operator (aten.floor_divide), y<0 --")
        x = torch.tensor([10, 20, 30, -10, 7, -7, 12], dtype=torch.int64)
        y = torch.tensor([-3, -4, -5, -3, -2, -2, -3], dtype=torch.int64)
        run("int64 '//', y<0", py_floordiv, x, y)

    if which in ("all", "mag"):
        print("\n-- int64 magnitude (via fp32) --")
        for shift in (20, 24, 26, 31):
            g = torch.Generator().manual_seed(7)
            x = torch.randint(0, 2**shift, (256,), generator=g, dtype=torch.int64)
            y = torch.randint(1, 1000, (256,), generator=g, dtype=torch.int64)
            run(f"int64 floor |x|<2^{shift}, y>0", floor_div, x, y)

    if which in ("all", "mixed"):
        print("\n-- randomized MIXED-SIGN sweep, 67x256 --")
        g = torch.Generator().manual_seed(0xBEEF)
        x = torch.randint(-200, 201, (67, 256), generator=g, dtype=torch.int64)
        y = torch.randint(-20, 21, (67, 256), generator=g, dtype=torch.int64)
        y[y == 0] = 1
        run("int64 floor, mixed-sign 67x256", floor_div, x, y)
        xf = (
            torch.randn((67, 256), generator=g, dtype=torch.float32) * 50.0
        )  # non-integer
        yf = torch.randn((67, 256), generator=g, dtype=torch.float32) * 10.0
        yf[yf.abs() < 0.5] = 0.5
        run("fp32 floor, mixed-sign 67x256", floor_div, xf, yf)

    if which in ("all", "truediv"):
        print("\n-- true division sanity (no rounding mode) --")
        x = torch.tensor([10, 20, 30, -10], dtype=torch.int64)
        y = torch.tensor([-3, 4, -5, 3], dtype=torch.int64)
        run("int64 true_div, mixed sign", true_div, x, y)

    print("\ndone")


if __name__ == "__main__":
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    os.environ.setdefault("SPYRE_KERNEL_CACHE", "0")
    main()
