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
"""Surface CPU fallbacks in the model-op test frameworks, and fail the ones
that mask a device bug.

When a compiled op runs on Spyre but hits a CPU fallback, the op test would
otherwise pass on the CPU implementation -- not the device path it claims to
exercise -- masking real backend bugs. We capture the ``FallbackWarning`` during
the compiled run, always report it, and fail the test **only when the op that
fell back has a Spyre device kernel** (so the fallback hid a broken kernel).
Ops that have no device kernel at all (a known gap, e.g. ``sin``/``cos``,
factory ops, int<->float conversions) fall back by design, so those are
reported but not failed.

Set ``SPYRE_OPTEST_ALLOW_CPU_FALLBACK=1`` to downgrade every fallback to a
printed warning (e.g. while triaging).
"""

import contextlib
import os
import warnings
from typing import Iterator, List

try:  # torch_spyre may be absent during collection on a non-device host
    from torch_spyre.ops.fallbacks import FallbackWarning
except Exception:  # pragma: no cover - import guard
    FallbackWarning = None  # type: ignore[assignment, misc]

ALLOW_ENV = "SPYRE_OPTEST_ALLOW_CPU_FALLBACK"

# aten ops that have NO Spyre device kernel: a CPU fallback here is expected and
# is reported, not failed. Kept in sync with the CPU-fallback ops registered in
# torch_spyre/ops/fallbacks.py, MINUS ops that do have a device kernel (e.g.
# index_copy via index_put) -- for those a fallback masks a real bug and must
# fail. int<->float dtype conversions (a known device limitation) are also
# treated as a known gap.
_KNOWN_NO_KERNEL_OPS = (
    "aten.sin",
    "aten.cos",
    "aten.arange",
    "aten.cumsum",
    "aten.repeat",
    "aten.tril",
    "aten.triu",
    "aten.isin",
    "aten.argmax",
    "aten.argmin",
    "aten.where",
    "aten.bitwise_xor",
    "aten.bitwise_or",
    "aten.ne",
    "aten.any",
    "aten.normal",
    "aten.random",
)


def _is_known_gap(message: str) -> bool:
    """True if this fallback is an expected no-device-kernel gap, not a bug."""
    if message.startswith("conversion from"):  # int<->float dtype conversion
        return True
    return any(op in message for op in _KNOWN_NO_KERNEL_OPS)


@contextlib.contextmanager
def capture_cpu_fallbacks(sink: List[str]) -> Iterator[None]:
    """Record ``FallbackWarning`` messages emitted in the block into ``sink``.

    Non-fallback warnings raised in the block are re-emitted so nothing else is
    swallowed.
    """
    if FallbackWarning is None:
        yield
        return
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", FallbackWarning)
        yield
    for w in records:
        if issubclass(w.category, FallbackWarning):
            sink.append(str(w.message))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def assert_no_cpu_fallback(case_name: str, messages: List[str]) -> None:
    """Report CPU fallbacks; fail only those that mask a device-kernel bug.

    Every fallback is printed. A fallback for an op that has a Spyre device
    kernel (i.e. not a known no-kernel gap, see ``_is_known_gap``) fails the
    test, because it silently ran on CPU instead of exercising that kernel.
    Set ``SPYRE_OPTEST_ALLOW_CPU_FALLBACK=1`` to downgrade all to warnings.
    """
    if not messages:
        return
    unique = sorted(set(messages))
    print(f"\n[CPU-FALLBACK] {case_name}: {'; '.join(unique)}")
    if os.environ.get(ALLOW_ENV):
        return
    masking = [m for m in unique if not _is_known_gap(m)]
    if not masking:
        return
    raise AssertionError(
        f"{case_name}: op(s) with a Spyre device kernel silently fell back to "
        f"CPU, masking a device bug: {masking}. Fix the kernel, mark the case "
        f"xfail, or set {ALLOW_ENV}=1 to allow."
    )
