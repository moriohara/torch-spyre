## Round-2 review — head `1c4e621b`

Thanks for the rework. The round-1 blockers are all resolved, and the design moved
to where I hoped it would go: `_promoted_dtype` / `_convert_to_dtype` are now
shared lowering-layer helpers, and `lower_where` (#4383) and `_cmp_operand_dtype`
/ `_lower_cmp_impl` (#3802) are refactored onto them. That is the convergence I
asked for — thank you for taking it on rather than adding a fourth mechanism.

Verdict stays **REQUEST CHANGES**, for one new blocker that only shows up on
device: the quotient correction is unreliable whenever the **compute dtype is
fp16**. Everything else below is minor or FYI.

### Resolved since round 1

- **Floor-div with negative divisors** — fixed; the sign-folded, division-free
  correction with `qf_minus1` derived from the *updated* `qf` is in. Device-verified
  clean: 0/17152 wrong on mixed-sign int64, 0/1024 on every fp32 and int64
  configuration I threw at it.
- **`aten.floor_divide` registered as an `OpOverloadPacket`** — gone; it is a
  lowering now, so the `.out`-overload defect class from #4029 no longer applies.
- **`trunc` returning a wrong dtype** — now raises `Unsupported`. Confirmed on
  device for fp16/fp32/int64.
- **`constants.py` collision with #4383** — gone, no `constants.py` hunk remains.
  #4383 owns that line.
- **Exact tolerances for floor-div** — `test_div_rounding_mode_cpu` now uses
  `atol=0, rtol=0`. See finding 2 for the two families that still don't.

### CI — cleared, inherited not caused

`run-tests / Inductor / Test Coarse Tile E2e` is **not** your failure. The failing
test is `test_coarse_tile_e2e__oot_wrapper.py::test_flash_v3_tile_B`, and it fails
bit-identically on `main` commits that do not contain this PR:

| run | result |
|---|---|
| this PR (`1c4e621b`) | 1 failed, 221 passed, 8 skipped — `test_flash_v3_tile_B` |
| `main` `ca14fcc5` | 1 failed, 221 passed, 8 skipped — `test_flash_v3_tile_B` |
| `main` `d22c73c5` | 1 failed, 221 passed, 8 skipped — `test_flash_v3_tile_B` |

All three report `Mismatched elements: 12506 / 262144`, greatest absolute
difference `0.68310546875` at index `(0, 1, 100, 38)`, greatest relative
difference `52928.0` at `(1, 1, 171, 63)`. Identical to the digit. Incidentally
this is also evidence that globally claiming `aten.div` does not perturb
flash-attention numerics at all.

---

### 1. [BLOCKER] The quotient correction is blind when comp_dtype is fp16

The correction detects a misestimated quotient from `rem = x - qf*y`. But `prod`
and `rem` are computed *in comp_dtype*, so when comp_dtype is fp16 the residual it
needs to see is rounded away.

Worked example, straight off the device (both operands fp16, `x = 95`,
`y = 4.3203125`; exact quotient `21.98915`, correct floor `21`):

```
device div       -> 22.0        # fp16 divider rounds across the integer boundary
device floor     -> 22.0        # already +1 too high
device qf*y      -> 95.0        # 22 * 4.3203125 = 95.046875, rounds to 95.0 in fp16
device rem       -> 0.0         # -> over_est (rem >= |y|) false, under_est (rem < 0) false
result           -> 22          # correction cannot see the error; returns 22
```

`rem == 0` is indistinguishable from "exact division, quotient already correct",
so the correction declines to fire on an off-by-one. The failures are quotient
*boundary* cases in both tails — true quotient within fp16 resolution of an
integer, either just above or just below.

Measured on device at this head (`wrong / N`, floor mode):

| operands (comp_dtype) | data | wrong | worst err |
|---|---|---|---|
| fp16 / fp16 | integer-valued `\|x\| < 101`, `y ∈ 2..10` *(the PR's own test domain)* | 0 / 1024 | — |
| fp16 / fp16 | integer-valued `\|x\| < 501`, `y ∈ 2..10` | 0 / 1024 | — |
| fp16 / fp16 | integer-valued `\|x\| < 2001`, `y ∈ 2..10` | **53 / 1024** | 1 |
| fp16 / fp16 | integer-valued `\|x\| < 8001`, `y ∈ 2..10` | **263 / 1024** | 4 |
| fp16 / fp16 | integer `\|x\| < 100`, non-integer `y` | **30 / 2048** | 1 |
| fp16 / fp16 | non-integer, `\|x\| ~ 10` | **7 / 1024** | 1 |
| fp16 / fp16 | non-integer, `\|x\| ~ 100` | **24 / 1024** | 1 |
| fp16 / fp16 | non-integer, `\|x\| ~ 1000` | **222 / 1024** | 8 |
| fp16 / fp16 | non-integer, small divisor | **97 / 1024** | 3 |
| int32 / fp16 | `test_div_mixed_dtype`'s own params | **2 / 256** | 1 |
| fp32 compute | int64/int64 `\|x\|<100`, `\|x\|<10000` | 0 / 1024 | — |
| fp32 compute | fp32 integer `x`/non-int `y`; `\|x\|~1e5`/non-int `y` | 0 / 2048 | — |

So: **fp32 compute is solid, fp16 compute is not.** Per the promotion table, fp16
comp_dtype is reached by `fp16/fp16`, `int32/fp16`, `int64/fp16` and `bool/fp16`
— i.e. "fp16 dominates" is exactly the set of broken rows.

**What makes it blocking** is the same asymmetry as round 1: on the merge-base
(`1a56c846`) these cases *raise*, so the PR converts a hard error into silent
wrong numerics. A/B on the `int32/fp16` case:

```
merge-base 1a56c846 : InductorError: KeyError: 'No FX node for buf2'
this PR    1c4e621b : runs, returns 2/256 elements wrong
```

**The obvious remedy does not work** — I tried it, so you don't have to. Forcing
`comp_dtype = fp32` for floor mode when it would otherwise be fp16 fails to
compile, because the fp16→fp32 cast is stick-reordering and the fp16 cast-back
then can't be reconciled:

```
NotImplementedError: buf15 (Pointwise): no mechanism to resolve stick incompatibility
```

So I don't think this algorithm can be made correct for fp16 operands on this
hardware. My suggestion is to **raise `Unsupported` for floor mode when comp_dtype
is fp16**, exactly as `trunc` now does. That costs nothing against `main` (these
cases already raise there), keeps the int32 / int64 / fp32 wins this PR is
actually about, and leaves a clear TODO. If you'd rather keep fp16 working for the
narrow domain where it is exact, that needs a documented precondition plus a
pinning test at the boundary — but silently-wrong-at-22% for `|x| ~ 1000` is not
something I can approve.

### 2. [SUGGESTION] Two of the three new families still use default tolerances

`test_div_rounding_mode_cpu` got `atol=0, rtol=0`, but
`test_div_mixed_dtype_cpu` (:10425) and `test_div_scalar_dtypes_cpu` (:10433) still
call `compare_with_cpu` with the defaults — and both include a `floor_div` entry
in `ops_dict`. This is not theoretical: re-running those two families at
`atol=0, rtol=0` is exactly how I found finding 1. Of the 15 cases that then fail,
14 are `true_div` (expected float rounding — exact comparison is wrong for those)
and the 1 remaining is `floor_div_int32_fp16_1d256`, the real bug.

Worth splitting the tolerance by rounding mode so the `floor_div` entries compare
exactly and `true_div` keeps the defaults.

### 3. [SUGGESTION] 17 of 22 `randint` params in the new blocks are unseeded

`floor_fp16_rand_2d` correctly pins `generator=torch.Generator().manual_seed(...)`,
but 17 of the 22 `torch.randint` calls in the three new param blocks have no
`generator=`. They are evaluated at class-definition time off the global RNG, so
the data changes run to run. That is why finding 1 is green in CI: the
`int32/fp16` floor case is only wrong for some draws. Pinning a generator on each
would make these reproducible (and would have caught finding 1).

### 4. [SUGGESTION] `_replace_near_zero` is a no-op for non-int64 integer dtypes

`tests/inductor/test_inductor_ops.py:547` picks `eps = 1` for int64, `FP32_EPS` for
fp32, and `FP16_EPS` for everything else — so an int32/int16/int8 tensor gets
`t[mask] = 0.0009765625`, which truncates back to `0`:

```
torch.int64    eps=1              -> [1, 3, -2]  zero_survives=False
torch.int32    eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
torch.int16    eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
torch.int8     eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
```

Latent today (every int32 divisor in the new params is `randint(1, 100)`), but the
helper is now called from `test_binary_op` too. One-line fix:
`if not t.dtype.is_floating_point: eps = 1`.

### 5. [QUESTION] int64-via-fp32 domain limit — still undocumented

Still unanswered from ani300's June question and from round 1. The correction is
only meaningful while fp32 spacing is below 1, i.e. `|x| < 2^24`. Measured on
device at this head, `int64 / int64` with divisors in `2..20`:

| `\|x\|` bound | wrong |
|---|---|
| `2^20` | 0 / 256 |
| `2^24` | 0 / 256 |
| `2^26` | **26 / 256** |
| `2^31` | **213 / 256** |

The cliff is exactly where predicted, and the tests only reach `|x| ≲ 200`. #3802
documented the equivalent caveat in comments and pinned it with `bigint` cases; a
sentence in the docstring plus one pinning test would close this for good.

---

### FYI — no change requested

Noting these for the record; please don't respin for them.

- **The PR body overstates `trunc`.** It still says `torch.div` with
  `rounding_mode="trunc"` "now works for `int64` tensors", but
  `trunc_int64_rand_2d` is in `expect_fail` and the lowering raises
  `Unsupported` — trunc works for no dtype at this head. Worth a line in the body
  whenever you next touch it, since `# TODO(PR#3610)` is the real status.
- **The lx mirror is more conservative than it needs to be.** The lx config lists
  only `floor_fp16_rand_2d` and the two `bool_int_scalar` cases as
  `mandatory_success`; int64 and fp32 floor-div are skipped. I expected those to
  be genuinely broken under lx (an int64→fp32 `to_dtype_cpu` FallbackKernel is
  what `test_indexing.yaml` blames for a missing `FixedTiledLayout`), but they
  are not — with `LX_PLANNING=1` both compile and run clean:
  `int64/int64 0/17152 wrong`, `fp32/fp32 0/17152 wrong`. You could promote both
  to `mandatory_success` and get real lx coverage for free.
- **`bool/bool` in the promotion table is unreachable** — aten raises
  `NotImplementedError` for bool `floor_divide`, so that row can never be hit.
- **`realize()` count.** The floor path takes ~10–14 fusion barriers per division,
  on an `aten.div` that is now claimed globally. Not a correctness issue and I
  have no measurement saying it matters; flagging it only because the blast radius
  is every division in every model.
- **Deleting `probe2729.py`** was the right call — thank you.

### Reproduction

All numbers above are from head `1c4e621b` on device, with
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SPYRE_KERNEL_CACHE=0`. The full new test
selection passes as shipped: `49 passed, 11 xfailed` for
`-k "div_rounding_mode or div_mixed_dtype or div_scalar_dtypes"`.

