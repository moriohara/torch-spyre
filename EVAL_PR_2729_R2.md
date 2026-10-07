## Round-2 review — head `1c4e621b`

Thanks for the rework. #3802 and #4383 are **both merged** — they are ancestors
of this PR's merge-base `1a56c846` — so the dtype-promotion code this PR sits
next to is settled, not a moving target. And this head converges on it rather
than adding to it: the `spyre::adapt_dtype` / `adapt_dtype_scalar` custom ops are
**gone** (they would have been a fourth mechanism, at the *decomposition* layer),
replaced by two *lowering*-layer helpers, `_promoted_dtype` and
`_convert_to_dtype` — and the merged `lower_where` (#4383) and
`_cmp_operand_dtype` / `_lower_cmp_impl` (#3802) are **refactored onto those
helpers** rather than left standing beside them. So this is a retrofit of
already-merged code onto a shared mechanism. That is the convergence I asked for
in round 1, and it leaves the tree better than either merged PR did on its own —
thank you for taking it on. (`with_int64_fallback`, also on main, is still
separate with its own six call sites; not this PR's job, and I'll follow up on
it.)

Verdict stays **REQUEST CHANGES**, for one blocker that only shows up on device:
the quotient correction is unreliable whenever the **compute dtype is fp16**.
Everything else below is a suggestion or FYI.

### Resolved since round 1

The `B`/`Q`/`S` labels below are my own shorthand for the items in the
[round-1 review](https://github.com/torch-spyre/torch-spyre/pull/2729#pullrequestreview-5390400153)
— **B1**, **B2** its two BLOCKERs, **Q1**–**Q3** its QUESTIONs, **S1**, **S2**
its SUGGESTIONs, numbered in the order they appear there. Round 1 did not label
them, and its bullets have no anchors of their own, so each is quoted by title
here. Plain **finding 1**–**4** always means this review's numbered findings
below.

- **B1 — "Floor division returns the wrong value for every negative divisor."**
  **Resolved for `floor`.** The sign-folded, division-free correction with
  `qf_minus1` derived from the *updated* `qf` is in, and I confirmed it is not
  merely untested: reinstating the round-1 defect (compare `rem` against `y`
  instead of `rem_s` against `abs_y`) makes **`floor_int64_negdiv_2d`** and
  **`floor_int64_mixedsign_2d`** fail, so the fix is real and pinned by the new
  tests. Those two are the only ones of the five negative-divisor sets that catch
  it — finding 2 is about why the other three do not. All 11
  `test_div_rounding_mode` cases pass on device. Two things to keep in
  mind, neither of them a re-open:
  - For **`trunc`** the round-1 defect is *gone* rather than fixed: there is no
    trunc branch any more. `lowering.py:2143-2145` raises
    `Unsupported("div with rounding_mode='trunc' is not yet implemented")` with a
    `TODO(PR#3610)`, and all 3 trunc cases xfail on that rejection. This is the
    right outcome — round 1's objection was that a *wrong answer* was being
    xfailed — so no ask here. Just note that trunc inherits two obligations from
    this review whenever `PR#3610` lands it: a negative-divisor
    exact-division case (the B1 defect class was never exercisable in trunc), and
    the quotient-range problem in finding 3, since `trunc_fp16_rand_2d` reaches
    `|q| = 40608` against an fp16 ceiling of 1024.
  - B1 was one of **two independent defects in the same block of code**, and only
    one of them is fixed. The block is the quotient correction at
    `lowering.py:2103-2137`: `rem = x - qf*y`, compare `rem_s` against `abs_y`
    and against `0`, then add or subtract 1. B1 was about how that block handles
    **signs**, and the sign handling is now right. Finding 1 of this review (the
    blocker, below) is about the **dtype the block is evaluated in**: `prod` and
    `rem` are computed at `comp_dtype`, so in fp16 the residual the block needs to
    see can round away. In the worked example it rounds to exactly `0.0`, which is
    indistinguishable from "exact division, quotient already correct" — neither
    comparison fires and the off-by-one stands. Despite the correct sign handling,
    rounding error in the residual still leads to a wrong output.
- **B2 — "No new test uses a negative divisor, and the default tolerances would
  hide it even if one did."** Largely resolved. Round 1 asked for two things and
  each got a partial answer:
  - *Add negative-divisor coverage* → five param sets added, and all five are
    useful, but **only two of them pin the defect**: with the round-1 bug
    reinstated the other three still pass, because the defect needs exact
    division with a negative divisor and those three contain none. Two
    one-element edits would close that (finding 2).
  - *Tighten the tolerances* → done in `test_div_rounding_mode_cpu` (`atol=0,
    rtol=0`), but **the other two new families were not touched**:
    `test_div_mixed_dtype_cpu` and `test_div_scalar_dtypes_cpu` still compare at
    the default `atol=rtol=0.1`, and both include `floor_div` (finding 3).
- **Q2 — "Is the mixed-registry access in the floor path deliberate?"**
  Answered by the new comment at `lowering.py:2098-2100`, and the answer checks
  out: `register_spyre_lowering` writes into a separate `spyre_lowerings` dict
  (`:84`), so `lowering.lowerings[aten.ge.Tensor]` really is the upstream
  built-in. I substituted Spyre's `ge`/`lt` wrappers and got bit-identical
  results across six dtype domains, so the "both would be no-ops" claim holds for
  the comparisons. One nuance in the FYI list.
- **S1 — "Mirror the three new groups into the lx-planning config."** Complete
  and live. All five name patterns collect; the 6 `mandatory_success` lx cases
  pass and the 4 `fp16_fp32_1d256` cases xfail,
  exactly as the yaml claims. I also ran the whole div family under
  `LX_PLANNING=1` (147 tests): the only failure is
  `test_pointwise_binary_op_div_67x71x256_..._reduction` at 1/18176 elements,
  which is already `mode: xfail` in a config file this PR does not touch. Not a
  mirror gap.

Still open from round 1: **Q1** ("What is the intended domain for the
int64-via-fp32 path?" — carried forward as finding 4), **S2** ("Consider a trunk
perf run before merge."), the four round-1 FYIs, and **Q3**, which is worth
restating since it asks about what the PR *claims* rather than what it does:

- **Q3 — "Which of the 14 `test_div_mixed_dtype` cases actually exercise the
  device div?"** The point of the question was to pin down which half of
  "converting them into fp32 tensors" — this PR's own title — actually happens on
  device. The measured answer is that the **division** runs on device but **both
  conversions are host-side**: `int64 → float32` and `float32 → int64` each report
  `falling back to cpu`, because `DtypeOpTable` has no int64 entries at all (still
  true at this head — `dtype_ops.py` does not mention int64 anywhere). I am not
  asking for a code change; that is the only route available today, and #3802 took
  the same one. Two things follow from it, though:
  - The description still reads as device int64 support. #3802 documented the
    identical mechanism explicitly as a host round-trip, and it would be good for
    the two to say the same thing.
  - The `filterwarnings` decorators on all three new tests
    (`:10406-10407` and the two others) suppress `FallbackWarning` and
    `"Backend Spyre does not support int64"` — exactly the signals that say work
    moved to the host. They are legitimate noise suppression, but they do mean a
    green run tells a reader nothing about where the work ran, which is why the
    description carrying it matters.

  A sentence in the body whenever you next touch it closes this; no respin on its
  own account.

### Withdrawn — claims of mine that were wrong

Both of these are from the round-2 review I took back on 10-06, not from round 1.
If you read that version in a notification, please disregard these two parts of
it; everything else in this review supersedes it.

- **The unseeded-`randint` suggestion ("17 of 22 new `randint` params are
  unseeded", item 3 there).** Retracted. The count is right (22 added calls, 5
  with an explicit `generator=`), but the conclusion was not: `TestOps` runs
  `torch.manual_seed(0xAFFE)` in its class body (`:801`), which executes
  *before* the `PARAMS` dict literal is evaluated, so every one of those calls
  is in fact seeded. I verified it — two fresh processes give byte-identical
  param data (`x[:8] = [35, 94, 14, 90, 97, 9, 73, 87]`, `initial_seed 45054`),
  and torch's default generator is otherwise per-process random, so the seed is
  doing the work. Nothing to fix. The one residual property, and it is a note
  not an ask: this determinism is *positional*, so inserting a param set above
  an existing one shifts the data of everything after it.
- **Its corollary, "that is why finding 1 is green in CI — the `int32/fp16`
  floor case is only wrong for some draws".** Wrong twice over. The draws do not
  vary at all (same seed), and the case is wrong on **every** run — 4 of its 256
  elements mismatch. What keeps CI green is that this family compares at
  `atol=rtol=0.1`, which absorbs those mismatches, plus `expect_fail` on the
  trunc family. I also reasoned from "the data stays at `|x| ≲ 200`, so it is
  below the problem range": the *numerators* do, but the *quotients* reach
  2×10⁴, well past what device fp16 can represent. The real explanation is
  finding 3.

---

### 1. [BLOCKER] The quotient correction is blind when comp_dtype is fp16

The correction detects a misestimated quotient from `rem = x - qf*y`. But `prod`
and `rem` are computed *in comp_dtype*, so when comp_dtype is fp16 the residual
it needs to see can round away entirely.

Worked example, straight off the device — `x = 95`, `y = 4.3203125`, both fp16.
I picked these two because both are **exactly representable** on device, so
nothing here is transfer rounding. Exact quotient `21.98915`, correct floor `21`:

```
device div       -> 22.0        # the divider rounds across the integer boundary
device floor     -> 22.0        # already +1 too high
device qf*y      -> 95.0        # 22 * 4.3203125 = 95.046875, rounds to exactly 95.0
device rem       -> 0.0         # over_est (rem >= |y|) false, under_est (rem < 0) false
result           -> 22          # correction cannot see the error; returns 22
```

`rem == 0` is indistinguishable from "exact division, quotient already correct",
so the correction declines to fire on an off-by-one. The failures are quotient
*boundary* cases in both tails — a true quotient within fp16 resolution of an
integer, from either side.

**This reproduces at every shape your tests use.** Same two values, floor mode:

| shape (your param sets) | as shipped | computed in fp32 |
|---|---|---|
| `(67, 256)` — every rand 2-D case | **17152 / 17152 wrong** | 0 wrong |
| `(3, 5, 256)` — `fp16_3d` | **3840 / 3840 wrong** | 0 wrong |
| `(256,)` — the `1d256` cases | **256 / 256 wrong** | 0 wrong |

Rates over random domains (fp16/fp16, floor). The middle column is how many of
the correct quotients are not representable in the device's fp16 at all — those
are not the kernel's fault, and I have excluded them from the judgement:

| data | `q` unrepresentable | as shipped | in fp32 |
|---|---|---|---|
| integer `\|x\| < 101` / `< 501` / `< 1024`, `y ∈ 2..10` *(your test domain)* | 0 | 0 | 0 |
| integer `\|x\| < 2001` | 0 | **10 / 1024** | 0 |
| integer `\|x\| < 8001` | 64 | **169 / 1024** | 0 |
| non-integer, `\|x\| ~ 10` / `~ 100` / `~ 1000` | 0 / 0 / 46 | **1 / 27 / 168** per 1024 | 0 / 0 / 0 |
| `x = 95`, `y = 4.3203125` | 0 | **1024 / 1024** | 0 |

Two measurement notes, because both bit me:

- Device `torch.float16` is DLFLOAT16 (`SEN169_FP16`), not IEEE fp16. A bare
  `.to("spyre:0").cpu()` already changes roughly half of all fp16 values —
  `1853.0` comes back as `1854.0`. So a host fp16 expectation is not a valid
  reference; every number above is measured against the operands the device
  actually holds, with the expected value rounded through the device format too.
- `_dlfloat16_saturating_ref` is **not** the tool for that. It emulates overflow
  *saturation* only and is the identity for every value below the fp16 max — I
  checked, it leaves `2251.0`, `7948.0` and `21183.0` untouched, while a real
  device round-trip maps them to `2252.0`, `7952.0` and `21184.0`. The only
  reference I found that is actually faithful is a device round-trip of the
  expected value. Worth knowing before writing any new DLFLOAT16 assertion.

The relevant resolution limit, measured: **device fp16 represents integers
exactly only up to 1024.** Spacing is 2 at 1024, 4 at 2048, 8 at 4096, 32 at
16384. Floor division is therefore only well-posed in fp16 comp_dtype while
`|quotient| <= 1024` — which matters for the tests, see finding 3.

Per the promotion table, fp16 comp_dtype is reached by `fp16/fp16`, `int32/fp16`,
`int64/fp16` and `bool/fp16` — "fp16 dominates" is exactly the set of affected
rows. **fp32 compute is clean in every domain I measured.**

**What makes it blocking** is the same asymmetry as round 1: on the merge-base
these cases *raise*, so the PR converts a hard error into silent wrong numerics.

```
merge-base 1a56c846 : InductorError: KeyError: 'No FX node for buf2'
this PR    1c4e621b : runs, returns a wrong quotient
```

**Remedy — do the compute in fp32.** This is a verified positive
recommendation, not a guess: in user code the same algorithm expressed as
`.float()` → floor-div → `.half()` is **0 wrong at `(67,256)`, `(3,5,256)` and
`(256,)`** where the shipped path is 100% wrong, and 0 wrong across all the random
domains in the table above. It needs no `Unsupported` carve-out and it keeps the
fp16 cases working rather than rejecting them.

One practical note from trying it: when I forced `comp_dtype = fp32` inside
`_lower_div_impl` as a quick experiment, the result was shape-dependent — several
shapes were exact, but `(67,256)` and `(256,)` hit
`NotImplementedError: no mechanism to resolve stick incompatibility` in
`optimize_restickify`. I did not chase that down and it may well be something
simple in how I forced it. Flagging it only so the upcast gets validated across
the existing op-test shapes rather than a single one.

### 2. [SUGGESTION] Only two of the five negative-divisor sets pin the round-1 bug — two one-element edits would fix that

To check whether the new tests actually pin B1 I reinstated the round-1 defect
(compare `rem` against `y` instead of `rem_s` against `abs_y`) and re-ran. Only
**two** of the five sets fail: `floor_int64_negdiv_2d` and
`floor_int64_mixedsign_2d`. `floor_fp32_negdiv_2d`, `floor_fp32_negscalar` and
`floor_int64_negscalar` still pass with the bug reinstated.

The reason is structural: for `y < 0` a correct remainder lies in `(y, 0]`, so the
spurious `rem >= y` fires on every element and is then cancelled by `rem < 0` —
*unless* `rem == 0`. So the defect is only observable on **exact division with a
negative divisor**, and those three sets contain **zero** exact divisions (the two
that do catch it have 2556 and 1603): the two scalar sets use only odd numerators
(`[-11, -21, 31, -7] // -2`, `[-10.5, -20.3, 30.7, -5.2] // -2.0`), and the fp32
2-D set draws from `randn`, where exact division has measure zero.

**To be clear, these three are not redundant** — they are simply not *regression
tests for B1*, which is a different thing:

- `floor_fp32_negscalar` and `floor_int64_negscalar` are the only negative-divisor
  cases with a **Python scalar** divisor, and that takes the other branch of the
  sign folding you added — the compile-time fold at `lowering.py:2126-2129`
  (`rem_s = neg(rem) if y < 0 else rem`), not the runtime
  `lt`/`where` pair. Both catching sets have tensor divisors, so **that branch is
  exercised only by these two**. Dropping them would lose real coverage.
- `floor_fp32_negdiv_2d` is the only one where `result_dtype` is float, so it is
  the negative-divisor case that does not exit through the int cast-back.

So the gap is narrow and the fix is correspondingly small: the scalar branch is
covered but **not pinned**, and making it pinned costs two elements. Give each
scalar set one exactly-divisible numerator — e.g. `[-12, -21, 31, -7] // -2` and
`[-10.0, -20.3, 30.7, -5.2] // -2.0`. For `-12 // -2`: `qf = 6`, `rem = 0`, so
the defect's `rem >= y` (`0 >= -2`) fires, `rem < 0` does not, and the result is
**7** instead of 6. That is cheaper than a new param set, keeps the dtype matrix
as it is, and leaves the scalar fold with a real regression test. (If you would
rather add a case, `x = [12, -12, 20, -20, 7, -7, 0, 13]`, `y = -4` gives 5 exact
divisions: correct `[-3, 3, -5, 5, -2, 1, 0, -4]`, defect
`[-2, 4, -4, 6, -2, 1, 1, -4]`.)

`floor_fp32_negdiv_2d` I would leave alone — random floats will not produce exact
divisions, and the tensor branch is already pinned by the two int64 sets.

Related, so you can weigh it: the `atol=0, rtol=0` tightening is **not** what
catches the round-1 defect — I ran the 2×2 over {defect, default tolerances} and
the same two tests fail either way. The deciding factor is the *size* of the
quotient, not the tolerance. `assert_close` passes when
`|got - exp| <= atol + rtol*|exp|`, so at `atol=rtol=0.1` an off-by-one is
tolerated for every `|q| >= 9` and caught for every `|q| < 9`. B1's off-by-ones
land on small quotients (`-3` becomes `-2`), so the defaults already caught them.
Finding 1's land on large ones (`22` for `95 // 4.3203125`, `229` for the
`int32/fp16` case), which is why only the strict tolerance catches those. Both
changes are worth keeping, just for different reasons than the round-1 comment
implied.

### 3. [SUGGESTION] Two families compare at default tolerances, and one of them tests quotients fp16 cannot represent

Round 1 made this and the `_replace_near_zero` point as two separate asks. They
turn out to be coupled — the tolerance is what hides the problem and the
near-zero guard is what fails to prevent it — so I have merged them.

`test_div_rounding_mode_cpu` got `atol=0, rtol=0`, but `test_div_mixed_dtype_cpu`
(:10425) and `test_div_scalar_dtypes_cpu` (:10433) still use the defaults — and
both include a `floor_div` entry. Re-running those two at `atol=0, rtol=0`:
**15 fail, 14 of them `true_div`** (exact comparison is simply wrong for true
division) and exactly one `floor_div` — `int32_fp16_1d256`.

I then looked at *why* that one fails, and only one of its four mismatching
elements is the kernel's fault:

| `x` | `y` | exact quotient | correct | device | quotient representable in fp16? |
|---|---|---|---|---|---|
| 72 | 0.314453125 | 228.968944 | 228 | **229** | yes — **genuine, finding 1** |
| 38 | 0.016876220703125 | 2251.688969 | 2251 | 2252 | no (nearest is 2252) |
| 50 | 0.006290435791015625 | 7948.574894 | 7948 | 7936 | no (nearest is 7952) |
| 22 | 0.0010385513305664062 | 21183.353535 | 21183 | 21152 | no (nearest is 21184) |

Three of the four have a *correct answer the comp dtype cannot hold* — integers
are exact in device fp16 only to 1024, and these quotients are 2×10³ to 2×10⁴.
Those three are not the kernel's fault; the param set is simply asking fp16 floor
division a question it cannot answer.

The cause is the divisor *distribution*, not the near-zero guard — I checked, and
on this param set `_replace_near_zero` does not fire on a single element. `y` is
`cached_randn((256,), abs=True, scale=10.0, dtype=fp16)`, and fp16 has fine
resolution near zero, so it naturally produces divisors down to ~1e-3; against
`|x| <= 99` that is `|q| ~ 2×10⁴` on its own. What the guard *could* have done is
prevent it, and it cannot: `eps = FP16_EPS = 0.0009765625` is sized to avoid
**zero** (which is what its comment says — "Division by 0 or near-zero differs on
Spyre from CPU") and it admits `|q|` up to **102,400**, a hundred times past the
ceiling. The default `rtol=0.1` then hides the whole thing — slack is ±2118 at
`|q| = 21183`.

Scoped across both tensor-divisor families, exactly **two** param sets exceed
their comp dtype's exact-integer ceiling, and both are fp16-comp:

| param set | comp | max \|q\| | ceiling | currently masked by |
|---|---|---|---|---|
| `int32_fp16_1d256` | fp16 | 21183 | 1024 | the default tolerances — this finding |
| `trunc_fp16_rand_2d` | fp16 | 40608 | 1024 | `expect_fail` (trunc raises `Unsupported`) |

No fp32-comp set is at risk: the worst is `floor_fp32_rand_2d` at `|q| ~ 3.4×10⁵`
against a 2²⁴ ceiling, four orders of magnitude of headroom. (Which is another
point in favour of finding 1's fp32 remedy — it fixes the representability
problem as a side effect.) `trunc_fp16_rand_2d` is worth noting now because it is
only green by virtue of the xfail: whenever `PR#3610` lands trunc, that case goes
red for a reason that is not the kernel's.

Raising the divisor floor so the quotient stays inside the exact-integer range
fixes it. Verified on device, same param set:

| divisor floor | max \|q\| | \|q\| > 1024 | wrong at `atol=0, rtol=0` |
|---|---|---|---|
| `FP16_EPS` = 0.0009765625 *(as shipped)* | 21183 | 3 / 256 | 4 / 256 |
| `\|x\|.max() / 1024` ≈ 0.0967 | 650 | 0 / 256 | **1 / 256** |

and the single remaining failure is exactly the `x = 72` row above — so with the
floor raised, this family becomes a clean exact test that catches finding 1 and
nothing spurious. Suggested change, both halves:

```python
def _replace_near_zero(t, numerator=None):
    ...
    if numerator is not None and t.dtype.is_floating_point:
        # Floor the divisor so |numerator / t| stays inside the compute dtype's
        # exact-integer range (1024 for device fp16, 2**24 for fp32).  Merely
        # avoiding zero is not enough: floor division of an unrepresentable
        # quotient has no right answer.
        exact_int_max = 1024 if t.dtype == torch.float16 else 2**24
        eps = max(eps, float(numerator.abs().max()) / exact_int_max)
    t[torch.abs(t) < eps] = eps
```

and, since the harness passes the op *function* and not its name, a module-level
`_floor_div` makes the tolerance split a one-liner:

```python
def _floor_div(a, b):
    return torch.div(a, b, rounding_mode="floor")

# ops_dict: {"true_div": lambda a, b: torch.div(a, b), "floor_div": _floor_div}

def test_div_mixed_dtype_cpu(self, op, x, y):
    if isinstance(y, torch.Tensor):
        _replace_near_zero(y, numerator=x)
    # floor division is exact integer arithmetic; true division is not.
    tol = {"atol": 0, "rtol": 0} if op is _floor_div else {}
    self.compare_with_cpu(op, x, y, **tol)
```

**Impact if left as-is:** these two families cannot detect an off-by-one quotient
for any `|q| >= 9` (that is the `atol + rtol·|e|` crossover), so they are blind to
finding 1's entire class today *and* would stay blind after finding 1 is fixed — a
regression would not be caught. Tightening them is how finding 1's fix gets
pinned, which is why I am still asking.

While you are in that helper: it also picks `FP16_EPS` for **int32/int16/int8**,
where `t[mask] = 0.0009765625` truncates straight back to `0`:

```
torch.int64    eps=1              -> [1, 3, -2]  zero_survives=False
torch.int32    eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
torch.int16    eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
torch.int8     eps=0.0009765625   -> [0, 3, -2]  zero_survives=True
```

I checked whether this is live and it is **not** — every tensor divisor in all
three new families is a float tensor, an int64 tensor (`eps = 1`, correct), or an
int32 `randint(1, 100)` that contains no zeros by construction. The
zero-inclusive `randint(0, 100, ...)` tensors are all **numerators**. So this half
is latent; `if not t.dtype.is_floating_point: eps = 1` closes it for free while
the file is open, but it is not why I am requesting a change.

### 4. [QUESTION] int64-via-fp32 domain limit — still undocumented

Still unanswered from ani300's June question and from round 1. The correction is
only meaningful while fp32 spacing is below 1, i.e. `|x| < 2^24`. Measured on
device at this head, `int64 / int64` with divisors in `2..20`:

| `\|x\|` bound | wrong |
|---|---|
| `2^20` | 0 / 256 |
| `2^24` | 0 / 256 |
| `2^26` | **26 / 256** |
| `2^31` | **213 / 256** |

The cliff is exactly where predicted, and the tests only reach `|x| ≲ 200`, so
nothing in the suite would notice.

**Impact if left as-is:** silent wrong quotients for int64 operands above 2^24,
with no comment telling the next reader the limit exists. It is a narrow band in
practice — `aten.div` is now claimed globally, but int64 values that large are
mostly token/position arithmetic, and the device downcasts int64 to int32 anyway
— so I am not calling it blocking. What makes it worth closing is that #3802 was
asked the same question and answered it with a comment plus `bigint` pinning
cases; leaving it open here means the two converged code paths document their
shared limitation inconsistently.

Suggested change, roughly the #3802 shape:

```python
# The ±1 quotient correction is only meaningful while the compute dtype's
# spacing is below 1: |x| < 2^24 for fp32, |x| <= 1024 for fp16.  Above that
# the correction cannot distinguish a misestimated quotient from an exact one
# and the result is silently wrong.  Not rejected, because the practical int64
# domain (indices, positions) is far below the bound.
```

plus one `bigint` param set at, say, `|x| ~ 2^26` marked `expect_fail` — that
documents the bound *and* turns it red the day someone fixes it, which a comment
alone does not. If you would rather answer the question than close it ("we accept
the limit, here is why"), that works for me too; what I do not want is a third
round where it is still open.

---

### FYI — no change requested

Noting these for the record; please don't respin for them.

- **`lower_where` is not substitutable here, and it is worth a comment.** The
  `:2098-2100` rationale is right about the comparisons but the "no-op" wording
  does not extend to `where`: substituting Spyre's `lower_where` for
  `lowering.where` does not compile (`KeyError: 'No FX node for buf4'`), because
  computed bools are deliberately *not* cast-skipped — per `lower_where`'s own
  comment at `:1919-1920` they must pass through `to_dtype` so
  `propagate_layouts` can resolve their device dtype. So the wrapper emits an
  extra op inside a step that has to stay single-op. Using the built-in is
  correct; the reason is "the wrapper is non-composable inside a `_realized()`
  chain", not "it would be a no-op".
- **The PR body overstates `trunc`.** It still says `torch.div` with
  `rounding_mode="trunc"` "now works for `int64` tensors", but
  `trunc_int64_rand_2d` is in `expect_fail` and the lowering raises `Unsupported`
  — trunc works for no dtype at this head. Worth a line whenever you next touch
  the body, since `# TODO(PR#3610)` is the real status.
- **The lx mirror is more conservative than it needs to be.** The lx config lists
  only `floor_fp16_rand_2d` and the two `bool_int_scalar` cases as
  `mandatory_success`; int64 and fp32 floor-div are skipped. I expected those to
  be genuinely broken under lx (an int64→fp32 `to_dtype_cpu` FallbackKernel is
  what `test_indexing.yaml` blames for a missing `FixedTiledLayout`), but they are
  not — with `LX_PLANNING=1` both compile and run clean: `int64/int64 0/17152
  wrong`, `fp32/fp32 0/17152 wrong`. You could promote both and get real lx
  coverage for free.
- **`_replace_near_zero` mutates a cached tensor in place.** `cached_randn` is
  `@functools.lru_cache`, so the rewrite lands on the shared entry. Demonstrated,
  not theoretical: `trunc_fp16_rand_2d`'s `y` *is* the live cache entry for
  `((67,256), fp16, abs=True, scale=10.0)` and 2 elements are bumped in place. Any
  later caller asking for those same args gets the mutated tensor. Pre-existing
  pattern, low impact at these params, but it is now applied across more families.
- **`bool/bool` in the promotion table is unreachable** — aten raises
  `NotImplementedError` for bool `floor_divide`, so that row can never be hit.
- **`realize()` count.** The floor path takes ~10–14 fusion barriers per division,
  on an `aten.div` that is now claimed globally. Not a correctness issue and I
  have no measurement saying it matters; flagging it only because the blast radius
  is every division in every model.
- **Deleting `probe2729.py`** was the right call — thank you.

### Reproduction

All numbers above are from head `1c4e621b` on device, with
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SPYRE_KERNEL_CACHE=0`, one fresh process
per shape. The full new test selection passes as shipped: `49 passed, 11 xfailed`
for `-k "div_rounding_mode or div_mixed_dtype or div_scalar_dtypes"`.
