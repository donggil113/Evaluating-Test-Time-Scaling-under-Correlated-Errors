# STATUS — N5 (The Marginal Value of Verification)

**Decision: NO_GO** (2026-09-26), under the rule frozen in `docs/protocol.md` §8.

Pre-existing state: the repository was empty (no CLAUDE.md, AGENTS.md,
README, STATUS, protocol or logs; no remote branches). Nothing was
overwritten or re-run from earlier work.

## Why NO_GO

1. **No cost–error improvement (E1–E3, test split, 32 problems, paired
   problem-level bootstrap).** At every budget from 2× to 256× the cost of
   one generation, the best dev-selected verifier-using policy is not
   significantly better than the best generation-only policy (differences
   −0.02 to −0.04 error, all 95% CIs include 0, Holm p = 1.00). Calibrated
   stopping does not beat fixed allocation (one budget significantly *worse*
   before adjustment: +0.030 [+0.006, +0.056], Holm p = 0.09). The
   correlation-aware (Beta-binomial) posterior does not beat the independence
   posterior (−0.03 to −0.055, CIs include 0), and both are badly calibrated on
   test states (ECE 0.15 / 0.19; log-loss 1.32 vs 1.386 for a uniform guess).
2. **The measurements confirm existing conclusions instead of adding a new
   one.** Repeated verdicts of the same verifier on the same candidate are
   strongly correlated (ρ ≈ 0.72 wrong / 0.78 correct); a third of wrong
   candidates are accepted by all 32 calls; a Beta fit on 8 repeats predicts
   the held-out all-accept rate of 24 further gates (0.363 observed vs 0.375
   predicted) while independence predicts 1.4e-5; posterior log-odds saturate
   (0.49 after 24 accepts vs 6.2 under independence). This is exactly what
   PCVC (arXiv 2607.13918) predicts, the false-positive ceiling of Stroebl et
   al. (2411.17501), and the "SC ≥ GenRM at matched compute" result of Singhi
   et al. (2504.01005), which produced these very traces.
3. **No oracle leak** was found (mutation test + negative control), so the
   NO_GO is not due to a leak.

The only observation not directly stated in the works read is that the
verifier's discrimination pooled over problems (AUROC 0.66) collapses within
a problem (0.545 [0.470, 0.612], the quantity that matters for selection).
It is one generator–verifier pair on 46 problems with both classes, the CI
includes chance, and pooled-vs-within confounding by difficulty is a known
statistical pattern. It does not meet the GO bar and is not counted as a
result.

### Power caveat

32 test problems give wide CIs: at budgets ≥ 32× the E1 intervals do not
exclude a verification benefit of up to ~11–15 points. Dev differences were
at most 2 points in favour of verification (and negative at low budgets), and within-problem discrimination is near
chance, so a large hidden benefit is unlikely here, but the negative result is
"no detectable gain", not "proven zero gain".

## What ran (see `results/manifest.json`)

| id | what | status |
|---|---|---|
| R0 | build 64 × 256 × 1 × 32 response table from released traces | DONE |
| R1 | dev fitting + per-budget selection | DONE (twice, see D1) |
| R2 | single test evaluation of the frozen selection | DONE (once) |
| R3 | M1–M3 measurements with problem-cluster bootstrap | DONE |
| R4 | descriptive (N, V) grids and cost accounting | DONE (after freeze) |
| T | leakage / replay-equivalence tests | PASS (10) |

## NOT_RUN

* **Model-diversified verifier on the same candidate pool** — the released
  GPQA pool has one verifier. Candidate data: MATH-128 Qwen-2.5-7B solutions
  verified with repeats by two fine-tuned verifiers (185 MB + 1.04 GB). Not
  downloaded: exceeds the unapproved-download budget. Needs approval.
* **History-dependent query policies** (e.g., choose which candidate to verify
  next from past verdicts) — forbidden on a replayed table; need a live online
  runner. No GPU or API credentials in this environment → not possible here.
* **Code domain** with independent hidden tests and repeated verifier calls —
  no suitable released traces identified within budget.
* **A manuscript** — not written; no result supports a paper.

## Deviations and process notes

* **D1** (dev-driven, before test): the calibrated posterior over the four
  GPQA letters stays in 0.25–0.45 for most dev states (global temperature
  calibration on a task where majority vote is right only ~50% of the time), so
  the pre-registered stopping thresholds τ ≥ 0.5 made every calibrated
  stopping policy degenerate (never stop). The τ / AC-threshold grid was
  extended with {0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45}. Budgets,
  families, endpoints and statistics are unchanged.
* The answer/verdict parser was revised twice (formats `final answer is: X`
  and `Is the answer correct (No)`) after inspecting unparsed *output text*
  only, before any accuracy was computed. Remaining unparsed: 487 candidates
  (all truncated at ≥ 1024 tokens) and 8 verdicts.
* The released verifier prompt omits the answer options (question stem +
  solution only). This is a property of the source data, not changed here.
* Gold letters come from the released files; the original GPQA file is gated
  and was not used to re-check them. Gold letters are balanced (A 17, B 14,
  C 17, D 16).
* Split difficulty differs (single-sample error 0.572 dev vs 0.467 test), and
  the best fixed policy on test (`BAYES-CORR N64 V32`, 0.375) was the worst on
  dev (0.594): with 32 problems per split, per-policy differences of a few
  points are noise. Nothing was selected on test.

## Costs

| item | tokens | note |
|---|---|---|
| full response-table collection (paid by the original authors) | 222.1 M (17.8 M generation + 204.3 M verification) | ≈ 3.1e19 FLOPs at 2 · 70e9 · tokens |
| policy execution, per test problem | 1.1 k – 68.3 k | selected policies; see README |
| controller | ≤ 0.5 ms CPU / problem | negligible vs. one 70B token |
| this study | 0 LLM calls; 72 MB downloaded; ≈ 17 CPU-minutes total (3 table builds, 2 dev runs, 1 test run, analyses) | 4-core CPU container, no GPU |

## If this is revisited (requires new approval; not recommended by default)

The pre-registered question is answered on the approved data. A different,
smaller question — whether within-problem discrimination of LLM verifiers is
near chance across several generator–verifier pairs, and whether a second
verifier decorrelates it — would need ≥ 3 released multi-verifier trace sets
(~1–3 GB) and should be registered as a new study, not as a continuation.
