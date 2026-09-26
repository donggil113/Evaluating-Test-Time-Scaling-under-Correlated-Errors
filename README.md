# The Marginal Value of Verification: Evaluating Test-Time Scaling under Correlated Errors

**Status: NO_GO** — see [`STATUS.md`](STATUS.md). No manuscript was written.

## Question

When an imperfect LLM verifier's repeated verdicts are correlated, what does
one more verifier call buy compared with one more generation, once generator,
verifier and controller costs are counted — and can calibrated stopping turn
that into a better cost–error trade-off than fixed allocations?

Closest prior work and what was read: [`docs/related_work.md`](docs/related_work.md).
Frozen protocol (committed before any test evaluation): [`docs/protocol.md`](docs/protocol.md).

## Data (real traces, reused)

Released traces of Singhi et al. 2025 (*When To Solve, When To Verify*, HF org
`sc-genrm-scaling`): 64 GPQA-Diamond questions; generator Llama-3.3-70B-Instruct
(256 candidates / question, T = 0.7); verifier Llama-3.3-70B-Instruct as
GenRM-Base (self-verification), 32 calls on each of the first 64 candidates.
Frozen table: `data/tables/gpqa_l33_70b/` (64 × 256 candidates, 131,072
verifier calls, token counts from the Llama-3 tokenizer; gold letters in a
separate file read only by evaluation code). Split: 32 dev / 32 test by
salted question hash.

## Runner

`src/ttsce/env.py` replays the table under a fixed candidate pool and
history-independent query schedules: a policy may only execute the next
pre-committed query or stop, so replay is exactly equivalent to live
querying. Policies receive revealed answers/verdicts only;
`tests/test_replay.py` checks that decisions are invariant when every
unrevealed cell is randomised (with a negative control showing the test
detects a peeking policy).

## Results (test split, 32 problems × 20 order seeds; selection on dev only)

Cost = prompt + completion tokens of all generator and verifier calls
(c̄_gen ≈ 1.1k tokens = one generation). Error = 1 − accuracy.

| budget (× c̄_gen) | best GEN on dev → test err (cost) | best VER on dev → test err (cost) | E1: VER − GEN [95% CI], Holm p |
|---|---|---|---|
| 2 | `FIX:MV:N1:V0` 0.467 (1.1k) | none within budget | — |
| 4 | `AC:G256:c0.8` 0.495 (4.9k) | `FIX:BAYES-CORR:N1:V1` 0.467 (2.6k) | -0.028 [-0.069, +0.009], 1.00 |
| 8 | `AC:G256:c0.85` 0.505 (6.2k) | `FIX:BAYES-CORR:N2:V2` 0.477 (8.2k) | -0.028 [-0.083, +0.023], 1.00 |
| 16 | `AC:G256:c0.85` 0.505 (6.2k) | `STOP:CORR:GV64x2:tau0.26` 0.467 (11.8k) | -0.037 [-0.102, +0.023], 1.00 |
| 32 | `FIX:MV:N16:V0` 0.486 (17.1k) | `FIX:BAYES-CORR:N8:V2` 0.463 (32.8k) | -0.023 [-0.109, +0.053], 1.00 |
| 64 | `FIX:MV:N16:V0` 0.486 (17.1k) | `FIX:BAYES-CORR:N8:V4` 0.459 (57.0k) | -0.027 [-0.111, +0.053], 1.00 |
| 128 | `FIX:MV:N64:V0` 0.500 (68.3k) | `FIX:BAYES-CORR:N8:V4` 0.459 (57.0k) | -0.041 [-0.153, +0.075], 1.00 |
| 256 | `FIX:MV:N64:V0` 0.500 (68.3k) | `FIX:BAYES-CORR:N8:V4` 0.459 (57.0k) | -0.041 [-0.153, +0.075], 1.00 |

| budget | E2: STOP − FIXED [95% CI], Holm p | E3: STOP-CORR − STOP-IND [95% CI], Holm p |
|---|---|---|
| 2 | -0.002 [-0.005, +0.000], 1.00 | — (no policy within budget) |
| 4 | +0.028 [-0.009, +0.069], 1.00 | — (no policy within budget) |
| 8 | +0.030 [+0.006, +0.056], 0.09 | — (no policy within budget) |
| 16 | -0.005 [-0.023, +0.014], 1.00 | -0.030 [-0.098, +0.037], 1.00 |
| 32 | +0.005 [-0.031, +0.041], 1.00 | -0.030 [-0.100, +0.042], 1.00 |
| 64 | +0.008 [-0.022, +0.037], 1.00 | -0.030 [-0.100, +0.042], 1.00 |
| 128 | -0.058 [-0.178, +0.066], 1.00 | -0.055 [-0.155, +0.042], 1.00 |
| 256 | -0.058 [-0.178, +0.066], 1.00 | -0.055 [-0.155, +0.042], 1.00 |

Test-state calibration (dev-fit posteriors): IND log-loss 1.327, ECE 0.148; CORR log-loss 1.320, ECE 0.191 (uniform = 1.386).

### Measurements (all problems; problem-cluster bootstrap 95% CIs)

| measurement | all 64 | dev 32 | test 32 |
|---|---|---|---|
| single-call accept rate, correct candidates (TPR) | 0.817 [0.721, 0.895] | 0.829 [0.716, 0.924] | 0.807 [0.638, 0.928] |
| single-call accept rate, wrong candidates (FPR) | 0.628 [0.525, 0.723] | 0.630 [0.486, 0.752] | 0.625 [0.466, 0.761] |
| repeat correlation ρ, correct candidates | 0.781 [0.668, 0.841] | 0.739 [0.599, 0.806] | 0.811 [0.608, 0.883] |
| repeat correlation ρ, wrong candidates | 0.716 [0.653, 0.765] | 0.707 [0.622, 0.766] | 0.726 [0.617, 0.790] |
| wrong candidates accepted by all 32 calls | 0.336 [0.252, 0.427] | 0.344 [0.227, 0.478] | 0.327 [0.201, 0.452] |
| correct candidates rejected by all 32 calls | 0.090 [0.031, 0.157] | 0.068 [0.019, 0.142] | 0.109 [0.011, 0.240] |
| AUROC, 1 verdict (pooled over problems) | 0.597 [0.536, 0.658] | 0.598 [0.526, 0.675] | 0.598 [0.497, 0.695] |
| AUROC, mean of 32 verdicts (pooled) | 0.663 [0.583, 0.738] | 0.665 [0.568, 0.758] | 0.663 [0.527, 0.784] |
| AUROC, mean of 32 verdicts (within problem, mean) | 0.545 [0.470, 0.612] | 0.560 [0.476, 0.646] | 0.529 [0.417, 0.651] |
| share of false-accept signal variance explained by (problem, answer) | 0.624 [0.489, 0.726] | 0.587 [0.384, 0.705] | 0.666 [0.421, 0.800] |

Held-out gate test (fit on repeats 1–8, evaluate on repeats 9–32):

| k gates (held-out repeats) | wrong: empirical all-accept | Beta fit on 8 repeats | independence | correct: empirical | Beta | indep. |
|---|---|---|---|---|---|---|
| 1 | 0.628 | 0.617 | 0.63 | 0.817 | 0.808 | 0.82 |
| 2 | 0.563 | 0.551 | 0.39 | 0.787 | 0.777 | 0.67 |
| 4 | 0.502 | 0.494 | 0.15 | 0.757 | 0.748 | 0.44 |
| 8 | 0.446 | 0.443 | 0.024 | 0.725 | 0.721 | 0.2 |
| 16 | 0.392 | 0.399 | 0.00057 | 0.689 | 0.696 | 0.038 |
| 24 | 0.363 | 0.375 | 1.4e-05 | 0.666 | 0.682 | 0.0075 |

Posterior log-odds of correctness after k accepts (all 64): k=1: 0.15 (indep. 0.15), k=4: 0.29 (indep. 0.94), k=8: 0.37 (indep. 1.99), k=24: 0.49 (indep. 6.22)


Oracle coverage (gold among the first n candidates): n=1 45%, 4 69%, 16 77%,
64 84%, 256 92% — while majority vote stays near 50% accuracy.

## Reading

* Verification adds no detectable accuracy at any matched budget, stopping
  rules do not beat fixed allocations, and modelling the correlation does not
  help on held-out problems.
* The correlation structure matches the partially-correlated cascade theory
  (arXiv 2607.13918) closely on real data, but that is a confirmation of an
  existing prediction, together with existing empirical conclusions
  (Singhi et al.; Stroebl et al.). Hence NO_GO.

## Reproduce (CPU only, ~20 min)

```bash
pip install -r requirements.txt
scripts/fetch_data.sh raw            # 72 MB, sha256-checked
python scripts/build_table.py --raw raw/gpqa --tokenizer raw/llama3_tokenizer.json --out data/tables/gpqa_l33_70b
python -m pytest tests -q
python scripts/run_experiment.py --phase dev    # fit + select on dev
python scripts/run_experiment.py --phase test   # evaluate frozen selection once
python scripts/analyze_correlation.py           # M1–M3
python scripts/describe_grid.py                 # descriptive grids, cost accounting
python scripts/make_manifest.py
```

Outputs: `results/gpqa_l33_70b/`, run manifest `results/manifest.json`.
