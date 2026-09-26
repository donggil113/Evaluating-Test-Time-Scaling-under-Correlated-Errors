# Closest prior work and what was actually read

"Read" states exactly what was consulted in this project (2026-09-26); nothing
beyond that is claimed about each paper.

| work | read | what it establishes | difference from N5 |
|---|---|---|---|
| **Partially Correlated Verifier Cascades in LLM Harnesses** (arXiv 2607.13918) | arXiv HTML v1 (theorems, experiments, limitations); code repo not inspected | Exact cascade posterior ℓ_k = ℓ_0 − ln m_k with per-instance false-accept propensity α ~ G; concave log-odds; polynomial (not exponential) reliability under Beta; blind-spot ceiling. Validation is **synthetic only**; the paper states the measurement on real generator–verifier pairs "is in progress". | N5 measures the same quantities on real traces (M1, M2) and asks whether they change cost–error decisions. The theory is theirs; N5's M2 is an empirical check of it, not a new theorem. |
| **Odds Law** (arXiv 2606.15712) | abstract only | Under conditionally independent gates, log-odds grow linearly in k. | Used only as the independence baseline (IND model). |
| **Variation in Verification** (Zhou et al., arXiv 2509.17995, ICLR 2026) | abstract, arXiv HTML v2 setup/metrics section, GitHub README | 64 candidates / problem, generative verifiers with **greedy** decoding (one verdict per candidate), pooled TPR/TNR/balanced accuracy and verification gain across 12 benchmarks, 15 verifiers. Data on HF (`YefanZhou98/LLMVerify-*`, not downloaded). | No repeated stochastic verdicts, no correlation analysis, no cost accounting. N5 has 32 stochastic repeats per candidate and reports pooled vs within-problem discrimination. |
| **When To Solve, When To Verify** (Singhi et al., arXiv 2504.01005) | arXiv HTML (setup, cost model, conclusions), GitHub README; **its released traces are N5's data** | Compute C(S,V) = S(1+λV); self-consistency is more compute-efficient than GenRM for most budgets; GenRM needs up to 8× compute to match SC; scale solutions 1.5–2× faster than verifications. Scores average V verdicts; no correlation analysis. | Same traces, different questions: N5 adds problem-level paired inference, a held-out split, calibrated/adaptive stopping, controller cost, and correlation measurements. N5's E1 result agrees with their conclusion. |
| **Adaptive Generate-Rank-Verify** (Dughmi et al., arXiv 2605.17609) | abstract, arXiv HTML v2 | Cost-sensitive adaptive search with a cheap reward and an **exact** verifier (answer match / hidden tests); ADAP within a constant factor of optimal. | Their verifier is perfect; N5's verifier is the imperfect, correlated object of study. |
| **What If We Allocate Test-Time Compute Adaptively?** (Bilal et al., arXiv 2602.01070) | arXiv HTML v4 | Per-problem adaptive choice of tools/strategies with PRM selection; FLOPs incl. verifier. | Different adaptivity (strategy choice, history-dependent); not replayable on a fixed table. |
| **Adaptive-Consistency** (Aggarwal et al., arXiv 2305.11860) | ar5iv HTML (stopping criterion) | Beta stopping rule ∫₀^0.5 p₂^{v₂}(1−p₂)^{v₁}dp₂ ≥ C, default C = 0.95. | Implemented as the `AC` baseline (`models.adaptive_consistency_prob`, equivalent closed form). |
| **The Limits of Inference Scaling Through Resampling** (Stroebl et al., arXiv 2411.17501) | abstract | Imperfect verifiers with false positives cap resampling accuracy; optimal attempts often < 10. | N5 observes the same ceiling mechanism (34% of wrong candidates accepted by all 32 calls). |
| **Are More LLM Calls All You Need?** (Chen et al., arXiv 2403.02419) | abstract | Majority-vote accuracy can be non-monotone in the number of calls due to mixed query difficulty. | N5's MV curve on test is flat/non-monotone (0.467 → 0.500 error from N=1 to N=64), consistent with this. |
| **BEACON** (arXiv 2510.15945) | abstract | Bayesian optimal stopping for sampling with reward posteriors. | Does not model correlated verifier evidence (per abstract); not reimplemented. |
| **Sample, Scrutinize and Scale** (Zhao et al., arXiv 2502.01839) | abstract | Self-verification of sampled candidates scales; frontier models verify weakly out of the box. | Not reimplemented. |

## What would have been new

1. A real-data measurement of repeated-verdict correlation and of the PCVC
   moment identity (PCVC lists it as open). → N5 did it (M1, M2); it
   **confirms** PCVC's predictions.
2. A cost–error improvement from calibrated or correlation-aware stopping over
   the best fixed allocation, overhead included. → **Not found** (E2, E3).
3. A positive marginal value of verification at matched cost. → **Not found**
   (E1), consistent with Singhi et al.
