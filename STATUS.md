# STATUS — N5 (The Marginal Value of Verification)

## Deviations from docs/protocol.md (all made on dev, before any test run)

* **D1** (dev-driven, before test): the calibrated posterior over the four
  GPQA letters stays in 0.25–0.45 for most dev states (global temperature
  calibration on a task where majority vote is right only ~50% of the time), so
  the pre-registered stopping thresholds τ ≥ 0.5 made every calibrated
  stopping policy degenerate (never stop). The τ / AC-threshold grid was
  extended with {0.26, 0.28, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45}. Budgets,
  families, endpoints and statistics are unchanged.
