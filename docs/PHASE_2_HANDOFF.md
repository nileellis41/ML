# Phase 2 Handoff Document

**From:** Phase 1 (price-based ML allocation study)  
**To:** Phase 2 (fundamentals + macro signal study)  
**Date:** 2026-05-27  
**Git tag:** `phase-1-final`

---

## What Phase 1 Established

- **Price-based ML signals do not beat passive baselines on sector ETFs (2023–2026).** Eight
  configurations (linear, logistic, LSTM, PCA × base/regime) were tested over 640 out-of-sample
  trading days; 0 of 32 Diebold-Mariano comparisons survived Bonferroni correction (best raw
  p = 0.071, logistic/regime vs equal-weight).

- **Regime conditioning reduces drawdown but not return.** Across all three architectures that
  used it, HMM-regime variants cut maximum drawdown by 1.6–7.6 pp while leaving annualised return
  roughly unchanged. The regime HMM is a validated risk-management tool, not an alpha source.

- **LSTM underperforms linear models on this dataset.** Despite higher model complexity, the LSTM
  posts the two lowest Sharpe ratios among all eight ML configurations (0.86, 1.00). Linear ridge
  and logistic models are the appropriate starting point for Phase 2 baselines.

---

## Baselines Phase 2 Must Beat

Phase 2 must demonstrate statistically significant improvement (DM test, family-wise corrected
p < 0.05) over the following hurdles, ordered by difficulty:

| Rank | Baseline | Sharpe [95 % CI] | Ann. Return | Max DD |
|------|----------|-------------------|-------------|--------|
| Hardest | logistic/regime (best Phase 1 ML) | 2.06 [0.62, 3.53] | 25.70 % | −16.11 % |
| | momentum_12_1 | 1.89 [0.56, 3.06] | 22.57 % | −15.61 % |
| | equal_weight | 1.90 [0.58, 3.14] | 19.37 % | −13.51 % |
| | spy_bh | 1.75 [0.39, 2.95] | 26.60 % | −20.75 % |
| Easiest | sixty_forty | 1.53 [0.26, 2.73] | 16.86 % | −13.82 % |

**Minimum detectable effect (80 % DM power, T = 640):** ≥ 8 % annualised return differential
over equal-weight. Phase 2 signal design should target a plausible alpha of ≥ 8 % before
significance testing is a realistic goal.

**Primary hurdle:** Beat logistic/regime at the family-wise corrected p < 0.05 level over Phase 2
comparisons. If Phase 2 tests K new configurations, the corrected threshold is p < 0.05 / (K × 4).

---

## Methodology Assets Inherited (use as-is)

| Module | Path | Description |
|--------|------|-------------|
| Walk-forward harness | `src/training/walk_forward.py` | 30-fold expanding window; add new folds for Phase 2 test data |
| Baselines | `src/baselines/portfolios.py` | All four baselines; do not modify |
| Metrics | `src/evaluation/metrics.py` | Full suite incl. bootstrap Sharpe CI; do not modify |
| Significance | `src/evaluation/significance.py` | DM + Bonferroni + BH; extend N_trials for Phase 2 |
| Deflated Sharpe | `src/evaluation/deflated_sharpe.py` | PSR / DSR / MinTRL; update N_trials |
| Regime HMM | loaded via reporting pipeline | 4-state labels on all test dates; available for Phase 2 conditioning |
| Reporting | `src/evaluation/reporting.py` | Master table, figures, metadata; extend for new columns |
| Seeds | `src/utils/seeds.py` | Bit-identical reproducibility; do not modify |
| Test suite | `tests/` | 78 tests; add Phase 2 tests, do not break existing |
| Frozen dependencies | `requirements_lock.txt` | Pin new additions; do not downgrade existing packages |

---

## Assets NOT Inherited

| Asset | Reason |
|-------|--------|
| `src/models/` — current implementations | Treat as reference only; Phase 2 models should be new modules |
| `config/experiments.yaml` — Phase 1 config | Start a fresh `config/phase2_experiments.yaml` |
| Phase 1 parquet predictions | Kept for comparison but not inputs to Phase 2 training |
| `results/` figures and tables | Phase 1 outputs are archived under `phase-1-final` tag; Phase 2 writes to a separate `results/phase2/` directory |

---

## Phase 2 Scope

Phase 2 should incorporate at least one of the following fundamentally different signal sources:

1. **Earnings and revenue surprise** — analyst estimate revisions and beat/miss relative to
   consensus. Requires EDGAR or a data provider with fundamental coverage.

2. **Macro factor exposure** — sector-level sensitivity to yield curve slope, credit spreads,
   and commodity prices. FRED series `GS10`, `GS2`, `BAMLH0A0HYM2`, `DCOILWTICO` are already
   loaded via the existing data pipeline.

3. **Options-implied information** — put/call ratios or implied vol skew as a sentiment proxy.

The signal must be available *at the time of the trading decision* (no look-ahead). All new data
sources must be gated by the fold boundary in the walk-forward harness.

---

## Pre-Registration Requirement

Before any Phase 2 model is trained on test data, the following must be documented in a
`docs/PHASE_2_PREREGISTRATION.md` file and committed:

1. Signal hypothesis: precisely what information is being used and why it should predict returns.
2. Feature list: exact feature names and construction formulas.
3. Model architecture(s): type, hyperparameter grid, selection criterion.
4. Portfolio construction rule: how model outputs map to weights (may differ from above-median).
5. Primary evaluation metric and test: which DM comparison is the pre-specified primary test.
6. Number of configurations K to be evaluated (determines Bonferroni threshold).

**No hyperparameter changes, architecture changes, or feature additions after the first test-set
prediction has been generated.** All such changes restart the experiment from a fresh fold split.

---

*Phase 1 results are archived and reproducible. Questions about Phase 1 implementation details
should be answered by reading the code and `docs/PHASE_1_FINAL.md` rather than re-running
exploratory experiments.*
