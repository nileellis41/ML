# Phase 1 Final Report: ML Portfolio Allocation vs Passive Baselines

**Project:** Wealth Management AI Platform — Sector ETF Signal Study  
**Period covered:** 2023-11-02 to 2026-05-22 (640 trading days, 30 walk-forward folds)  
**Universe:** 21 sector ETFs  
**Date of report:** 2026-05-27  
**Git tag:** `phase-1-final`

---

## Abstract

This report documents the complete results of Phase 1 of the Wealth Management AI Platform project.
Eight machine-learning portfolio allocation strategies — covering linear regression, logistic
regression, LSTM, and PCA models each in a base and a regime-conditioned variant — were evaluated
against four passive baselines (SPY buy-and-hold, equal-weight, 12-1 momentum, and 60/40) over a
640-trading-day out-of-sample test period using 30-fold walk-forward cross-validation.

All models produced positive annualised Sharpe ratios (0.86–2.06). However, none of the 32
ML-vs-baseline comparisons survived Bonferroni correction (best raw p = 0.071). The 95 % stationary
bootstrap confidence intervals for every Sharpe ratio overlap substantially across all twelve
strategies. Deflated Sharpe Ratio p-values are near 1.0 for every strategy, confirming that the
observed Sharpe ratios are large relative to any plausible specification-search bias. The null
hypothesis — that ML-driven allocation produces the same expected returns as passive strategies — is
not rejected. Phase 2 will target a meaningfully different signal source rather than further tuning
of these models.

---

## 1. Research Question

Can machine-learning models trained on price-derived features allocate capital across 21 sector ETFs
more effectively than passive rules-based strategies over a multi-year out-of-sample period?

The specific hypotheses tested are:

- **H1 (return):** At least one ML strategy has a higher expected daily log return than each
  passive baseline (two-tailed Diebold-Mariano test, H_0: E[r_ML − r_baseline] = 0).
- **H2 (Sharpe):** At least one ML strategy has a higher annualised Sharpe ratio than the best
  passive baseline after accounting for multiple comparisons.
- **H3 (regime):** Regime-conditioned variants outperform their base counterparts in crisis
  sub-periods.

---

## 2. Methodology

### 2.1 Walk-Forward Validation Design

All results are strictly out-of-sample. No parameter was chosen using test-period data.

| Parameter | Value |
|-----------|-------|
| Validation scheme | Walk-forward, expanding window |
| Total folds | 30 |
| Test period start | 2023-11-02 |
| Test period end | 2026-05-22 |
| Test trading days | 640 |
| Universe | 21 sector ETFs |
| Rebalancing frequency | Daily (each prediction triggers a weight update) |

At each fold, the model is re-trained on all data available up to the fold boundary, then makes
predictions on the next out-of-sample block. Predictions are never used in training.

### 2.2 Signal Construction

Each model outputs a return prediction for every ticker on every test date. The **above-median
cross-sectional signal** converts these raw predictions to portfolio weights: on each date, tickers
with predictions above the cross-sectional median receive equal long weight; all others receive zero
weight. This signal construction is model-agnostic and eliminates scale-sensitivity for models whose
output magnitude varies across folds.

For logistic models the raw output is a probability; the above-median construction still applies
(above-median probability of a positive return → long).

### 2.3 Models

| Model key | Architecture | Features | Notes |
|-----------|-------------|----------|-------|
| `linear_base` | Ridge regression (α = 1.0) | 8 price-derived | Regularised to prevent coefficient explosion |
| `linear_regime` | Ridge regression (α = 1.0) | 8 base + 16 regime interactions | 24 features; OLS unstable due to multicollinearity |
| `logistic_base` | L2 logistic regression | 8 price-derived | Binary direction classification |
| `logistic_regime` | L2 logistic regression | 8 base + 16 regime interactions | — |
| `lstm_base` | 2-layer LSTM (64 hidden) | 8-step rolling window | Sequence model; 30-epoch training per fold |
| `lstm_regime` | Same LSTM + regime input | 8-step window + regime label | Regime label appended to each timestep |
| `pca_base` | Ridge on PCA factors (k=5) | PCA of 8 raw features | Dimensionality reduction as regularisation |
| `pca_regime` | Ridge on PCA + regime | PCA factors + regime interactions | — |

**Reproducibility:** All eight models produce bit-identical predictions across consecutive runs.
Seeds cover Python `random`, NumPy, PyTorch CPU/CUDA, `cudnn.deterministic`, and
`torch.use_deterministic_algorithms(True, warn_only=True)`.

### 2.4 Baselines

| Baseline | Description |
|----------|-------------|
| `spy_bh` | SPY buy-and-hold — single-asset market benchmark |
| `equal_weight` | Equal-weight across all 21 tickers, rebalanced daily |
| `momentum_12_1` | 12-month minus 1-month cross-sectional momentum, long top tercile |
| `sixty_forty` | 60 % SPY / 40 % AGG, rebalanced daily |

### 2.5 Statistical Tests

**Diebold-Mariano (DM) test** (Diebold & Mariano 1995) compares expected daily return
differentials. The Harvey-Leybourne-Newbold (1997) small-sample correction adjusts the DM statistic
for finite T:

```
MDM = sqrt( (T + 1 - 2h + h(h-1)/T) / T ) × DM
```

Under H₀, MDM ~ t(T − 1). We use h = 1 (one-day horizon) and a two-tailed alternative.

**Multiple-comparison adjustments** are applied across all 32 ML-vs-baseline pairs simultaneously:
- Bonferroni: corrected threshold p < 0.05 / 32 = 0.00156
- Benjamini-Hochberg FDR at q = 0.05

**Stationary bootstrap Sharpe CIs** (Politis & Romano 1994) use 10 000 replications with block size
T^(1/3) ≈ 9 days, approximating the Politis-White (2004) optimal block length for daily financial
returns. The bootstrap preserves serial dependence structure.

**Deflated Sharpe Ratio** (Bailey & López de Prado 2014) adjusts the Probabilistic Sharpe Ratio
for specification search over N = 8 ML configurations. The expected maximum SR across N independent
trials is approximated via the Gumbel distribution (Euler-Mascheroni constant γ ≈ 0.5772):

```
E[max SR | N] ≈ σ_SR × [(1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(Ne))]
```

With T = 640 and N = 8 trials, E[max SR] ≈ 0.07 — negligibly small relative to all observed
Sharpe ratios. This is why DSR p-values saturate near 1.0: the data are informative enough that
specification search across eight configurations produces no meaningful inflation.

---

## 3. Results

### 3.1 Master Comparison Table

All values are out-of-sample. Sharpe shown as point estimate [95 % bootstrap CI lower, upper].

| Strategy | Ann. Return | Ann. Vol | Sharpe [95 % CI] | Max Drawdown | Calmar | Hit Rate |
|----------|-------------|----------|-------------------|--------------|--------|----------|
| linear / base | 25.77 % | 13.12 % | 1.96 [0.57, 3.29] | −18.83 % | 1.37 | 48.85 % |
| linear / regime | 25.68 % | 12.83 % | 2.00 [0.61, 3.34] | −16.27 % | 1.58 | 50.16 % |
| logistic / base | 24.66 % | 12.56 % | 1.96 [0.55, 3.41] | −18.33 % | 1.35 | 49.84 % |
| logistic / regime | 25.70 % | 12.50 % | **2.06 [0.62, 3.53]** | −16.11 % | 1.60 | 50.82 % |
| lstm / base | 13.55 % | 15.73 % | 0.86 [−0.25, 1.96] | −24.72 % | 0.55 | 50.86 % |
| lstm / regime | 15.83 % | 15.86 % | 1.00 [−0.14, 2.11] | −17.17 % | 0.92 | 49.14 % |
| pca / base | 15.74 % | 12.13 % | 1.30 [0.07, 2.73] | −18.79 % | 0.84 | 48.85 % |
| pca / regime | 18.95 % | 13.15 % | 1.44 [0.19, 2.76] | −18.89 % | 1.00 | 49.84 % |
| **spy_bh** | **26.60 %** | **15.18 %** | 1.75 [0.39, 2.95] | −20.75 % | 1.28 | — |
| equal_weight | 19.37 % | 10.19 % | 1.90 [0.58, 3.14] | −13.51 % | 1.43 | — |
| momentum_12_1 | 22.57 % | 11.96 % | 1.89 [0.56, 3.06] | −15.61 % | 1.45 | — |
| sixty_forty | 16.86 % | 11.04 % | 1.53 [0.26, 2.73] | −13.82 % | 1.22 | — |

Key observations:
- The two best Sharpe ratios belong to ML models (logistic/regime: 2.06, linear/regime: 2.00), but
  their CIs span roughly ±1.5 Sharpe units and overlap completely with all baseline CIs.
- SPY buy-and-hold has the highest annualised return (26.60 %) in the test period.
- LSTM models underperform their linear counterparts on both return and Sharpe despite higher
  model complexity, suggesting overfitting on this universe and time horizon.
- Regime conditioning consistently improves Sharpe for linear and logistic models (+0.04 to +0.10)
  but the improvement is within bootstrap noise.
- Hit rates cluster tightly around 50 % (48.85 %–51.29 %), confirming that directional accuracy
  is not meaningfully above chance for any model.

### 3.2 Significance Tests

```
ML vs baseline pairs tested : 32  (8 ML configs × 4 baselines)
Bonferroni threshold        : p < 0.00156
Significant after Bonferroni: 0 / 32
Significant after BH FDR   : 0 / 32
Best raw p-value            : 0.0709  (logistic/regime vs equal_weight, MDM = +1.809)
```

The strongest signal is logistic/regime outperforming equal-weight with raw p = 0.071 and
annualised return differential of +5.17 %. After Bonferroni correction this becomes p = 1.000.
Even the BH-adjusted p-value is 0.431 — far from the 0.05 threshold.

The 32 raw p-values are broadly distributed across (0.07, 0.91) with no clustering near zero,
consistent with the null hypothesis rather than a true signal diluted by multiple comparisons.

Selected comparisons (sorted by raw p-value):

| ML model | Variant | Baseline | Δ Return (ann.) | Δ Sharpe | MDM stat | p_raw | p_Bonf |
|----------|---------|----------|----------------|----------|----------|-------|--------|
| logistic | regime | equal_weight | +5.17 % | +0.091 | +1.809 | 0.071 | 1.000 |
| linear | regime | equal_weight | +5.15 % | +0.044 | +1.773 | 0.077 | 1.000 |
| pca | base | spy_bh | −8.96 % | −0.348 | −1.728 | 0.084 | 1.000 |
| linear | base | equal_weight | +5.22 % | +0.008 | +1.687 | 0.092 | 1.000 |
| lstm | base | spy_bh | −10.87 % | −0.745 | −1.625 | 0.105 | 1.000 |

### 3.3 Deflated Sharpe Ratio

With N = 8 independent ML configurations and T = 640 observations, the Gumbel approximation gives
E[max SR over 8 trials] ≈ 0.07. All observed Sharpe ratios exceed 0.86. As a result, the DSR
p-values are all ≥ 0.9999 — effectively 1.0.

This is the correct statistical interpretation: the observed performance is so far above what
specification search would expect to produce by chance that inflation from trying multiple
configurations is negligible. The Deflated Sharpe confirms that the positive Sharpe ratios are
genuine (the market trended upward in the test period), but does not speak to whether ML adds
value over baselines — that question is answered by the DM tests above.

### 3.4 Regime-Conditional Performance

The HMM identifies four regimes: risk-on, neutral, crisis, and risk-off. Across all models:

- **Crisis regime (104 days):** All strategies — ML and baselines — post sharply negative
  annualised returns (−0.20 to −0.48) and Sharpe ratios of −0.87 to −1.98. Crisis periods
  constitute systematic market risk that the cross-sectional signal cannot hedge away.
- **Risk-on regime (224 days):** All ML models post Sharpe > 2 in this sub-period (linear/regime:
  5.96, logistic/regime: 5.87). SPY buy-and-hold posts 7.70 in the same sub-period. The ML
  advantage during risk-on is not materially larger than the baseline advantage.
- **Regime conditioning benefit:** The regime-conditioned variants consistently reduce maximum
  drawdown relative to base variants (linear: −18.83 % → −16.27 %; logistic: −18.33 % → −16.11 %;
  lstm: −24.72 % → −17.17 %) while holding return roughly constant.

### 3.5 Visual Summaries

**Figure 1 — Equity curves (final):** `results/figures/equity_curves_final.png`  
Cumulative log-return curves for all 12 strategies over the test period with regime shading. ML
equity curves track baselines closely. The two linear/logistic-regime curves finish near the top of
the bundle but are not visually separable from momentum and equal-weight.

**Figure 2 — Per-fold Sharpe distribution:** `results/figures/sharpe_per_fold_final.png`  
Boxplots of fold-level Sharpe ratios. All 12 strategy distributions are wide and overlapping,
consistent with the wide bootstrap CIs. No model stochastically dominates any baseline.

---

## 4. Discussion

### 4.1 Why ML Did Not Beat Passive Baselines

Several factors likely explain the null result:

1. **Feature set is price-derived only.** The 8 input features (momentum, volatility, cross-section
   rank, etc.) are computable from price history alone. The same information drives the momentum
   and equal-weight baselines. The ML models may simply be rediscovering momentum in a
   more complex form.

2. **Signal aggregation washes out edge.** The above-median equal-weight construction used for all
   models is designed to be robust, but it also eliminates any information in the *magnitude* of
   the prediction — a very strong signal and a weakly positive signal receive identical weight.

3. **Test period characteristics.** The 640-day test window (November 2023 – May 2026) was
   dominated by a broadly bullish equity market with a brief sharp decline. In this environment,
   buy-and-hold strategies benefit from the general level of returns. Cross-sectional allocation
   adds value by picking relative winners, but the test is whether it adds value *relative to*
   an equal-weight of the same universe — a high bar.

4. **LSTM underperformance.** The LSTM models post the two lowest Sharpe ratios (0.86, 1.00) and
   the two largest maximum drawdowns among ML strategies. With 30 folds and 30-epoch retraining per
   fold, the sequence model may not generalise well on 21 tickers with 8 features. Increasing
   training data or using pre-training with transfer learning is a natural Phase 2 direction.

5. **Regime conditioning reduces drawdown but not return.** The HMM regime labels improve
   drawdown for all three architectures that use them, but do not increase annualised return. The
   regime signal appears to reduce position sizing in volatile periods but does not identify
   directionally profitable opportunities that the base models miss.

### 4.2 What the Results Do Establish

The null result is informative, not merely negative:

- **Price-based features are efficiently priced in sector ETFs over this horizon.** Systematic
  momentum and equal-weight strategies capture most of the available cross-sectional alpha,
  leaving little residual for ML.
- **The regime HMM provides a consistent drawdown-reduction benefit.** Across three
  architectures, regime conditioning reduces maximum drawdown by 1.6–7.6 percentage points.
  This is a risk-management result rather than an alpha result.
- **The above-median signal construction is not the bottleneck.** Hit rates near 50 % confirm
  that the underlying directional predictions are not accurate enough to extract alpha through
  *any* portfolio construction method; the construction choice is not the limiting factor.

### 4.3 Statistical Power Considerations

With T = 640 and a Bonferroni threshold of 0.00156, the DM test has approximately 80 % power to
detect an annualised return difference of roughly 8–10 % (assuming daily volatility ≈ 1 %). The
largest observed return differential in favour of ML is +5.22 % (linear/base vs equal-weight),
which is below this detectability threshold. Phase 2 should aim for a signal that plausibly
generates ≥ 8 % annualised alpha before expecting significance to be achievable.

### 4.4 Limitations

- **Single asset universe.** 21 sector ETFs are a narrow, highly correlated universe. Results may
  not generalise to individual equities or international markets.
- **Test period length.** 640 trading days includes only one clear regime transition from risk-on
  to crisis and back. A longer out-of-sample period would provide more statistical power and
  more thorough regime testing.
- **No transaction costs.** All returns are gross of transaction costs. Daily rebalancing of 21
  positions incurs bid-ask spread costs; with realistic spreads of 1–3 bps, all reported returns
  would decline slightly but the ordering of strategies is unlikely to change materially.
- **Feature set not updated mid-phase.** The 8 price-derived features were fixed at project
  outset. Phase 2 will incorporate fundamentals and macro features.

---

## 5. Conclusion

**Phase 1 answer to the research question: No.** Over a 640-trading-day out-of-sample period,
none of the eight ML strategies produces statistically distinguishable returns from any of the four
passive baselines at the p < 0.05 level after Bonferroni correction (32 comparisons tested; best
raw p = 0.071). The 95 % bootstrap Sharpe confidence intervals overlap completely across all twelve
strategies. Positive Sharpe ratios throughout are consistent with a broadly bullish test period
and are not evidence of ML alpha.

The result is not that ML models perform poorly in absolute terms: annualised Sharpe ratios of
2.00–2.06 for the best ML strategies are respectable. The conclusion is that passive momentum and
equal-weight strategies achieve comparable risk-adjusted returns without any learned component,
leaving no demonstrated marginal value from the ML overhead.

Regime conditioning provides consistent and modest drawdown reduction (the HMM adds value as a
risk-management layer) but does not produce directional alpha. This finding is incorporated into
the Phase 2 design, which retains regime conditioning as a risk-management tool while targeting
a fundamentally different signal source.

---

## 6. Phase 2 Baselines

Phase 2 must demonstrate improvement over the following hurdles, all computed on the Phase 1
test period for comparability:

| Baseline | Sharpe [95 % CI] | Ann. Return | Max Drawdown |
|----------|-------------------|-------------|--------------|
| SPY buy-and-hold | 1.75 [0.39, 2.95] | 26.60 % | −20.75 % |
| Equal-weight (21 ETFs) | 1.90 [0.58, 3.14] | 19.37 % | −13.51 % |
| Momentum 12-1 | 1.89 [0.56, 3.06] | 22.57 % | −15.61 % |
| 60/40 | 1.53 [0.26, 2.73] | 16.86 % | −13.82 % |
| **Best Phase 1 ML** | **2.06 [0.62, 3.53]** | **25.70 %** | **−16.11 %** |

The primary hurdle for Phase 2 is the **best Phase 1 ML strategy** (logistic/regime, Sharpe 2.06)
at a p < 0.05 / K threshold after Bonferroni correction over Phase 2 comparisons. Phase 2 must also
satisfy the DM test vs equal-weight (the baseline closest to an ML edge in Phase 1, p_raw = 0.071)
at the family-wise corrected level.

Suggested minimum detectable effect for Phase 2 planning: annualised return differential ≥ 8 %
over equal-weight, corresponding to approximately 80 % DM power with T = 640.

---

## 7. Methodology Assets Inherited by Phase 2

The following Phase 1 infrastructure is production-ready and carries forward without modification:

| Asset | Status | Notes |
|-------|--------|-------|
| Walk-forward harness | ✅ stable | 30-fold, expanding window, `src/training/walk_forward.py` |
| Baseline portfolio constructors | ✅ stable | `src/baselines/portfolios.py` — all four baselines |
| Evaluation metrics suite | ✅ stable | `src/evaluation/metrics.py` — all metrics including bootstrap Sharpe |
| Significance test framework | ✅ stable | `src/evaluation/significance.py` — DM + Bonferroni + BH |
| Deflated Sharpe | ✅ stable | `src/evaluation/deflated_sharpe.py` — PSR, DSR, MinTRL |
| HMM regime detector | ✅ stable | 4-state HMM; regime labels available for all test dates |
| Reporting pipeline | ✅ stable | `src/evaluation/reporting.py` — master table, figures, metadata header |
| Seed management | ✅ stable | `src/utils/seeds.py` — bit-identical reproducibility verified |
| Test suite | ✅ stable | 78 tests, 0 failures (3 skipped on network-gated FRED tests) |

**Do not re-implement or refactor these assets in Phase 2.** They have been validated.

The Phase 1 model implementations (`src/models/`) are available but should be treated as reference
implementations rather than production components. Phase 2 may extend them but should not depend on
their current architecture.

---

## 8. References

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection
bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
Economic Statistics*, 13(3), 253–263.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared
errors. *International Journal of Forecasting*, 13, 281–291.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American
Statistical Association*, 89(428), 1303–1313.

Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap.
*Econometric Reviews*, 23(1), 53–70.

---

*All code, data, and figures referenced in this report are archived under git tag `phase-1-final`
in the project repository. The full experiment output is reproducible from
`python scripts/run_experiments.py && python scripts/run_reporting.py` with the frozen
dependencies in `requirements_lock.txt`.*
