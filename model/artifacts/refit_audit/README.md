# Refit audit

This diagnostic quantifies the impact of two logic fixes on Change D MAPE.

**A1. Fold-aware winsorisation.** The original code computes 2.5th and 97.5th percentile
clip bounds on the full filtered frame (including the test year) and uses the clipped
series as both the training target and the test actual. A1 instead recomputes the bounds
from training rows only per fold, and applies those bounds to both train and test. This
removes a mild target leak where future test rows help define their own clip bounds.

**A2. Duan smearing on RF predictions (rejected).** The random forest is trained on the
log target and naively exponentiated, which is negatively biased under heteroskedastic
residuals. A2 multiplies `exp(log_prediction)` by `mean(exp(train_residuals))`. In practice
this **worsens** RF MAPE in this sample (see table below), because the smearing factor is
computed from high-variance in-sample residuals and over-corrects predictions whose
log-scale variance on unseen rows is already compressed by tree averaging. The correction
is therefore not applied in `rf_test.py`; it is kept only as a reference column here.

## Results

| Scenario | OLS rolling mean | OLS headline | OLS random 5-fold | RF rolling mean | RF headline |
|---|---|---|---|---|---|
| Current code | 87.67% | 74.12% | 84.67% | 97.93% | 63.30% |
| A1 only | 87.07% | 74.11% | 84.12% | 97.93% | 63.30% |
| A2 only | 87.67% | 74.12% | 84.67% | 123.74% | 79.71% |
| A1 + A2 | 87.07% | 74.11% | 84.12% | 123.74% | 79.71% |

## Headline fold fragility

The 2026 headline fold contains roughly 17 rows. OLS rolling MAPE standard deviation across the four folds under the A1 fix is 15.24 percentage points. A single large misprediction on 17 rows can move headline MAPE by several percentage points, so the `headline_mape <= 55` promotion gate in `refit_stage_two.py` should be read as noisy.

## Interpretation

A1 produces a small improvement (~0.5 percentage points of rolling-origin mean MAPE),
confirming that the original full-sample winsorisation was only mildly leaky. A2 tested
a Duan smearing correction for the RF and was rejected as it degraded out-of-sample
performance. Neither fix closes the gap to the ~10% target. The remaining error is driven
by the feature ceiling of the observed sample (size, asset type, country group, year), not
by logic bugs. See the mock completeness benchmark for the analysis of how much additional
feature explanatory power would be needed to approach a 10% MAPE target.
