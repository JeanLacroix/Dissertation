# Mock completeness benchmark

This analysis creates a synthetic richer-data benchmark to estimate what predictive precision could be achievable if materially more deal-level information were observed. The starting point is the existing Change D sample and fitted structure. A mock target is generated as the sum of: (i) the fitted Change D signal already explained by observed covariates, (ii) up to eight synthetic standardised features (micro-location, building quality, lease quality, tenant covenant, capex need, refurbishment potential, credit spread, rental growth), and (iii) irreducible noise.

The synthetic features are not copied from external data. They are generated with group structure by country, asset type, and year plus idiosyncratic noise, then scaled so that they explain 50% of the current Change D residual variance in the base case. The remaining 50% is left as noise.

On the real target, Change D records a rolling-origin mean MAPE of 87.7%. On the base-case mock target, the observed-feature model records 75.7%, the +4 synthetic-feature model records 55.2%, and the +8 synthetic-feature model records 47.4%. The sensitivity sweep covers 30%, 50%, 70%, 85%, and 95% explainable residual-variance shares. This should be interpreted as a structured sensitivity benchmark rather than as a claim about true achievable production accuracy.

## How complete must the data be to reach 10% random 5-fold MAPE?

Assuming log-price residuals are approximately normal with variance sigma-squared, a random 5-fold mean MAPE of 10% implies a residual log-sigma of roughly 0.125 (residual variance 0.0155). The Change D sample has log-target variance 1.625, so reaching this MAPE requires a log-scale R-squared of at least **0.990**. The current Change D deployed fit achieves log-scale R-squared around 0.60, so the observed feature set would need to explain roughly 39 additional percentage points of log-price variance relative to today.

At signal_share = 95% with 8 synthetic features, random 5-fold mean MAPE reaches 14.8%. This is the best case produced by the mock and is still well above the 10% target, which confirms that the observed hedonic feature set alone cannot close the gap.

