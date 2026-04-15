# Mock completeness benchmark

This analysis creates a synthetic richer-data benchmark to estimate what predictive precision could be achievable if materially more deal-level information were observed. The starting point is the existing Change D sample and fitted structure. A mock target is generated as the sum of: (i) the fitted Change D signal already explained by observed covariates, (ii) four synthetic standardised features representing micro-location quality, building quality, lease quality, and tenant covenant strength, and (iii) irreducible noise.

The synthetic features are not copied from external data. They are generated with group structure by country, asset type, and year plus idiosyncratic noise, then scaled so that they explain 50% of the current Change D residual variance in the base case. The remaining 50% is left as noise.

On the real target, Change D records a rolling-origin mean MAPE of 87.7%. On the base-case mock target, the observed-feature model records 90.9%, while the extensive-data model with all four synthetic features records 57.6%. The script also runs a sensitivity sweep over 30%, 50%, and 70% explainable residual-variance shares. This should be interpreted as a structured sensitivity benchmark rather than as a claim about true achievable production accuracy.
