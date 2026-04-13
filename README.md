# Hedonic AVM Webapp

Streamlit web application for Jean Lacroix's UCL MSIN0032 dissertation. The app packages a hedonic automated valuation model for French commercial real estate comparable-sales selection using European transaction data with country fixed effects.

## Repository Layout

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- indices/
`-- model/
    |-- __init__.py
    |-- pipeline.py
    |-- train.py
    `-- artifacts/
```

## Status

Stage one is implemented in `model/pipeline.py`.
Stage two is implemented in `model/train.py`.
Stage three (`app.py`) is still scaffolded only and intentionally left for the next validation checkpoint.

## Data Handling

- `data/raw/` contains licensed Preqin and Capital IQ source files and is gitignored.
- `data/indices/` contains local macro-rate source files and is gitignored.
- `model/artifacts/` is reserved for anonymised committed outputs only.

## Current Macro Proxy Choice

The original country-specific commercial property index plan was replaced with a standardised macro-rate proxy:

- ECB HICP housing, water, electricity, gas and other fuels annual-rate series for euro-area countries
- UK CPI/CPIH annual-rate series for the United Kingdom

Because these inputs are rates rather than level indices, stage one constructs a rebased quarterly proxy index anchored at `2021Q1 = 100`. This is a practical approximation and should be flagged in the dissertation writeup.

## Training Artifacts

Running `python -m model.train` writes the stage-two outputs to `model/artifacts/`:

- `model.pkl`: fitted statsmodels OLS result
- `residuals.npy`: residual bootstrap pool for prediction intervals
- `comps_sample.parquet`: anonymised comparables sample for the app
- `metadata.json`: model specification, validation metrics, macro-source notes, and limitations

## Limitations

- United Kingdom transactions constitute 58 percent of the training sample, so the model is UK-dominated and country fixed effects mainly shift levels for smaller markets.
- Size elasticity is assumed homogeneous across countries and asset types; no interaction specification was tested in this version.
- Mixed Use and Niche are heterogeneous buckets in the Preqin taxonomy and are likely to contribute higher residual variance.
- The Other Europe dummy pools roughly 20 countries with small sample sizes.
- Commercial property level indices were not used in this version; a chained quarterly proxy was built from HICP/CPI inflation-rate series rebased to `2021Q1 = 100`.
- Year built was excluded from the current model because the Capital IQ enrichment produced fewer than 150 confident matches.
