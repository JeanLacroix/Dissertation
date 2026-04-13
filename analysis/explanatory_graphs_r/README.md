# Explanatory Graphs in R

This folder is separate from the application code. It is an opinionated, descriptive analysis bundle built from the committed anonymised sample in `model/artifacts/comps_sample.parquet`.

The goal is not to chart everything. It focuses on the structural facts that matter most for interpretation:

- Country concentration is severe: the United Kingdom accounts for 57.9% of the sample, and `United Kingdom + Other Europe` reaches 74.4%.
- Asset coverage is uneven: Office, Industrial, and Retail make up 80.7% of rows, while Hotel, Residential, Land, and Niche are thin buckets.
- Time coverage is back-loaded poorly: 58.1% of rows sit in 2021-2022, while 2026 contains only 19 observations.
- The dataset is centered on mid-sized, mid-ticket deals: 46.6% of rows fall in the `2,500-25,000 sqm` and `EUR 10m-100m` corridor.
- Value density differs materially by use type: more than `EUR 5,000/sqm` appears in 52.1% of Office rows, but only 5.1% of Industrial rows.
- `Other Europe` is a pooled bucket, not a clean market: it contains 23 countries, with Poland and Ireland alone contributing 38 rows.

## Files

- `make_graphs.R`: base-R script that renders the charts
- `data/comps_sample.csv`: R-friendly export of the anonymised comparables sample
- `data/other_europe_bucket_composition.csv`: breakdown of the pooled `Other Europe` bucket
- `output/`: rendered PNGs are written here and ignored locally

## Run

From the repository root:

```powershell
.\run_graphs.ps1
```

If you want to call `Rscript` directly:

```powershell
& 'C:\Program Files (x86)\R\R-4.5.3\bin\Rscript.exe' analysis\explanatory_graphs_r\make_graphs.R
```

Or, after opening a new terminal with `Rscript` on PATH:

```powershell
Rscript analysis\explanatory_graphs_r\make_graphs.R
```

## Expected outputs

The script writes these files to `analysis/explanatory_graphs_r/output/`:

- `01_country_concentration.png`
- `02_asset_coverage.png`
- `03_year_coverage.png`
- `04_size_price_heatmap.png`
- `05_value_density_by_asset.png`
- `06_other_europe_bucket.png`
