# Dissertation Toolkit

This repository now contains two connected but distinct workstreams:

1. the original dissertation artefact for French commercial real-estate comparable retrieval and model diagnostics
2. a single-investor SCBSM mandate-fit prototype that uses the disclosed SCBSM portfolio, a minimal interaction log, a staged HTTP mandate queue, and a Streamlit interface

The point of the second workstream is practical: Alantra should not start each mandate from zero. This version narrows the workflow to one investor only, SCBSM, and lets the analyst test how a mandate fits that disclosed portfolio while still keeping a comparable-retrieval module alongside it.

## What has been built

### 1. Original comparable-retrieval stack

The original dissertation pipeline is still in place:

- `model/pipeline.py`
  Loads and filters the Preqin sample, builds country groups, and prepares the modelling frame.
- `model/train.py`
  Fits and evaluates the hedonic specifications and exports the deployed comparable-retrieval artefacts.
- `app.py`
  The original Streamlit comparable-retrieval interface.
- `model/scenario_analysis.py`, `model/rf_test.py`, `model/residual_diagnostics.py`
  Diagnostic scripts used to justify why the deployed artefact became a retrieval tool rather than a point-valuation AVM.

### 2. Public market-context extraction

A new scraper and normalisation layer were added:

- `analysis/scrape_scbsm.py`

What it does:

1. scrapes the SCBSM report pages by year
2. selects the annual URD-style filings
3. downloads the public Actusnews documents locally into `data/raw/scbsm/`
4. extracts candidate tables from the XHTML/HTML filings

What was confirmed:

- the SCBSM source is scrapable
- the URD contains exploitable asset, valuation, and expert-assumption information
- the 2024 filing contains proper HTML tables that can be extracted cleanly

What remains hard:

- the 2022, 2023, and 2025 filings use a layout-oriented XHTML structure with positioned text blocks rather than standard `<table>` markup
- those years will need a second parser if you want a fully reconstructed multi-year dataset from the same source

## Yield extraction: what was added and how

The main public-data product is:

- `data/outreach/seed_assets.csv`

It is produced by:

- `src/backend/scbsm_assets.py`

### Source tables used

The normalisation step uses three SCBSM tables already scraped from the 2024 URD:

1. `table_134.csv`
   Asset list, city, zone, valuation date, and last expert visit.
2. `table_137.csv`
   Asset-level fair value (`Juste Valeur`).
3. `table_135.csv`
   Zone-level valuation assumptions, including the capitalisation-rate range.

### Join logic

The logic is intentionally simple and transparent:

1. join `table_134` and `table_137` on `asset_number`
2. join the zone-level assumptions from `table_135` back onto each asset on `zone`
3. materialise:
   - `yield_min_pct`
   - `yield_max_pct`
   - `yield_mid_pct`

### Important caveat

The yield is **not** an asset-specific expert yield for each building.

It is the **zone-level capitalisation band** disclosed in the SCBSM expertise assumptions table:

- Paris assets receive the Paris band
- IDF assets receive the IDF band
- Province assets receive the Province band

This is still useful because it gives the SCBSM-fit prototype a benchmark for portfolio cap-rate alignment. It should not be presented as a bespoke appraised yield for each individual asset.

The detailed note is written to:

- `data/outreach/SCBSM_YIELD_EXTRACTION.md`

## New outreach application

The outreach prototype is split into `src/backend` and `src/frontend` as requested.

### Backend

- `src/backend/paths.py`
  Centralises the project paths and outreach data locations.
- `src/backend/scbsm_assets.py`
  Builds the clean SCBSM asset dataset with value and yield fields.
- `src/backend/outreach_db.py`
  Creates and seeds the local SQLite database, including the staged mandate queue.
- `src/backend/outreach_scoring.py`
  Holds the mandate-scoring logic, including the SCBSM-derived profile.
- `src/backend/outreach_service.py`
  Orchestrates SCBSM profile derivation, mandate scoring, fiche generation, staged-mandate loading, and follow-up logging.
- `src/backend/recommend_outreach.py`
  CLI script that scores a mandate against SCBSM and exports the result.
- `src/backend/api.py`
  FastAPI sidecar for `GET /health`, `POST /mandates/staging`, and `GET /mandates/staging`.

### Frontend

- `src/frontend/app.py`
  Streamlit UI for the outreach workflow.

What the app does:

1. lets the analyst input the characteristics of a new mandate
2. scores that mandate against SCBSM only
3. shows one SCBSM fit card with reasons, watch-outs, and portfolio context
4. exposes a visible inbound queue of mandates received over HTTP
5. lets the analyst load a staged mandate into the working screen
6. logs SCBSM follow-ups into the local SQLite database
7. keeps the comparable-retrieval module available as a separate valuation-context tab

### Ranking logic

The scoring model is deliberately transparent. It combines:

- geography fit
- asset-class fit
- ticket-size fit
- zone-level yield fit
- current portfolio cap-rate alignment
- recent interaction outcome
- strategic-priority bonus
- cooldown penalty if the contact was touched too recently

The aim is not to mimic a production CRM. The aim is to create a usable, explainable single-investor prototype before broadening the workflow back out to a larger registry.

## Local data layer

The outreach prototype uses:

- `data/outreach/seed_assets.csv`
  Public market context for the prototype, used to enrich cap-rate benchmarks.
- `data/outreach/seed_contacts.csv`
  Legacy support row retained for backward compatibility only. It is no longer the canonical scoring source.
- `data/outreach/scbsm_profile.json`
  Small qualitative metadata source for the SCBSM prototype.
- `data/outreach/seed_outreach_events.csv`
  Minimal SCBSM follow-up history.
- `data/outreach/outreach.db`
  Local SQLite database created on first run. It now stores the staged HTTP mandate queue as well. This file is gitignored.

The current contact seed is retained only as a compatibility artifact. Canonical fit is derived from the SCBSM portfolio plus `scbsm_profile.json`.

## How to run

### Quick start from the project root

Run the commands below from:

```powershell
cd c:\Users\jeanl\Documents\UCL\dissertation\Solution\Dissertation
```

You can now activate the local virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If your PowerShell execution policy blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
```

Install or refresh dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you want to refresh the public market-context layer, rebuild the SCBSM asset seed first, then launch the outreach app:

```powershell
.\.venv\Scripts\python.exe -m src.backend.scbsm_assets
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

If you want the one-line launcher instead:

```powershell
.\run_outreach.ps1
```

Once Streamlit starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### What happens on first launch

- `seed_assets.csv` is used to seed the public market-context table
- `seed_outreach_events.csv` is used to seed the SCBSM follow-up history
- staged mandates posted to the API are appended to the SQLite queue
- `data/outreach/outreach.db` is created automatically if it does not exist

### Build the SCBSM asset dataset

```powershell
.\.venv\Scripts\python -m src.backend.scbsm_assets
```

### Score SCBSM from the command line

```powershell
.\.venv\Scripts\python -m src.backend.recommend_outreach
```

### Launch the HTTP mandate-intake API

```powershell
.\.venv\Scripts\python -m src.backend.api
```

### Launch the outreach Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

Or use:

```powershell
.\run_outreach.ps1
```

### Launch the original dissertation retrieval app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## What was validated locally

The following parts were actually run in the current environment:

- `python -m src.backend.scbsm_assets`
  Wrote `data/outreach/seed_assets.csv` and `data/outreach/SCBSM_YIELD_EXTRACTION.md`.
- `python -m src.backend.recommend_outreach --top-k 5`
  Scored the current mandate against SCBSM and exported a CSV into `data/outreach/exports/`.
- `fastapi.testclient` against `src.backend.api`
  Verified `GET /health`, `POST /mandates/staging`, and `GET /mandates/staging`.
- `from src.frontend.app import main`
  Confirmed that the new Streamlit frontend imports cleanly against the backend modules.

## Repository layout

```text
.
|-- app.py                         # original comparable-retrieval UI
|-- analysis/
|   `-- scrape_scbsm.py           # SCBSM scraping proof of concept
|-- data/
|   |-- raw/                      # local raw inputs, gitignored
|   `-- outreach/                 # outreach seeds, SQLite db, and notes
|-- model/
|   |-- pipeline.py
|   |-- train.py
|   |-- scenario_analysis.py
|   |-- rf_test.py
|   `-- residual_diagnostics.py
`-- src/
    |-- backend/                  # outreach backend
    `-- frontend/                 # outreach Streamlit app
```

## What to keep in mind

- The original dissertation deployment remains a retrieval tool, not a valuation engine.
- The SCBSM yield added for outreach is zone-level, not asset-specific, and is used only as context.
- The prototype currently evaluates one investor only: SCBSM.
- `seed_contacts.csv` is no longer the canonical scoring source.
- The SQLite database is local and intentionally lightweight.
- Multi-year SCBSM reconstruction is possible, but older years need a second parsing strategy because they are not exposed as normal HTML tables.
