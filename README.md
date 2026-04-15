# Dissertation Toolkit

This repository now contains two connected but distinct workstreams:

1. the original dissertation artefact for French commercial real-estate comparable retrieval and model diagnostics
2. a new SCBSM-based outreach-selection prototype that uses public asset data plus a local contact database and a Streamlit interface

The point of the second workstream is practical: once the SCBSM asset universe has been normalised into a usable dataset, the repo can support an outreach workflow that helps decide **who to contact for which asset, in what order, and with what pitch angle**.

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

### 2. SCBSM scraping proof of concept

A new scraper was added:

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

The main new data product is:

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

This is still useful for outreach selection because it gives the ranking engine a benchmark to compare against each contact's target yield range. It should not be presented as a bespoke appraised yield for each individual asset.

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
  Creates and seeds the local SQLite database.
- `src/backend/outreach_scoring.py`
  Holds the ranking logic for contact selection.
- `src/backend/outreach_service.py`
  Orchestrates asset loading, ranking, fiche generation, and follow-up logging.
- `src/backend/recommend_outreach.py`
  CLI script that prints and exports the ranked outreach list for a selected asset.

### Frontend

- `src/frontend/app.py`
  Streamlit UI for the outreach workflow.

What the app does:

1. selects an asset from the SCBSM universe
2. ranks contacts against that asset
3. shows a full "fiche outreach" for the selected contact
4. logs follow-ups into a local SQLite database
5. lets you inspect both the asset dataset and the contact database from the UI

### Ranking logic

The scoring model is deliberately transparent. It combines:

- geography fit
- asset-class fit
- ticket-size fit
- target-yield fit
- relationship-stage bonus
- historical response bonus
- strategic-priority bonus
- last-outcome bonus or penalty
- cooldown penalty if the contact was touched too recently

The aim is not to mimic a production CRM. The aim is to create a usable, explainable selection layer that can later be swapped onto a real contact universe.

## Local data layer

The outreach prototype uses:

- `data/outreach/seed_assets.csv`
  SCBSM asset universe for the prototype.
- `data/outreach/seed_contacts.csv`
  Demo contact universe.
- `data/outreach/seed_outreach_events.csv`
  Demo follow-up history.
- `data/outreach/outreach.db`
  Local SQLite database created on first run. This file is gitignored.

The current contact dataset is a **demo seed**, not a live CRM export. Replace it with your real contact universe when the structure is validated.

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

Then rebuild the SCBSM asset seed and launch the outreach app:

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

- `seed_assets.csv` is used to seed the asset universe
- `seed_contacts.csv` is used to seed the contact database
- `seed_outreach_events.csv` is used to seed the follow-up history
- `data/outreach/outreach.db` is created automatically if it does not exist

### Build the SCBSM asset dataset

```powershell
.\.venv\Scripts\python -m src.backend.scbsm_assets
```

### Rank contacts from the command line

```powershell
.\.venv\Scripts\python -m src.backend.recommend_outreach --top-k 5
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
  Ranked contacts successfully and exported a CSV into `data/outreach/exports/`.
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
- The SCBSM yield added for outreach is zone-level, not asset-specific.
- The outreach contact database is seeded with demo contacts for the prototype.
- The SQLite database is local and intentionally lightweight.
- Multi-year SCBSM reconstruction is possible, but older years need a second parsing strategy because they are not exposed as normal HTML tables.
