# Outreach App Data Layer

This folder contains the local data and documentation for the SCBSM mandate-fit prototype.

## Purpose

The outreach workflow in this version starts from one investor only: SCBSM. Public SCBSM data is both the investor context and the market context, while new mandates can arrive through the staged HTTP intake queue. The objective is not to predict value. The objective is to decide:

- which mandate is in scope
- how well that mandate fits SCBSM
- what the ticket, yield, and portfolio-cap-rate context are
- what the last follow-up was
- what the next move should be

## Files in this folder

### Seed data

- `seed_assets.csv`
  Public market-context rows built from SCBSM's 2024 URD.
- `seed_contacts.csv`
  Legacy support row retained for backward compatibility only.
- `scbsm_profile.json`
  Small qualitative metadata file used to complete the SCBSM profile where the portfolio data is silent.
- `seed_outreach_events.csv`
  Minimal SCBSM follow-up history used to initialise the SQLite store.

### Generated / local files

- `outreach.db`
  Local SQLite database created on first run. Gitignored. This now stores the staged HTTP mandate queue as well.
- `exports/`
  Ranked-investor CSV exports written by the CLI recommendation script.

### Documentation

- `SCBSM_YIELD_EXTRACTION.md`
  Detailed note explaining how the yield fields were derived.

## What was added to the SCBSM dataset

The key enrichment is the yield block:

- `yield_min_pct`
- `yield_max_pct`
- `yield_mid_pct`
- `cap_rate_range_pct`

These fields come from the SCBSM expertise assumptions table. They are joined back to each public row by `zone`.

### Important interpretation

This means the yield is:

- **disclosed**
- **public**
- **usable for outreach filtering**

but it is **not** an asset-specific valuation yield for each individual building.

The precision flag in the dataset is therefore:

- `yield_precision = zone_range`

That is deliberate. The app uses the yield to compare the mandate against the investor's target-return range and current portfolio cap rate. It does not pretend to have a bespoke per-asset expert cap rate.

## Backend components using this folder

- `src/backend/scbsm_assets.py`
  Rebuilds `seed_assets.csv`.
- `src/backend/outreach_db.py`
  Seeds and manages the SQLite database.
- `src/backend/outreach_scoring.py`
  Computes the SCBSM fit score.
- `src/backend/outreach_service.py`
  Serves the frontend, loads staged mandates, and writes follow-up logs.
- `src/backend/recommend_outreach.py`
  Exports a one-row SCBSM fit result from the command line.
- `src/backend/api.py`
  Exposes `GET /health`, `POST /mandates/staging`, and `GET /mandates/staging`.

## Frontend component

- `src/frontend/app.py`

The Streamlit frontend reads from the SQLite database and exposes four things:

1. the SCBSM fit card for the current mandate
2. the staged HTTP mandate queue
3. the follow-up form that writes a new SCBSM event into the database
4. a snapshot of the SCBSM portfolio context

## Scoring logic in plain English

For each input mandate, SCBSM receives a score made from:

- `zone_match_score`
  Does SCBSM already hold disclosed assets in the same zone or city?
- `asset_focus_score`
  Does SCBSM already hold disclosed assets in the same asset class?
- `ticket_fit_score`
  Is the deal size inside the observed SCBSM portfolio range?
- `yield_fit_score`
  Is the mandate cap-rate estimate close to the relevant SCBSM public yield context?
- `portfolio_cap_rate_score`
  Is the overall SCBSM portfolio cap-rate profile aligned with the mandate?
- relationship and history terms
  Recent SCBSM interactions affect the next-action recommendation.
- cooldown
  Very recent SCBSM touches are penalised to avoid over-contacting.

## Commands

### Fastest way to run the outreach app

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.backend.scbsm_assets
python -m streamlit run src/frontend/app.py
```

If `Activate.ps1` is blocked by PowerShell execution policy, use the executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.backend.scbsm_assets
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

Or use:

```powershell
.\run_outreach.ps1
```

The app will normally be available at:

```text
http://localhost:8501
```

### Rebuild the SCBSM asset seed

```powershell
.\.venv\Scripts\python.exe -m src.backend.scbsm_assets
```

### Rebuild or seed the local database implicitly

This happens automatically when the app or recommendation script runs.

### Export a ranked contact list

```powershell
.\.venv\Scripts\python.exe -m src.backend.recommend_outreach
```

### Launch the Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

### Launch the HTTP mandate-intake API

```powershell
.\.venv\Scripts\python.exe -m src.backend.api
```

## Current limitations

- The prototype currently evaluates one investor only: SCBSM.
- `seed_contacts.csv` is retained only for backward compatibility and is not the canonical scoring source.
- The yield is zone-level, not asset-specific, and is used only as market context.
- The asset universe currently depends on the 2024 URD tables that parsed cleanly.
- A richer multi-year SCBSM dataset will require a second parser for the older positioned-XHTML filings.
