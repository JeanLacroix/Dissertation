# Outreach App Data Layer

This folder contains the local data and documentation for the outreach-selection prototype.

## Purpose

The outreach workflow starts from a public asset universe extracted from SCBSM and combines it with a lightweight contact database. The objective is not to predict value. The objective is to decide:

- which asset is in scope
- which contact looks most relevant for that asset
- what the yield and ticket context are
- what the last follow-up was
- what the next move should be

## Files in this folder

### Seed data

- `seed_assets.csv`
  Clean asset universe built from SCBSM's 2024 URD.
- `seed_contacts.csv`
  Demo contact database used to initialise the SQLite store.
- `seed_outreach_events.csv`
  Demo follow-up history used to initialise the SQLite store.

### Generated / local files

- `outreach.db`
  Local SQLite database created on first run. Gitignored.
- `exports/`
  Ranked-contact CSV exports written by the CLI recommendation script.

### Documentation

- `SCBSM_YIELD_EXTRACTION.md`
  Detailed note explaining how the yield fields were derived.

## What was added to the SCBSM dataset

The key enrichment is the yield block:

- `yield_min_pct`
- `yield_max_pct`
- `yield_mid_pct`
- `cap_rate_range_pct`

These fields come from the SCBSM expertise assumptions table. They are joined back to each asset by `zone`.

### Important interpretation

This means the yield is:

- **disclosed**
- **public**
- **usable for outreach filtering**

but it is **not** an asset-specific valuation yield for each individual building.

The precision flag in the dataset is therefore:

- `yield_precision = zone_range`

That is deliberate. The app uses the yield to compare the asset against the contact's target-return range. It does not pretend to have a bespoke per-asset expert cap rate.

## Backend components using this folder

- `src/backend/scbsm_assets.py`
  Rebuilds `seed_assets.csv`.
- `src/backend/outreach_db.py`
  Seeds and manages the SQLite database.
- `src/backend/outreach_scoring.py`
  Computes the outreach ranking.
- `src/backend/outreach_service.py`
  Serves the frontend and writes follow-up logs.
- `src/backend/recommend_outreach.py`
  Exports a ranked contact list from the command line.

## Frontend component

- `src/frontend/app.py`

The Streamlit frontend reads from the SQLite database and exposes three things:

1. the ranked outreach list
2. the full fiche for a selected contact
3. the follow-up form that writes a new event into the database

## Scoring logic in plain English

For each selected asset, every contact receives a score made from:

- `zone_match_score`
  Does the contact cover Paris, IDF, Province, or Nationwide?
- `asset_focus_score`
  Does the contact typically buy Office, Retail, or Mixed Commercial?
- `ticket_fit_score`
  Is the asset size inside the contact's target ticket range?
- `yield_fit_score`
  Is the asset's disclosed yield midpoint close to the contact's target-return band?
- relationship and history terms
  Warm contacts, responsive contacts, and high-priority contacts get bonuses.
- cooldown
  Very recent touches are penalised to avoid spamming the same person.

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
.\.venv\Scripts\python.exe -m src.backend.recommend_outreach --top-k 5
```

### Launch the Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```

## Current limitations

- The contact database is seeded with demo contacts, not a live CRM export.
- The yield is zone-level, not asset-specific.
- The asset universe currently depends on the 2024 URD tables that parsed cleanly.
- A richer multi-year SCBSM dataset will require a second parser for the older positioned-XHTML filings.
