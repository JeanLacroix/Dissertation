from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTREACH_DIR = DATA_DIR / "outreach"
RAW_SCBSM_DIR = RAW_DATA_DIR / "scbsm"
RAW_SCBSM_TABLES_DIR = RAW_SCBSM_DIR / "tables" / "scbsm-2024-06-30-fr"

SEED_ASSETS_PATH = OUTREACH_DIR / "seed_assets.csv"
SEED_CONTACTS_PATH = OUTREACH_DIR / "seed_contacts.csv"
SEED_EVENTS_PATH = OUTREACH_DIR / "seed_outreach_events.csv"
SCBSM_PROFILE_PATH = OUTREACH_DIR / "scbsm_profile.json"
OUTREACH_DB_PATH = OUTREACH_DIR / "outreach.db"
OUTREACH_README_PATH = OUTREACH_DIR / "README.md"
YIELD_EXTRACTION_NOTE_PATH = OUTREACH_DIR / "SCBSM_YIELD_EXTRACTION.md"
EXPORTS_DIR = OUTREACH_DIR / "exports"


def ensure_outreach_dirs() -> None:
    OUTREACH_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
