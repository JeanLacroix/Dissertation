from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import OUTREACH_DB_PATH, SEED_ASSETS_PATH, SEED_CONTACTS_PATH, SEED_EVENTS_PATH, ensure_outreach_dirs
from .scbsm_assets import build_scbsm_asset_dataset


ASSET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    asset_number INTEGER,
    asset_name TEXT NOT NULL,
    asset_class TEXT,
    city TEXT,
    zone TEXT,
    investment_profile TEXT,
    valuation_date TEXT,
    last_visit_date TEXT,
    fair_value_eur REAL,
    fair_value_eur_mn REAL,
    vlm_range_eur_sqm_year TEXT,
    vlm_min_eur_sqm_year REAL,
    vlm_max_eur_sqm_year REAL,
    vacancy_months_range TEXT,
    vacancy_min_months REAL,
    vacancy_max_months REAL,
    cap_rate_range_pct TEXT,
    yield_min_pct REAL,
    yield_max_pct REAL,
    yield_mid_pct REAL,
    yield_precision TEXT,
    yield_source TEXT,
    generated_at_utc TEXT
);
"""

CONTACT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS contacts (
    contact_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    company TEXT NOT NULL,
    title TEXT,
    email TEXT,
    city TEXT,
    zone_focus TEXT,
    asset_focus TEXT,
    min_ticket_eur_mn REAL,
    max_ticket_eur_mn REAL,
    min_target_yield_pct REAL,
    max_target_yield_pct REAL,
    relationship_stage TEXT,
    last_contact_date TEXT,
    last_outcome TEXT,
    response_rate_score REAL,
    strategic_priority REAL,
    preferred_channel TEXT,
    owner TEXT,
    notes TEXT
);
"""

EVENT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS outreach_events (
    event_id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    asset_id TEXT,
    event_date TEXT NOT NULL,
    channel TEXT,
    outcome TEXT,
    next_action_date TEXT,
    owner TEXT,
    notes TEXT,
    FOREIGN KEY(contact_id) REFERENCES contacts(contact_id),
    FOREIGN KEY(asset_id) REFERENCES assets(asset_id)
);
"""


def get_connection(db_path: Path = OUTREACH_DB_PATH) -> sqlite3.Connection:
    ensure_outreach_dirs()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_has_rows(connection: sqlite3.Connection, table_name: str) -> bool:
    query = f"SELECT EXISTS(SELECT 1 FROM {table_name} LIMIT 1)"
    return bool(connection.execute(query).fetchone()[0])


def _load_seed_assets() -> pd.DataFrame:
    if not SEED_ASSETS_PATH.exists():
        build_scbsm_asset_dataset()
    return pd.read_csv(SEED_ASSETS_PATH)


def _load_seed_contacts() -> pd.DataFrame:
    if not SEED_CONTACTS_PATH.exists():
        raise FileNotFoundError(f"Missing seed contacts file: {SEED_CONTACTS_PATH}")
    return pd.read_csv(SEED_CONTACTS_PATH)


def _load_seed_events() -> pd.DataFrame:
    if not SEED_EVENTS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "event_id",
                "contact_id",
                "asset_id",
                "event_date",
                "channel",
                "outcome",
                "next_action_date",
                "owner",
                "notes",
            ]
        )
    return pd.read_csv(SEED_EVENTS_PATH)


def initialize_outreach_db(force_reseed: bool = False) -> Path:
    ensure_outreach_dirs()
    assets = _load_seed_assets()
    contacts = _load_seed_contacts()
    events = _load_seed_events()

    with get_connection() as connection:
        connection.execute(ASSET_TABLE_SQL)
        connection.execute(CONTACT_TABLE_SQL)
        connection.execute(EVENT_TABLE_SQL)

        if force_reseed or not _table_has_rows(connection, "assets"):
            assets.to_sql("assets", connection, if_exists="replace", index=False)
        if force_reseed or not _table_has_rows(connection, "contacts"):
            contacts.to_sql("contacts", connection, if_exists="replace", index=False)
        if force_reseed or not _table_has_rows(connection, "outreach_events"):
            events.to_sql("outreach_events", connection, if_exists="replace", index=False)

    return OUTREACH_DB_PATH


def load_assets() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM assets ORDER BY fair_value_eur DESC, asset_number ASC", connection)


def load_contacts() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM contacts ORDER BY full_name ASC", connection)


def load_outreach_events() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM outreach_events ORDER BY event_date DESC, event_id DESC", connection)


def append_outreach_event(
    *,
    contact_id: str,
    asset_id: str | None,
    event_date: str,
    channel: str,
    outcome: str,
    next_action_date: str | None,
    owner: str,
    notes: str,
) -> str:
    initialize_outreach_db()
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    payload = {
        "event_id": event_id,
        "contact_id": contact_id,
        "asset_id": asset_id,
        "event_date": event_date,
        "channel": channel,
        "outcome": outcome,
        "next_action_date": next_action_date,
        "owner": owner,
        "notes": notes,
    }
    with get_connection() as connection:
        pd.DataFrame([payload]).to_sql("outreach_events", connection, if_exists="append", index=False)
    return event_id


def export_ranked_contacts(frame: pd.DataFrame, output_path: Path) -> Path:
    ensure_outreach_dirs()
    frame.to_csv(output_path, index=False)
    return output_path


def refresh_seed_assets() -> Path:
    build_scbsm_asset_dataset()
    initialize_outreach_db(force_reseed=True)
    return SEED_ASSETS_PATH
