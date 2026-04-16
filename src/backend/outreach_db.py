from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from .paths import OUTREACH_DB_PATH, SEED_ASSETS_PATH, SEED_CONTACTS_PATH, SEED_EVENTS_PATH, ensure_outreach_dirs
from .scbsm_assets import build_scbsm_asset_dataset


OUTREACH_EVENT_COLUMNS = [
    "event_id",
    "investor_id",
    "mandate_name",
    "deal_asset_type",
    "deal_zone",
    "deal_city",
    "price_min_eur_mn",
    "price_max_eur_mn",
    "deal_ticket_eur_mn",
    "deal_cap_rate_pct",
    "event_date",
    "touchpoint_type",
    "status_value",
    "owner",
    "notes",
    "created_at_utc",
    "backdated_flag",
]

STAGED_MANDATE_COLUMNS = [
    "staged_mandate_id",
    "mandate_name",
    "asset_type",
    "country",
    "zone",
    "city",
    "price_min_eur_mn",
    "price_max_eur_mn",
    "ticket_eur_mn",
    "cap_rate_pct",
    "size_sqm",
    "transaction_date",
    "noi_eur_mn",
    "lease_terms",
    "building_grade",
    "received_at",
    "source",
    "notes",
    "status",
]

PROFILE_EDIT_COLUMNS = [
    "edit_id",
    "investor_id",
    "edited_at_utc",
    "edited_by",
    "changed_fields_json",
    "note",
]


def get_connection(db_path: Path = OUTREACH_DB_PATH) -> sqlite3.Connection:
    ensure_outreach_dirs()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    query = f"SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}')"
    return bool(connection.execute(query).fetchone()[0])


def _table_row_count(connection: sqlite3.Connection, table_name: str) -> int:
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def _table_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.OperationalError:
        return []
    return [row[1] for row in rows]


def _seed_table_matches(connection: sqlite3.Connection, table_name: str, frame: pd.DataFrame) -> bool:
    return _table_columns(connection, table_name) == frame.columns.tolist()


def _load_seed_assets() -> pd.DataFrame:
    if not SEED_ASSETS_PATH.exists():
        build_scbsm_asset_dataset()
    return pd.read_csv(SEED_ASSETS_PATH)


def _load_seed_contacts() -> pd.DataFrame:
    if not SEED_CONTACTS_PATH.exists():
        raise FileNotFoundError(f"Missing seed contacts file: {SEED_CONTACTS_PATH}")
    frame = pd.read_csv(SEED_CONTACTS_PATH)
    if "marchand_de_bien" in frame.columns:
        frame["marchand_de_bien"] = (
            frame["marchand_de_bien"]
            .map(lambda value: str(value).strip().lower() in {"1", "true", "yes"} if pd.notna(value) else False)
            .fillna(False)
        )
    return frame


def _empty_outreach_events() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTREACH_EVENT_COLUMNS)


def _load_seed_events() -> pd.DataFrame:
    if not SEED_EVENTS_PATH.exists():
        return _empty_outreach_events()
    frame = pd.read_csv(SEED_EVENTS_PATH)
    missing = [column for column in OUTREACH_EVENT_COLUMNS if column not in frame.columns]
    if missing:
        return _empty_outreach_events()
    return frame.loc[:, OUTREACH_EVENT_COLUMNS].copy()


def _empty_staged_mandates() -> pd.DataFrame:
    return pd.DataFrame(columns=STAGED_MANDATE_COLUMNS)


def _empty_profile_edits() -> pd.DataFrame:
    return pd.DataFrame(columns=PROFILE_EDIT_COLUMNS)


def _should_replace_seed_table(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
    *,
    force_reseed: bool,
    mutable: bool,
) -> bool:
    if force_reseed:
        return True
    if not _table_exists(connection, table_name):
        return True
    if not _seed_table_matches(connection, table_name, frame):
        return True
    if not mutable and _table_row_count(connection, table_name) != len(frame):
        return True
    return False


def initialize_outreach_db(force_reseed: bool = False) -> Path:
    ensure_outreach_dirs()
    assets = _load_seed_assets()
    contacts = _load_seed_contacts()
    events = _load_seed_events()
    staged_mandates = _empty_staged_mandates()
    profile_edits = _empty_profile_edits()

    with get_connection() as connection:
        for table_name, frame, mutable in [
            ("assets", assets, False),
            ("contacts", contacts, False),
            ("outreach_events", events, True),
            ("staged_mandates", staged_mandates, True),
            ("profile_edits", profile_edits, True),
        ]:
            if _should_replace_seed_table(
                connection,
                table_name,
                frame,
                force_reseed=force_reseed,
                mutable=mutable,
            ):
                frame.to_sql(table_name, connection, if_exists="replace", index=False)

    return OUTREACH_DB_PATH


def load_assets() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM assets ORDER BY fair_value_eur DESC, asset_number ASC", connection)


def load_contacts() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        return pd.read_sql_query("SELECT * FROM contacts ORDER BY company ASC, full_name ASC", connection)


def load_outreach_events() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM outreach_events ORDER BY event_date DESC, created_at_utc DESC, event_id DESC",
            connection,
        )
    if "backdated_flag" in frame.columns:
        frame["backdated_flag"] = (
            frame["backdated_flag"]
            .map(lambda value: str(value).strip().lower() in {"1", "true", "yes"} if pd.notna(value) else False)
            .fillna(False)
        )
    return frame


def load_staged_mandates() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM staged_mandates ORDER BY received_at DESC, staged_mandate_id DESC",
            connection,
        )
    for column in ["price_min_eur_mn", "price_max_eur_mn", "ticket_eur_mn", "cap_rate_pct", "size_sqm", "noi_eur_mn"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_profile_edits(investor_id: str = "scbsm") -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM profile_edits WHERE investor_id = ? ORDER BY edited_at_utc DESC, edit_id DESC",
            connection,
            params=[investor_id],
        )
    return frame


def append_outreach_event(
    *,
    investor_id: str,
    mandate_name: str,
    deal_asset_type: str,
    deal_zone: str,
    deal_city: str,
    price_min_eur_mn: float | None,
    price_max_eur_mn: float | None,
    deal_ticket_eur_mn: float,
    deal_cap_rate_pct: float | None,
    event_date: str,
    touchpoint_type: str,
    status_value: str,
    owner: str,
    notes: str,
    created_at_utc: str,
    backdated_flag: bool,
) -> str:
    initialize_outreach_db()
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    payload = {
        "event_id": event_id,
        "investor_id": investor_id,
        "mandate_name": mandate_name,
        "deal_asset_type": deal_asset_type,
        "deal_zone": deal_zone,
        "deal_city": deal_city,
        "price_min_eur_mn": price_min_eur_mn,
        "price_max_eur_mn": price_max_eur_mn,
        "deal_ticket_eur_mn": deal_ticket_eur_mn,
        "deal_cap_rate_pct": deal_cap_rate_pct,
        "event_date": event_date,
        "touchpoint_type": touchpoint_type,
        "status_value": status_value,
        "owner": owner,
        "notes": notes,
        "created_at_utc": created_at_utc,
        "backdated_flag": bool(backdated_flag),
    }
    with get_connection() as connection:
        pd.DataFrame([payload]).to_sql("outreach_events", connection, if_exists="append", index=False)
    return event_id


def append_profile_edit(
    *,
    investor_id: str,
    edited_at_utc: str,
    edited_by: str,
    changed_fields: list[str],
    note: str,
) -> str:
    initialize_outreach_db()
    edit_id = f"ped_{uuid.uuid4().hex[:10]}"
    payload = {
        "edit_id": edit_id,
        "investor_id": investor_id,
        "edited_at_utc": edited_at_utc,
        "edited_by": edited_by,
        "changed_fields_json": json.dumps(changed_fields),
        "note": note,
    }
    with get_connection() as connection:
        pd.DataFrame([payload]).to_sql("profile_edits", connection, if_exists="append", index=False)
    return edit_id


def stage_mandate(
    *,
    mandate_name: str,
    asset_type: str,
    country: str,
    zone: str,
    city: str,
    price_min_eur_mn: float | None,
    price_max_eur_mn: float | None,
    ticket_eur_mn: float,
    cap_rate_pct: float | None,
    size_sqm: float,
    transaction_date: str,
    noi_eur_mn: float | None = None,
    lease_terms: str | None = None,
    building_grade: str | None = None,
    source: str | None = None,
    notes: str | None = None,
) -> str:
    initialize_outreach_db()
    staged_mandate_id = f"mdt_{uuid.uuid4().hex[:10]}"
    payload = {
        "staged_mandate_id": staged_mandate_id,
        "mandate_name": mandate_name,
        "asset_type": asset_type,
        "country": country,
        "zone": zone,
        "city": city,
        "price_min_eur_mn": price_min_eur_mn,
        "price_max_eur_mn": price_max_eur_mn,
        "ticket_eur_mn": ticket_eur_mn,
        "cap_rate_pct": cap_rate_pct,
        "size_sqm": size_sqm,
        "transaction_date": transaction_date,
        "noi_eur_mn": noi_eur_mn,
        "lease_terms": lease_terms or "",
        "building_grade": building_grade or "",
        "received_at": pd.Timestamp.now("UTC").isoformat(),
        "source": source or "",
        "notes": notes or "",
        "status": "staged",
    }
    with get_connection() as connection:
        pd.DataFrame([payload]).to_sql("staged_mandates", connection, if_exists="append", index=False)
    return staged_mandate_id


def mark_staged_mandate_loaded(staged_mandate_id: str) -> None:
    initialize_outreach_db()
    with get_connection() as connection:
        connection.execute(
            "UPDATE staged_mandates SET status = ? WHERE staged_mandate_id = ?",
            ("loaded", staged_mandate_id),
        )
        connection.commit()


def export_ranked_contacts(frame: pd.DataFrame, output_path: Path) -> Path:
    ensure_outreach_dirs()
    frame.to_csv(output_path, index=False)
    return output_path


def refresh_seed_assets() -> Path:
    build_scbsm_asset_dataset()
    initialize_outreach_db(force_reseed=True)
    return SEED_ASSETS_PATH
