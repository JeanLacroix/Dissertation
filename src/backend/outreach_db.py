from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from .paths import OUTREACH_DB_PATH, SEED_ASSETS_PATH, SEED_CONTACTS_PATH, SEED_EVENTS_PATH, ensure_outreach_dirs
from .scbsm_assets import build_scbsm_asset_dataset


def get_connection(db_path: Path = OUTREACH_DB_PATH) -> sqlite3.Connection:
    ensure_outreach_dirs()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    query = f"SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}')"
    return bool(connection.execute(query).fetchone()[0])


def _table_has_rows(connection: sqlite3.Connection, table_name: str) -> bool:
    if not _table_exists(connection, table_name):
        return False
    return bool(connection.execute(f"SELECT EXISTS(SELECT 1 FROM {table_name} LIMIT 1)").fetchone()[0])


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


def _load_seed_events() -> pd.DataFrame:
    if not SEED_EVENTS_PATH.exists():
        return pd.DataFrame(
            columns=[
                "event_id",
                "investor_id",
                "mandate_name",
                "deal_asset_type",
                "deal_zone",
                "deal_city",
                "deal_ticket_eur_mn",
                "deal_cap_rate_pct",
                "event_date",
                "channel",
                "outcome",
                "next_action_date",
                "owner",
                "notes",
            ]
        )
    return pd.read_csv(SEED_EVENTS_PATH)


def _empty_staged_mandates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "staged_mandate_id",
            "mandate_name",
            "asset_type",
            "country",
            "zone",
            "city",
            "ticket_eur_mn",
            "cap_rate_pct",
            "size_sqm",
            "transaction_date",
            "received_at",
            "source",
            "notes",
            "status",
        ]
    )


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

    with get_connection() as connection:
        for table_name, frame, mutable in [
            ("assets", assets, False),
            ("contacts", contacts, False),
            ("outreach_events", events, True),
            ("staged_mandates", staged_mandates, True),
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
        return pd.read_sql_query("SELECT * FROM outreach_events ORDER BY event_date DESC, event_id DESC", connection)


def load_staged_mandates() -> pd.DataFrame:
    initialize_outreach_db()
    with get_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM staged_mandates ORDER BY received_at DESC, staged_mandate_id DESC",
            connection,
        )
    for column in ["ticket_eur_mn", "cap_rate_pct", "size_sqm"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def append_outreach_event(
    *,
    investor_id: str,
    mandate_name: str,
    deal_asset_type: str,
    deal_zone: str,
    deal_city: str,
    deal_ticket_eur_mn: float,
    deal_cap_rate_pct: float,
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
        "investor_id": investor_id,
        "mandate_name": mandate_name,
        "deal_asset_type": deal_asset_type,
        "deal_zone": deal_zone,
        "deal_city": deal_city,
        "deal_ticket_eur_mn": deal_ticket_eur_mn,
        "deal_cap_rate_pct": deal_cap_rate_pct,
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


def stage_mandate(
    *,
    mandate_name: str,
    asset_type: str,
    country: str,
    zone: str,
    city: str,
    ticket_eur_mn: float,
    cap_rate_pct: float,
    size_sqm: float,
    transaction_date: str,
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
        "ticket_eur_mn": ticket_eur_mn,
        "cap_rate_pct": cap_rate_pct,
        "size_sqm": size_sqm,
        "transaction_date": transaction_date,
        "received_at": pd.Timestamp.utcnow().isoformat(),
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
