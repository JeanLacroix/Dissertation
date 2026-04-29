"""Build a clean SCBSM asset dataset from extracted public tables."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import (
    RAW_SCBSM_TABLES_DIR,
    SEED_ASSETS_PATH,
    YIELD_EXTRACTION_NOTE_PATH,
    ensure_outreach_dirs,
)


ZONE_STRATEGY = {
    "Paris": "Core Paris",
    "IDF": "Ile-de-France value-add",
    "Province": "Regional high-yield",
}


def _repair_text(value: Any) -> str:
    """Repair text."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if any(token in text for token in ("Ã", "Â", "â", "€™", "€")):
        try:
            text = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return " ".join(text.split())


def _parse_euro_number(value: Any) -> float | None:
    """Parse euro number."""
    text = _repair_text(value)
    if not text:
        return None

    digits = (
        text.replace("€", "")
        .replace("K€", "")
        .replace("HD", "")
        .replace("HFA", "")
        .replace("HT", "")
        .replace("HC", "")
        .replace(".", "")
        .replace(",", ".")
        .replace(" ", "")
        .strip()
    )
    if not digits:
        return None
    return float(digits)


def _parse_percent(value: Any) -> float | None:
    """Parse percent."""
    text = _repair_text(value).replace("%", "").replace(" ", "").replace(",", ".")
    if not text:
        return None
    return float(text)


def _parse_range(value: Any) -> tuple[float | None, float | None]:
    """Parse range."""
    text = _repair_text(value)
    if not text:
        return None, None

    parts = [part.strip() for part in text.split("à")]
    if len(parts) == 1:
        number = _parse_percent(parts[0]) if "%" in parts[0] else _parse_euro_number(parts[0])
        return number, number

    left = _parse_percent(parts[0]) if "%" in parts[0] else _parse_euro_number(parts[0])
    right = _parse_percent(parts[1]) if "%" in parts[1] else _parse_euro_number(parts[1])
    return left, right


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse date."""
    text = _repair_text(value)
    if not text:
        return pd.NaT
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _load_table(table_name: str) -> pd.DataFrame:
    """Load table."""
    table_path = RAW_SCBSM_TABLES_DIR / table_name
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing required SCBSM table: {table_path}. "
            "Run `python analysis\\scrape_scbsm.py` first."
        )
    return pd.read_csv(table_path)


def _infer_asset_class(asset_name: str, zone: str) -> str:
    """Infer asset class."""
    token = asset_name.lower()
    if "retail" in token or "franciades" in token:
        return "Retail"
    if "bureaux" in token:
        return "Office"
    if zone == "Paris":
        return "Office"
    if "mall" in token:
        return "Retail"
    if any(label in token for label in ("buchelay", "soyaux", "sillé", "epernay", "graham bell")):
        return "Mixed Commercial"
    return "Mixed Commercial"


def _load_asset_metadata() -> pd.DataFrame:
    """Load asset metadata."""
    raw = _load_table("table_134.csv")
    frame = raw.rename(
        columns={
            "col_1": "asset_number",
            "col_2": "asset_name",
            "col_3": "city",
            "col_4": "zone",
            "col_5": "valuation_date",
            "col_6": "last_visit_date",
        }
    ).copy()
    frame = frame.iloc[1:].copy()
    frame["asset_number"] = frame["asset_number"].astype(int)
    for column in ["asset_name", "city", "zone"]:
        frame[column] = frame[column].map(_repair_text)
    frame["valuation_date"] = frame["valuation_date"].map(_parse_date)
    frame["last_visit_date"] = frame["last_visit_date"].map(_parse_date)
    return frame.reset_index(drop=True)


def _load_asset_values() -> pd.DataFrame:
    """Load asset values."""
    raw = _load_table("table_137.csv")
    frame = raw.rename(
        columns={
            "col_1": "asset_number",
            "col_2": "asset_name",
            "col_3": "city",
            "col_4": "zone",
            "col_5": "valuation_date",
            "col_6": "fair_value_eur",
        }
    ).copy()
    frame = frame.iloc[1:].copy()
    frame["asset_number"] = frame["asset_number"].astype(int)
    for column in ["asset_name", "city", "zone"]:
        frame[column] = frame[column].map(_repair_text)
    frame["valuation_date"] = frame["valuation_date"].map(_parse_date)
    frame["fair_value_eur"] = frame["fair_value_eur"].map(_parse_euro_number)
    return frame.reset_index(drop=True)


def _load_zone_yields() -> pd.DataFrame:
    """Load zone yields."""
    raw = _load_table("table_135.csv")
    frame = raw.rename(
        columns={
            "col_1": "zone",
            "col_3": "vlm_range_eur_sqm_year",
            "col_4": "vacancy_months_range",
            "col_5": "cap_rate_range_pct",
        }
    ).copy()
    frame = frame.iloc[2:].copy()
    frame["zone"] = frame["zone"].map(_repair_text)
    frame["vlm_range_eur_sqm_year"] = frame["vlm_range_eur_sqm_year"].map(_repair_text)
    frame["vacancy_months_range"] = frame["vacancy_months_range"].map(_repair_text)
    frame["cap_rate_range_pct"] = frame["cap_rate_range_pct"].map(_repair_text)

    frame[["vlm_min_eur_sqm_year", "vlm_max_eur_sqm_year"]] = frame["vlm_range_eur_sqm_year"].apply(
        lambda value: pd.Series(_parse_range(value))
    )
    frame[["vacancy_min_months", "vacancy_max_months"]] = frame["vacancy_months_range"].apply(
        lambda value: pd.Series(_parse_range(value))
    )
    frame[["yield_min_pct", "yield_max_pct"]] = frame["cap_rate_range_pct"].apply(
        lambda value: pd.Series(_parse_range(value))
    )
    frame["yield_mid_pct"] = (frame["yield_min_pct"] + frame["yield_max_pct"]) / 2.0
    return frame.reset_index(drop=True)


def build_scbsm_asset_dataset(output_path: Path = SEED_ASSETS_PATH) -> pd.DataFrame:
    """Build SCBSM asset dataset."""
    ensure_outreach_dirs()

    metadata = _load_asset_metadata()
    values = _load_asset_values()
    zone_yields = _load_zone_yields()

    assets = metadata.merge(
        values[["asset_number", "fair_value_eur"]],
        how="inner",
        on="asset_number",
        validate="one_to_one",
    ).merge(
        zone_yields,
        how="left",
        on="zone",
        validate="many_to_one",
    )

    assets["asset_id"] = assets["asset_number"].map(lambda value: f"scbsm_2024_{value:02d}")
    assets["fair_value_eur_mn"] = assets["fair_value_eur"] / 1_000_000
    assets["asset_class"] = assets.apply(
        lambda row: _infer_asset_class(row["asset_name"], row["zone"]),
        axis=1,
    )
    assets["investment_profile"] = assets["zone"].map(ZONE_STRATEGY).fillna("Generalist")
    assets["yield_source"] = "SCBSM URD 2024 table_135 zone-level capitalisation range"
    assets["yield_precision"] = "zone_range"
    assets["valuation_date"] = pd.to_datetime(assets["valuation_date"]).dt.date.astype(str)
    assets["last_visit_date"] = pd.to_datetime(assets["last_visit_date"]).dt.date.astype(str)
    assets["generated_at_utc"] = datetime.now(UTC).isoformat()

    ordered_columns = [
        "asset_id",
        "asset_number",
        "asset_name",
        "asset_class",
        "city",
        "zone",
        "investment_profile",
        "valuation_date",
        "last_visit_date",
        "fair_value_eur",
        "fair_value_eur_mn",
        "vlm_range_eur_sqm_year",
        "vlm_min_eur_sqm_year",
        "vlm_max_eur_sqm_year",
        "vacancy_months_range",
        "vacancy_min_months",
        "vacancy_max_months",
        "cap_rate_range_pct",
        "yield_min_pct",
        "yield_max_pct",
        "yield_mid_pct",
        "yield_precision",
        "yield_source",
        "generated_at_utc",
    ]

    assets = assets.loc[:, ordered_columns].sort_values("asset_number").reset_index(drop=True)
    assets.to_csv(output_path, index=False)
    _write_extraction_note(assets)
    return assets


def _write_extraction_note(assets: pd.DataFrame) -> None:
    """Write extraction note."""
    note = f"""# SCBSM Yield Extraction

This dataset was built from the SCBSM 2024 universal registration document already scraped into `data/raw/scbsm/tables/scbsm-2024-06-30-fr/`.

## Source tables used

1. `table_134.csv` supplies the asset list, geography, valuation date, and the last Cushman & Wakefield visit date.
2. `table_137.csv` supplies the asset-level fair value (`Juste Valeur (€ HFA)`).
3. `table_135.csv` supplies the capitalisation-rate assumptions (`Taux de capitalisation`) by geographic zone: `Paris`, `IDF`, and `Province`.

## Join logic

1. I keyed the asset list on `asset_number` from tables `134` and `137`.
2. I then joined the zone-level capitalisation assumptions from `table_135` back onto each asset via the `zone` field.
3. Because the URD reports the yield as a **zone range**, not as a single asset-specific point estimate, I materialised three fields:
   - `yield_min_pct`
   - `yield_max_pct`
   - `yield_mid_pct`

## What the yield means here

The `yield_*` fields are **not bespoke expert yields for each asset**. They are the capitalisation-rate band disclosed for the asset's zone in the expertise assumptions table. That means:

- every Paris asset shares the Paris cap-rate band;
- every IDF asset shares the IDF band;
- every Province asset shares the Province band.

This is still useful for outreach prioritisation because it gives the selection algorithm a transparent benchmark to compare against each contact's target yield range. It should not be represented as a uniquely appraised yield per building.

## Current output summary

- Assets normalised: {len(assets)}
- Zones covered: {", ".join(sorted(assets["zone"].dropna().unique().tolist()))}
- Fair value total (EUR mn): {assets["fair_value_eur_mn"].sum():.1f}

## Practical consequence for the outreach app

The outreach algorithm uses `yield_mid_pct` as the comparable benchmark when ranking contacts, while still surfacing the full disclosed band (`cap_rate_range_pct`) in the contact fiche.
"""
    YIELD_EXTRACTION_NOTE_PATH.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Build a clean SCBSM asset dataset with zone-level yield assumptions for the outreach application."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SEED_ASSETS_PATH,
        help="CSV path to write the normalised asset dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the module entry point."""
    args = parse_args()
    assets = build_scbsm_asset_dataset(output_path=args.output)
    print("SCBSM outreach asset dataset written.")
    print(f"  assets: {len(assets)}")
    print(f"  zones: {', '.join(sorted(assets['zone'].unique().tolist()))}")
    print(f"  output: {args.output}")
    print(f"  note: {YIELD_EXTRACTION_NOTE_PATH}")


if __name__ == "__main__":
    main()
