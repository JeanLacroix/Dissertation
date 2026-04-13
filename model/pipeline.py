from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDICES_DIR = DATA_DIR / "indices"

DEFAULT_PREQIN_PATH = RAW_DIR / "Preqin_RealEstate_Deals-13_04_2026.xlsx"
DEFAULT_CAPIQ_PATH = RAW_DIR / "Data_test_2_cap_IQ_since_2021.xls"
DEFAULT_ECB_HICP_PATH = INDICES_DIR / "ecb_hicp_housing_water_electricity.csv"
DEFAULT_UK_CPI_PATH = INDICES_DIR / "uk_cpi_inflation_rate.csv"

PREQIN_REQUIRED_COLUMNS = [
    "DEAL ID",
    "DEAL NAME",
    "DEAL DATE",
    "DEAL TYPE",
    "PRIMARY ASSET TYPE",
    "ASSET REGIONS",
    "ASSET COUNTRIES",
    "ASSET CITIES",
    "DEAL SIZE (EUR MN)",
    "TOTAL SIZE (SQ. M.)",
    "INITIAL CAPITALIZATION RATE (%)",
]

ASSET_TYPE_LEVELS = [
    "Office",
    "Industrial",
    "Retail",
    "Mixed Use",
    "Hotel",
    "Niche",
    "Residential",
    "Land",
]

COUNTRY_GROUP_LEVELS = [
    "United Kingdom",
    "Netherlands",
    "Spain",
    "Germany",
    "France",
    "Other Europe",
]

COUNTRY_ALIASES = {
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "netherlands": "Netherlands",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    "spain": "Spain",
    "germany": "Germany",
    "france": "France",
}

ECB_CODE_TO_GROUP = {
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "NL": "Netherlands",
    "U2": "Other Europe",
}

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _normalise_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _split_first_country(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in re.split(r"[;|/]", text) if part.strip()]
    return parts[0] if parts else text


def canonicalise_country(value: Any) -> str:
    token = _normalise_token(_split_first_country(value))
    if token in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[token]
    return _split_first_country(value).strip() or "Other Europe"


def assign_country_group(country: str) -> str:
    if country in {"United Kingdom", "Netherlands", "Spain", "Germany", "France"}:
        return country
    return "Other Europe"


def load_preqin_transactions(path: Path | str = DEFAULT_PREQIN_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [column for column in PREQIN_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Preqin file is missing required columns: {missing}")
    return df


def filter_preqin_transactions(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["DEAL DATE"] = pd.to_datetime(frame["DEAL DATE"], errors="coerce")
    frame["DEAL SIZE (EUR MN)"] = pd.to_numeric(frame["DEAL SIZE (EUR MN)"], errors="coerce")
    frame["TOTAL SIZE (SQ. M.)"] = pd.to_numeric(frame["TOTAL SIZE (SQ. M.)"], errors="coerce")
    frame["INITIAL CAPITALIZATION RATE (%)"] = pd.to_numeric(
        frame["INITIAL CAPITALIZATION RATE (%)"],
        errors="coerce",
    )

    mask = (
        frame["DEAL TYPE"].astype(str).str.strip().eq("Single Asset")
        & frame["ASSET REGIONS"].astype(str).str.contains("Europe", case=False, na=False)
        & frame["DEAL SIZE (EUR MN)"].notna()
        & frame["TOTAL SIZE (SQ. M.)"].notna()
        & frame["DEAL DATE"].notna()
        & frame["TOTAL SIZE (SQ. M.)"].gt(0)
    )

    frame = frame.loc[mask, PREQIN_REQUIRED_COLUMNS].copy()
    frame["country"] = frame["ASSET COUNTRIES"].map(canonicalise_country)
    frame["country_group"] = frame["country"].map(assign_country_group)
    frame["primary_asset_type"] = frame["PRIMARY ASSET TYPE"].astype(str).str.strip()
    frame["asset_city"] = frame["ASSET CITIES"].fillna("").astype(str).str.strip()
    frame["transaction_year"] = frame["DEAL DATE"].dt.year.astype("Int64")
    frame["transaction_quarter"] = frame["DEAL DATE"].dt.to_period("Q").astype(str)
    frame["price_per_sqm_eur"] = (frame["DEAL SIZE (EUR MN)"] * 1_000_000) / frame["TOTAL SIZE (SQ. M.)"]

    winsor_low = frame["price_per_sqm_eur"].quantile(0.01)
    winsor_high = frame["price_per_sqm_eur"].quantile(0.99)
    frame["price_per_sqm_winsorized_eur"] = frame["price_per_sqm_eur"].clip(winsor_low, winsor_high)
    frame["deal_size_winsorized_eur_mn"] = (
        frame["price_per_sqm_winsorized_eur"] * frame["TOTAL SIZE (SQ. M.)"] / 1_000_000
    )

    frame["log_total_size_sqm"] = np.log(frame["TOTAL SIZE (SQ. M.)"])
    frame["log_deal_size_eur_mn"] = np.log(frame["deal_size_winsorized_eur_mn"])
    frame["primary_asset_type"] = pd.Categorical(
        frame["primary_asset_type"],
        categories=ASSET_TYPE_LEVELS,
    )
    frame["country_group"] = pd.Categorical(
        frame["country_group"],
        categories=COUNTRY_GROUP_LEVELS,
    )
    return frame.reset_index(drop=True)


def _extract_ecb_code(column_name: str) -> str | None:
    match = re.search(r"HICP\.M\.([A-Z0-9]{2})\.", column_name)
    return match.group(1) if match else None


def load_ecb_hicp_rates(path: Path | str = DEFAULT_ECB_HICP_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")

    series_frames: list[pd.DataFrame] = []
    for column in raw.columns:
        code = _extract_ecb_code(column)
        if code not in ECB_CODE_TO_GROUP:
            continue

        current = raw[["DATE", column]].copy()
        current = current.rename(columns={column: "rate_pct"})
        current["country_group"] = ECB_CODE_TO_GROUP[code]
        current["index_source"] = "ECB HICP rate proxy"
        series_frames.append(current)

    if not series_frames:
        raise ValueError("No recognised ECB HICP columns were found in the index file.")

    monthly = pd.concat(series_frames, ignore_index=True)
    monthly["rate_pct"] = pd.to_numeric(monthly["rate_pct"], errors="coerce")
    monthly = monthly.dropna(subset=["DATE", "rate_pct"])
    monthly["transaction_quarter"] = monthly["DATE"].dt.to_period("Q").astype(str)

    quarterly = (
        monthly.groupby(["country_group", "transaction_quarter", "index_source"], as_index=False)["rate_pct"]
        .mean()
        .sort_values(["country_group", "transaction_quarter"])
    )
    quarterly["index_value"] = _build_rebased_proxy_index(quarterly)
    return quarterly


def load_uk_cpi_rates(path: Path | str = DEFAULT_UK_CPI_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=["label", "rate_pct"])
    raw["label"] = raw["label"].astype(str).str.strip().str.replace('"', "", regex=False)
    raw["rate_pct"] = raw["rate_pct"].astype(str).str.strip().str.replace('"', "", regex=False)

    monthly_label = raw["label"].str.extract(r"^(?P<year>\d{4}) (?P<month>[A-Z]{3})$")
    monthly = raw.loc[monthly_label["year"].notna(), ["label", "rate_pct"]].copy()
    monthly[["year", "month"]] = monthly_label.loc[monthly.index]
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].map(MONTH_MAP)
    monthly["DATE"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1),
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    monthly["rate_pct"] = pd.to_numeric(monthly["rate_pct"], errors="coerce")
    monthly = monthly.dropna(subset=["DATE", "rate_pct"])
    monthly["country_group"] = "United Kingdom"
    monthly["index_source"] = "UK CPI rate proxy"
    monthly["transaction_quarter"] = monthly["DATE"].dt.to_period("Q").astype(str)

    quarterly = (
        monthly.groupby(["country_group", "transaction_quarter", "index_source"], as_index=False)["rate_pct"]
        .mean()
        .sort_values(["country_group", "transaction_quarter"])
    )
    quarterly["index_value"] = _build_rebased_proxy_index(quarterly)
    return quarterly


def _build_rebased_proxy_index(quarterly_rates: pd.DataFrame, anchor_quarter: str = "2021Q1") -> pd.Series:
    rebased = pd.Series(index=quarterly_rates.index, dtype=float)

    # The source files are annual inflation-rate series rather than level indices.
    # We therefore build a simple chained quarterly proxy anchored at 2021Q1 = 100.
    for _, group in quarterly_rates.groupby("country_group", sort=False):
        ordered = group.sort_values("transaction_quarter").copy()
        periods = pd.PeriodIndex(ordered["transaction_quarter"], freq="Q")
        factors = 1.0 + ordered["rate_pct"].astype(float) / 100.0

        if (factors <= 0).any():
            raise ValueError("Encountered a rate at or below -100%, cannot build a positive proxy index.")

        values = np.empty(len(ordered), dtype=float)
        if anchor_quarter in ordered["transaction_quarter"].values:
            anchor_pos = ordered.index.get_loc(
                ordered.index[ordered["transaction_quarter"].eq(anchor_quarter)][0]
            )
        else:
            anchor_pos = int(np.searchsorted(periods.astype(str).to_numpy(), anchor_quarter))
            anchor_pos = min(max(anchor_pos, 0), len(ordered) - 1)

        values[anchor_pos] = 100.0

        for position in range(anchor_pos + 1, len(ordered)):
            values[position] = values[position - 1] * factors.iloc[position]

        for position in range(anchor_pos - 1, -1, -1):
            values[position] = values[position + 1] / factors.iloc[position + 1]

        rebased.loc[ordered.index] = values

    return rebased


def load_macro_index_frame(
    ecb_path: Path | str = DEFAULT_ECB_HICP_PATH,
    uk_path: Path | str = DEFAULT_UK_CPI_PATH,
) -> pd.DataFrame:
    euro = load_ecb_hicp_rates(ecb_path)
    uk = load_uk_cpi_rates(uk_path)
    frame = pd.concat([euro, uk], ignore_index=True)
    frame["country_group"] = pd.Categorical(frame["country_group"], categories=COUNTRY_GROUP_LEVELS)
    return frame.sort_values(["country_group", "transaction_quarter"]).reset_index(drop=True)


def merge_macro_indices(transactions: pd.DataFrame, index_frame: pd.DataFrame) -> pd.DataFrame:
    transaction_frame = transactions.copy()
    transaction_frame["_row_order"] = np.arange(len(transaction_frame))
    transaction_frame["_quarter_period"] = pd.PeriodIndex(transaction_frame["transaction_quarter"], freq="Q")

    macro_frame = index_frame.copy()
    macro_frame["_quarter_period"] = pd.PeriodIndex(macro_frame["transaction_quarter"], freq="Q")

    merged_parts: list[pd.DataFrame] = []
    for country_group, transaction_group in transaction_frame.groupby("country_group", observed=False, sort=False):
        macro_group = macro_frame.loc[macro_frame["country_group"] == country_group].copy()
        if macro_group.empty:
            transaction_group["index_source"] = pd.NA
            transaction_group["rate_pct"] = np.nan
            transaction_group["index_value"] = np.nan
            merged_parts.append(transaction_group)
            continue

        macro_group = (
            macro_group.sort_values("_quarter_period")
            .drop_duplicates(subset=["_quarter_period"], keep="last")
            .set_index("_quarter_period")[["index_source", "rate_pct", "index_value"]]
        )

        quarter_index = macro_group.index.union(transaction_group["_quarter_period"]).sort_values()
        macro_group = macro_group.reindex(quarter_index).ffill()

        aligned = macro_group.loc[transaction_group["_quarter_period"], ["index_source", "rate_pct", "index_value"]]
        aligned = aligned.reset_index(drop=True)

        enriched_group = transaction_group.reset_index(drop=True).copy()
        enriched_group[["index_source", "rate_pct", "index_value"]] = aligned
        merged_parts.append(enriched_group)

    merged = pd.concat(merged_parts, ignore_index=True).sort_values("_row_order")
    merged = merged.drop(columns=["_row_order", "_quarter_period"])
    merged["log_index_value"] = np.log(merged["index_value"])
    return merged


def _pick_best_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for candidate in candidates:
        for upper_name, original_name in lookup.items():
            if candidate in upper_name:
                return original_name
    return None


def load_capiq_year_built(path: Path | str = DEFAULT_CAPIQ_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, header=4, skiprows=[5, 6])

    name_col = _pick_best_column(raw.columns, ["PPTY_NAME", "PROPERTY NAME", "ASSET NAME", "INSTN_NAME", "NAME"])
    date_col = _pick_best_column(raw.columns, ["SOLD_DATE", "ANN_DT", "SALE DATE", "TRANSACTION DATE", "CLOSE DATE"])
    size_col = _pick_best_column(raw.columns, ["PPTY_SIZE_AREA", "PROPERTY SIZE", "BUILDING SIZE"])
    year_built_col = _pick_best_column(raw.columns, ["YR_BUILT", "YEAR BUILT"])

    if not all([name_col, date_col, size_col, year_built_col]):
        return pd.DataFrame(columns=["match_name", "transaction_date", "size_sqm", "year_built"])

    frame = raw[[name_col, date_col, size_col, year_built_col]].copy()
    frame.columns = ["match_name", "transaction_date", "size_area_raw", "year_built"]
    frame["match_name"] = frame["match_name"].astype(str).str.strip()
    frame = frame.loc[~frame["match_name"].str.startswith(("+", "*"), na=False)].copy()

    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame["size_area_raw"] = pd.to_numeric(frame["size_area_raw"], errors="coerce")
    frame["size_sqm"] = frame["size_area_raw"] * 0.0929
    frame["year_built"] = pd.to_numeric(frame["year_built"], errors="coerce").astype("Int64")
    frame["match_name_norm"] = frame["match_name"].map(_normalise_token)
    return frame.dropna(subset=["transaction_date", "size_sqm"]).reset_index(drop=True)


def enrich_with_year_built(
    transactions: pd.DataFrame,
    capiq_path: Path | str = DEFAULT_CAPIQ_PATH,
    min_matches: int = 150,
    score_threshold: int = 85,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = transactions.copy()
    frame["year_built"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["deal_name_norm"] = frame["DEAL NAME"].map(_normalise_token)

    capiq_file = Path(capiq_path)
    if not capiq_file.exists():
        return frame.drop(columns=["deal_name_norm"]), {
            "year_built_enabled": False,
            "matched_rows": 0,
            "reason": "Capital IQ file not found.",
        }

    capiq = load_capiq_year_built(capiq_file)
    if capiq.empty:
        return frame.drop(columns=["deal_name_norm"]), {
            "year_built_enabled": False,
            "matched_rows": 0,
            "reason": "Capital IQ file did not expose the expected matching columns.",
        }

    matched_rows = 0
    for index, row in frame.iterrows():
        if row["deal_name_norm"] == "":
            continue

        size_lower = row["TOTAL SIZE (SQ. M.)"] * 0.9
        size_upper = row["TOTAL SIZE (SQ. M.)"] * 1.1
        start_date = row["DEAL DATE"] - pd.Timedelta(days=30)
        end_date = row["DEAL DATE"] + pd.Timedelta(days=30)

        candidates = capiq.loc[
            capiq["transaction_date"].between(start_date, end_date)
            & capiq["size_sqm"].between(size_lower, size_upper)
            & capiq["year_built"].notna()
        ].copy()
        if candidates.empty:
            continue

        candidates["score"] = candidates["match_name_norm"].map(
            lambda candidate: fuzz.token_sort_ratio(candidate, row["deal_name_norm"])
        )
        best = candidates.sort_values(["score", "transaction_date"], ascending=[False, True]).iloc[0]
        if int(best["score"]) < score_threshold:
            continue

        frame.at[index, "year_built"] = best["year_built"]
        matched_rows += 1

    year_built_enabled = matched_rows >= min_matches
    if not year_built_enabled:
        frame["year_built"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    return frame.drop(columns=["deal_name_norm"]), {
        "year_built_enabled": year_built_enabled,
        "matched_rows": matched_rows,
        "reason": (
            "Retained in model."
            if year_built_enabled
            else f"Dropped from model because only {matched_rows} rows matched, below the {min_matches} threshold."
        ),
    }


def build_training_frame(
    preqin_path: Path | str = DEFAULT_PREQIN_PATH,
    ecb_path: Path | str = DEFAULT_ECB_HICP_PATH,
    uk_path: Path | str = DEFAULT_UK_CPI_PATH,
    capiq_path: Path | str = DEFAULT_CAPIQ_PATH,
    min_year_built_matches: int = 150,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_preqin_transactions(preqin_path)
    filtered = filter_preqin_transactions(raw)
    macro = load_macro_index_frame(ecb_path=ecb_path, uk_path=uk_path)
    merged = merge_macro_indices(filtered, macro)
    enriched, year_built_stats = enrich_with_year_built(
        merged,
        capiq_path=capiq_path,
        min_matches=min_year_built_matches,
    )

    usable = enriched.dropna(subset=["index_value", "log_index_value"]).reset_index(drop=True)
    winsor_bounds = filtered["price_per_sqm_eur"].quantile([0.01, 0.99]).to_dict()

    metadata = {
        "raw_row_count": int(len(raw)),
        "filtered_row_count": int(len(filtered)),
        "usable_row_count": int(len(usable)),
        "country_group_counts": {
            str(key): int(value)
            for key, value in usable["country_group"].value_counts(dropna=False).sort_index().items()
        },
        "asset_type_counts": {
            str(key): int(value)
            for key, value in usable["primary_asset_type"].value_counts(dropna=False).sort_index().items()
        },
        "winsorization": {
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "lower_bound_eur_per_sqm": float(winsor_bounds[0.01]),
            "upper_bound_eur_per_sqm": float(winsor_bounds[0.99]),
        },
        "index_sources": {
            "United Kingdom": "UK CPI rate proxy",
            "Netherlands": "ECB HICP rate proxy",
            "Spain": "ECB HICP rate proxy",
            "Germany": "ECB HICP rate proxy",
            "France": "ECB HICP rate proxy",
            "Other Europe": "ECB HICP rate proxy (U2 aggregate)",
        },
        "macro_proxy_note": (
            "Quarterly macro index values are chained from annual inflation-rate series and rebased to 2021Q1 = 100. "
            "This is a pragmatic proxy because level property indices were not used in this version."
        ),
        "year_built": year_built_stats,
    }
    return usable, metadata


if __name__ == "__main__":
    dataset, summary = build_training_frame()
    print(f"Filtered rows: {summary['filtered_row_count']}")
    print(f"Usable rows after macro merge: {summary['usable_row_count']}")
    print(f"Year-built enrichment enabled: {summary['year_built']['year_built_enabled']}")
