from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.pipeline import (
    DEFAULT_PREQIN_PATH,
    assign_country_group,
    canonicalise_country,
    filter_preqin_transactions,
    load_preqin_transactions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_RESULTS_PATH = PROJECT_ROOT / "model" / "artifacts" / "scenario_analysis" / "scenario_results.csv"

ASSET_TYPE_MAP = {
    "Mixed Commercial": "Mixed Use",
}


@dataclass(frozen=True)
class ComparableQuery:
    asset_type: str
    country: str
    city: str
    size_sqm: float | None
    transaction_year: int | None
    cap_rate_pct: float | None


def _slug_text(value: Any) -> str:
    return "".join(character if character.isalnum() else " " for character in str(value).lower()).strip()


def _normalise_asset_type(value: str) -> str:
    token = str(value or "").strip()
    return ASSET_TYPE_MAP.get(token, token)


@lru_cache(maxsize=1)
def load_prepared_comparables() -> pd.DataFrame:
    frame = filter_preqin_transactions(load_preqin_transactions()).copy()
    frame["primary_asset_type"] = frame["primary_asset_type"].astype(str)
    frame["country"] = frame["country"].astype(str)
    frame["country_group"] = frame["country_group"].astype(str)
    frame["asset_city"] = frame["asset_city"].fillna("").astype(str)
    frame["price_per_sqm_eur"] = pd.to_numeric(frame["price_per_sqm_eur"], errors="coerce")
    frame["deal_size_winsorized_eur_mn"] = pd.to_numeric(frame["deal_size_winsorized_eur_mn"], errors="coerce")
    frame["TOTAL SIZE (SQ. M.)"] = pd.to_numeric(frame["TOTAL SIZE (SQ. M.)"], errors="coerce")
    frame["INITIAL CAPITALIZATION RATE (%)"] = pd.to_numeric(frame["INITIAL CAPITALIZATION RATE (%)"], errors="coerce")
    return frame.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_scenario_reference() -> pd.DataFrame:
    frame = pd.read_csv(SCENARIO_RESULTS_PATH)
    frame["scenario"] = frame["scenario"].astype(str)
    return frame


def available_comparable_asset_types() -> list[str]:
    frame = load_prepared_comparables()
    return sorted(frame["primary_asset_type"].dropna().astype(str).unique().tolist())


def available_comparable_countries() -> list[str]:
    frame = load_prepared_comparables()
    return sorted(frame["country"].dropna().astype(str).unique().tolist())


@lru_cache(maxsize=1)
def comparable_dataset_status() -> dict[str, Any]:
    frame = load_prepared_comparables()
    return {
        "source_label": "Available Preqin single-asset transaction dataset",
        "source_file": DEFAULT_PREQIN_PATH.name,
        "available_rows": int(len(frame)),
        "available_dataset_only": True,
        "mock_data_used": False,
        "dataset_note": (
            "Comparable retrieval runs only on the currently available transaction dataset. "
            "No mock or synthetic completeness-benchmark data are used in selection."
        ),
    }


def classify_comparable_scenario(*, has_size: bool, has_year: bool, cap_rate_pct: float | None) -> dict[str, Any]:
    if has_size and has_year:
        scenario_code = "A"
    elif has_size or has_year:
        scenario_code = "B"
    else:
        scenario_code = "C"

    reference = load_scenario_reference().set_index("scenario").loc[scenario_code]
    lower = min(float(reference["headline_fold_mape_pct"]), float(reference["rolling_mean_mape_pct"]))
    upper = max(float(reference["headline_fold_mape_pct"]), float(reference["rolling_mean_mape_pct"]))
    enhanced_mode = cap_rate_pct is not None

    return {
        "scenario": scenario_code,
        "label": str(reference["scenario_label"]).replace("â€”", "-").replace("\u2014", "-"),
        "expected_mape_range_pct": (lower, upper),
        "headline_mape_pct": float(reference["headline_fold_mape_pct"]),
        "rolling_mean_mape_pct": float(reference["rolling_mean_mape_pct"]),
        "enhanced_mode_note": (
            "Enhanced mode: the entered cap rate is shown for context alongside the comp set; "
            "it does not convert this prototype into a point valuation."
            if enhanced_mode
            else ""
        ),
    }


def _base_scored_pool(
    *,
    frame: pd.DataFrame,
    country: str,
    city: str,
    size_sqm: float | None,
    transaction_year: int | None,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["country_match"] = scored["country"].eq(country).astype(float)
    city_token = _slug_text(city)
    scored["city_match"] = scored["asset_city"].map(_slug_text).eq(city_token).astype(float) if city_token else 0.0
    if size_sqm is None or size_sqm <= 0:
        scored["log_size_gap"] = np.nan
        scored["size_similarity_score"] = 0.0
    else:
        query_log_size = float(np.log(max(size_sqm, 1.0)))
        scored["log_size_gap"] = np.abs(query_log_size - np.log(scored["TOTAL SIZE (SQ. M.)"].clip(lower=1.0)))
        scored["size_similarity_score"] = 40.0 / (1.0 + 4.0 * scored["log_size_gap"])

    if transaction_year is None:
        scored["year_gap"] = np.nan
    else:
        scored["year_gap"] = (scored["transaction_year"].astype(float) - float(transaction_year)).abs()

    scored["similarity_score"] = 60.0 + 25.0 * scored["country_match"] + 5.0 * scored["city_match"] + scored["size_similarity_score"]
    return scored.sort_values(
        ["country_match", "city_match", "log_size_gap", "year_gap", "DEAL DATE"],
        ascending=[False, False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def retrieve_comparables(query: ComparableQuery, top_k: int = 10) -> dict[str, Any]:
    frame = load_prepared_comparables()
    asset_type = _normalise_asset_type(query.asset_type)
    country = canonicalise_country(query.country)
    country_group = assign_country_group(country)
    asset_pool = frame.loc[frame["primary_asset_type"].eq(asset_type)].copy()
    exact_pool = asset_pool.loc[asset_pool["country"].eq(country)].copy()

    scope = "Type x country"
    widened = False
    pool = exact_pool
    if len(exact_pool) < 5:
        widened = True
        scope = f"Type x region ({country_group})"
        pool = asset_pool.loc[asset_pool["country_group"].eq(country_group)].copy()
        if len(pool) < 5:
            scope = "Type x Europe"
            pool = asset_pool.copy()

    scored = _base_scored_pool(
        frame=pool,
        country=country,
        city=query.city,
        size_sqm=query.size_sqm,
        transaction_year=query.transaction_year,
    )
    ranked = scored.head(top_k).copy()
    ranked["retrieval_scope"] = scope
    ranked["widened_from_exact_country"] = widened

    benchmark_pool = exact_pool.copy()
    benchmark_n = int(len(benchmark_pool))
    benchmark_ppsqm = float(benchmark_pool["price_per_sqm_winsorized_eur"].median()) if benchmark_n else np.nan
    implied_value = (
        float(benchmark_ppsqm * query.size_sqm / 1_000_000)
        if benchmark_n and query.size_sqm is not None and query.size_sqm > 0
        else np.nan
    )
    if len(exact_pool) == 0:
        coverage_note = (
            "No exact country matches were found in the available dataset, so the result relies entirely on widened coverage."
        )
    elif len(exact_pool) < 5:
        coverage_note = (
            "The exact country cell is thin in the available dataset, so the result widens to preserve a usable comp set."
        )
    elif len(exact_pool) < top_k:
        coverage_note = (
            "The exact country cell exists in the available dataset but has fewer rows than the requested top-k."
        )
    else:
        coverage_note = "The requested query is covered directly inside the available dataset."

    dataset_status = comparable_dataset_status() | {
        "query_asset_pool_rows": int(len(asset_pool)),
        "query_exact_country_rows": int(len(exact_pool)),
        "coverage_flag": len(exact_pool) < top_k,
        "coverage_note": coverage_note,
    }

    return {
        "query": asdict(query),
        "retrieval_scope": scope,
        "widened": widened,
        "exact_match_count": int(len(exact_pool)),
        "pool_count": int(len(pool)),
        "dataset_status": dataset_status,
        "results": ranked,
        "benchmark": {
            "asset_type": asset_type,
            "country": country,
            "median_price_per_sqm_eur": benchmark_ppsqm,
            "sample_size": benchmark_n,
            "thin_cell": benchmark_n < 5,
            "implied_deal_value_eur_mn": implied_value,
        },
    }


def format_comparable_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Similarity score",
                "Deal date",
                "Country",
                "City",
                "Asset type",
                "Size (sqm)",
                "Deal size (EUR mn)",
                "Price / sqm (EUR)",
                "Initial cap rate (%)",
                "Scope",
            ]
        )

    display = frame.loc[
        :,
        [
            "similarity_score",
            "DEAL DATE",
            "country",
            "asset_city",
            "primary_asset_type",
            "TOTAL SIZE (SQ. M.)",
            "DEAL SIZE (EUR MN)",
            "price_per_sqm_winsorized_eur",
            "INITIAL CAPITALIZATION RATE (%)",
            "retrieval_scope",
        ],
    ].copy()
    display["similarity_score"] = display["similarity_score"].round(1)
    display["DEAL DATE"] = pd.to_datetime(display["DEAL DATE"], errors="coerce").dt.date.astype(str)
    display["TOTAL SIZE (SQ. M.)"] = display["TOTAL SIZE (SQ. M.)"].round(0)
    display["DEAL SIZE (EUR MN)"] = display["DEAL SIZE (EUR MN)"].round(1)
    display["price_per_sqm_winsorized_eur"] = display["price_per_sqm_winsorized_eur"].round(0)
    display["INITIAL CAPITALIZATION RATE (%)"] = display["INITIAL CAPITALIZATION RATE (%)"].round(2)
    return display.rename(
        columns={
            "similarity_score": "Similarity score",
            "DEAL DATE": "Deal date",
            "country": "Country",
            "asset_city": "City",
            "primary_asset_type": "Asset type",
            "TOTAL SIZE (SQ. M.)": "Size (sqm)",
            "DEAL SIZE (EUR MN)": "Deal size (EUR mn)",
            "price_per_sqm_winsorized_eur": "Price / sqm (EUR)",
            "INITIAL CAPITALIZATION RATE (%)": "Initial cap rate (%)",
            "retrieval_scope": "Scope",
        }
    )
