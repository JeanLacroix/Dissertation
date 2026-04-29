"""Train, evaluate, and export the retained Change D retrieval artefacts."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

from .pipeline import (
    ASSET_TYPE_LEVELS,
    COUNTRY_GROUP_LEVELS,
    PROJECT_ROOT,
    build_training_frame,
)

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
RESIDUALS_PATH = ARTIFACTS_DIR / "residuals.npy"
COMPS_SAMPLE_PATH = ARTIFACTS_DIR / "comps_sample.parquet"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
CHANGE_D_METRICS_PATH = ARTIFACTS_DIR / "stage_two_refits" / "change_d_metrics.json"
CHANGE_D_FORMULA = "log_deal_size_eur_mn ~ log_total_size_sqm * C(primary_asset_type) + C(country_group) + C(model_year_effect)"

BASE_FORMULA = (
    "log_deal_size_eur_mn ~ log_total_size_sqm + C(primary_asset_type) + "
    "C(country_group) + log_index_value"
)
BOOTSTRAP_RESIDUAL_COUNT = 1_000
RANDOM_SEED = 42
YEAR_FE_CAP = 2025

ROLLING_FOLD_SPECS = [
    ("fold_1", [2021, 2022], 2023),
    ("fold_2", [2021, 2022, 2023], 2024),
    ("fold_3", [2021, 2022, 2023, 2024], 2025),
    ("fold_4", [2021, 2022, 2023, 2024, 2025], 2026),
]

SIZE_BUCKET_BINS = [0, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, np.inf]
PRICE_BUCKET_BINS = [0, 5, 10, 25, 50, 100, 250, 500, 1_000, np.inf]
PRICE_PER_SQM_BUCKET_BINS = [0, 1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 25_000, np.inf]


@dataclass(frozen=True)
class TrainingOutputs:
    """Collect the exported artefacts and metadata from model training."""
    model: Any
    model_frame: pd.DataFrame
    metadata: dict[str, Any]


def _build_formula(include_year_built: bool) -> str:
    """Build formula."""
    if include_year_built:
        return f"{BASE_FORMULA} + year_built"
    return BASE_FORMULA


def _prepare_model_frame(dataset: pd.DataFrame, pipeline_metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare model frame."""
    frame = dataset.copy()
    frame = frame.loc[np.isfinite(frame["log_deal_size_eur_mn"]) & np.isfinite(frame["log_total_size_sqm"])].copy()

    year_built_enabled = bool(pipeline_metadata["year_built"]["year_built_enabled"])
    missing_year_built_rows = int(frame["year_built"].isna().sum())
    if year_built_enabled:
        frame = frame.dropna(subset=["year_built"]).copy()

    frame["actual_deal_size_eur_mn"] = frame["deal_size_winsorized_eur_mn"]
    frame["actual_price_per_sqm_eur"] = frame["price_per_sqm_winsorized_eur"]
    frame["primary_asset_type"] = pd.Categorical(frame["primary_asset_type"], categories=ASSET_TYPE_LEVELS)
    frame["country_group"] = pd.Categorical(frame["country_group"], categories=COUNTRY_GROUP_LEVELS)

    prep_metadata = {
        "include_year_built": year_built_enabled,
        "rows_available_for_model": int(len(frame)),
        "rows_removed_for_missing_year_built": missing_year_built_rows if year_built_enabled else 0,
    }
    return frame.reset_index(drop=True), prep_metadata


def _fit_ols(train_frame: pd.DataFrame, formula: str):
    """Fit OLS."""
    return smf.ols(formula=formula, data=train_frame).fit()


def _map_prediction_year_effect(year_value: int, available_years: list[int]) -> int:
    """Map prediction year effect."""
    capped_year = min(int(year_value), YEAR_FE_CAP)
    eligible_years = [year for year in available_years if year <= capped_year]
    if eligible_years:
        return max(eligible_years)
    return min(available_years)


def _prepare_formula_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    formula: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare formula frames."""
    train = train_frame.copy()
    test = test_frame.copy()
    if "C(model_year_effect)" in formula:
        train["model_year_effect"] = train["model_year_effect"].astype(int)
        available_years = sorted(train["model_year_effect"].dropna().astype(int).unique().tolist())
        test["model_year_effect"] = test["transaction_year"].astype(int).map(
            lambda year_value: _map_prediction_year_effect(year_value, available_years)
        )
        train["model_year_effect"] = pd.Categorical(train["model_year_effect"], categories=available_years, ordered=True)
        test["model_year_effect"] = pd.Categorical(test["model_year_effect"], categories=available_years, ordered=True)
    return train, test


def _predict_deal_size_eur_mn(model, frame: pd.DataFrame) -> np.ndarray:
    """Predict deal size EUR mn."""
    log_predictions = model.predict(frame)
    return np.exp(np.asarray(log_predictions, dtype=float))


def _evaluate_predictions(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """Evaluate predictions."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    percentage_errors = np.abs((actual_array - predicted_array) / actual_array)
    rmse = float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))
    return {
        "mape_pct": float(np.mean(percentage_errors) * 100.0),
        "rmse_eur_mn": rmse,
    }


def _baseline_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    """Baseline predict."""
    train_copy = train_frame.copy()
    train_copy["asset_type_key"] = train_copy["primary_asset_type"].astype(str)
    train_copy["country_key"] = train_copy["country_group"].astype(str)

    cell_lookup = (
        train_copy.groupby(["asset_type_key", "country_key"], observed=True)["actual_price_per_sqm_eur"]
        .median()
        .rename("cell_median_ppsqm")
        .reset_index()
    )
    asset_lookup = train_copy.groupby("asset_type_key", observed=True)["actual_price_per_sqm_eur"].median()
    country_lookup = train_copy.groupby("country_key", observed=True)["actual_price_per_sqm_eur"].median()
    global_median = float(train_copy["actual_price_per_sqm_eur"].median())

    test_copy = test_frame.copy()
    test_copy["asset_type_key"] = test_copy["primary_asset_type"].astype(str)
    test_copy["country_key"] = test_copy["country_group"].astype(str)
    test_copy = test_copy.merge(cell_lookup, how="left", on=["asset_type_key", "country_key"])
    test_copy["baseline_ppsqm"] = test_copy["cell_median_ppsqm"]
    test_copy["baseline_ppsqm"] = test_copy["baseline_ppsqm"].fillna(test_copy["asset_type_key"].map(asset_lookup))
    test_copy["baseline_ppsqm"] = test_copy["baseline_ppsqm"].fillna(test_copy["country_key"].map(country_lookup))
    test_copy["baseline_ppsqm"] = test_copy["baseline_ppsqm"].fillna(global_median)
    return (test_copy["baseline_ppsqm"] * test_copy["TOTAL SIZE (SQ. M.)"] / 1_000_000).to_numpy(dtype=float)


def _compute_lift(model_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    """Compute lift."""
    return {
        "mape_reduction_pct": float(
            ((baseline_metrics["mape_pct"] - model_metrics["mape_pct"]) / baseline_metrics["mape_pct"]) * 100.0
        ),
        "rmse_reduction_pct": float(
            ((baseline_metrics["rmse_eur_mn"] - model_metrics["rmse_eur_mn"]) / baseline_metrics["rmse_eur_mn"]) * 100.0
        ),
    }


def _run_fold(train_frame: pd.DataFrame, test_frame: pd.DataFrame, formula: str, fold_name: str) -> dict[str, Any]:
    """Run fold."""
    prepared_train, prepared_test = _prepare_formula_frames(train_frame, test_frame, formula)
    model = _fit_ols(prepared_train, formula)
    model_predictions = _predict_deal_size_eur_mn(model, prepared_test)
    baseline_predictions = _baseline_predict(prepared_train, prepared_test)

    model_metrics = _evaluate_predictions(prepared_test["actual_deal_size_eur_mn"], model_predictions)
    baseline_metrics = _evaluate_predictions(prepared_test["actual_deal_size_eur_mn"], baseline_predictions)

    return {
        "fold": fold_name,
        "train_rows": int(len(prepared_train)),
        "test_rows": int(len(prepared_test)),
        "test_year": int(prepared_test["transaction_year"].iloc[0]),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "lift_vs_baseline": _compute_lift(model_metrics, baseline_metrics),
    }


def run_rolling_origin_cv(model_frame: pd.DataFrame, formula: str) -> dict[str, Any]:
    """Run rolling origin cross-validation."""
    folds: list[dict[str, Any]] = []
    for fold_name, train_years, test_year in ROLLING_FOLD_SPECS:
        train_mask = model_frame["transaction_year"].isin(train_years)
        test_mask = model_frame["transaction_year"].eq(test_year)
        train_frame = model_frame.loc[train_mask].copy()
        test_frame = model_frame.loc[test_mask].copy()
        if train_frame.empty or test_frame.empty:
            continue
        folds.append(_run_fold(train_frame, test_frame, formula, fold_name))

    model_mapes = [fold["model_metrics"]["mape_pct"] for fold in folds]
    model_rmses = [fold["model_metrics"]["rmse_eur_mn"] for fold in folds]
    baseline_mapes = [fold["baseline_metrics"]["mape_pct"] for fold in folds]
    baseline_rmses = [fold["baseline_metrics"]["rmse_eur_mn"] for fold in folds]

    headline_fold = folds[-1] if folds else None
    return {
        "strategy": "rolling_origin_expanding_window",
        "folds": folds,
        "mean_mape_pct": float(np.mean(model_mapes)),
        "std_mape_pct": float(np.std(model_mapes, ddof=0)),
        "mean_rmse_eur_mn": float(np.mean(model_rmses)),
        "std_rmse_eur_mn": float(np.std(model_rmses, ddof=0)),
        "baseline_mean_mape_pct": float(np.mean(baseline_mapes)),
        "baseline_mean_rmse_eur_mn": float(np.mean(baseline_rmses)),
        "headline_fold": headline_fold,
    }


def run_random_cv(model_frame: pd.DataFrame, formula: str, random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """Run random cross-validation."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    folds: list[dict[str, Any]] = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(model_frame), start=1):
        train_frame = model_frame.iloc[train_idx].copy()
        test_frame = model_frame.iloc[test_idx].copy()
        folds.append(_run_fold(train_frame, test_frame, formula, fold_name=f"random_fold_{fold_number}"))

    model_mapes = [fold["model_metrics"]["mape_pct"] for fold in folds]
    model_rmses = [fold["model_metrics"]["rmse_eur_mn"] for fold in folds]
    baseline_mapes = [fold["baseline_metrics"]["mape_pct"] for fold in folds]
    baseline_rmses = [fold["baseline_metrics"]["rmse_eur_mn"] for fold in folds]

    return {
        "strategy": "random_5_fold",
        "folds": folds,
        "mean_mape_pct": float(np.mean(model_mapes)),
        "std_mape_pct": float(np.std(model_mapes, ddof=0)),
        "mean_rmse_eur_mn": float(np.mean(model_rmses)),
        "std_rmse_eur_mn": float(np.std(model_rmses, ddof=0)),
        "baseline_mean_mape_pct": float(np.mean(baseline_mapes)),
        "baseline_mean_rmse_eur_mn": float(np.mean(baseline_rmses)),
    }


def _format_bucket_labels(bins: list[float], unit: str) -> list[str]:
    """Format bucket labels."""
    labels: list[str] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        if np.isinf(upper):
            labels.append(f"{int(lower):,}+ {unit}")
        else:
            labels.append(f"{int(lower):,}-{int(upper):,} {unit}")
    return labels


def _format_year_bucket(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Format year bucket."""
    year_order = series.astype("Int64")
    return year_order.astype(str), year_order


def _build_bucket_series(series: pd.Series, bins: list[float], unit: str) -> tuple[pd.Series, pd.Series]:
    """Build bucket series."""
    labels = _format_bucket_labels(bins, unit)
    bucketed = pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=True,
    )
    codes = bucketed.cat.codes.replace(-1, np.nan).astype("Int64")
    return bucketed.astype(str), codes


def _bucket_midpoints(bins: list[float]) -> list[float]:
    """Bucket midpoints."""
    midpoints: list[float] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        if np.isinf(upper):
            midpoints.append(float(lower * 1.25))
        else:
            midpoints.append(float((lower + upper) / 2.0))
    return midpoints


def build_anonymised_comps_sample(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Build anonymised comparables sample."""
    year_bucket, year_bucket_order = _format_year_bucket(model_frame["transaction_year"])
    size_bucket, size_bucket_order = _build_bucket_series(model_frame["TOTAL SIZE (SQ. M.)"], SIZE_BUCKET_BINS, "sqm")
    price_bucket, price_bucket_order = _build_bucket_series(
        model_frame["actual_deal_size_eur_mn"],
        PRICE_BUCKET_BINS,
        "EUR mn",
    )
    ppsqm_bucket, ppsqm_bucket_order = _build_bucket_series(
        model_frame["actual_price_per_sqm_eur"],
        PRICE_PER_SQM_BUCKET_BINS,
        "EUR/sqm",
    )
    size_midpoints = {
        label: midpoint
        for label, midpoint in zip(_format_bucket_labels(SIZE_BUCKET_BINS, "sqm"), _bucket_midpoints(SIZE_BUCKET_BINS))
    }

    return pd.DataFrame(
        {
            "asset_type": model_frame["primary_asset_type"].astype(str),
            "country": model_frame["country_group"].astype(str),
            "year_bucket": year_bucket,
            "year_bucket_order": year_bucket_order,
            "size_bucket": size_bucket,
            "size_bucket_order": size_bucket_order,
            "size_bucket_mid_sqm": size_bucket.map(size_midpoints).astype(float),
            "price_bucket": price_bucket,
            "price_bucket_order": price_bucket_order,
            "price_per_sqm_bucket": ppsqm_bucket,
            "price_per_sqm_bucket_order": ppsqm_bucket_order,
        }
    )


def _other_europe_composition(model_frame: pd.DataFrame) -> dict[str, int]:
    """Other europe composition."""
    other_europe = model_frame.loc[model_frame["country_group"].astype(str).eq("Other Europe"), "country"]
    counts = other_europe.value_counts().sort_values(ascending=False)
    return {str(country): int(count) for country, count in counts.items()}


def _build_bootstrap_residual_pool(model, n_samples: int = BOOTSTRAP_RESIDUAL_COUNT) -> np.ndarray:
    """Build bootstrap residual pool."""
    rng = np.random.default_rng(RANDOM_SEED)
    residuals = np.asarray(model.resid, dtype=float)
    return rng.choice(residuals, size=n_samples, replace=True)


def _cross_validation_comparison(rolling_cv: dict[str, Any], random_cv: dict[str, Any]) -> dict[str, Any]:
    """Cross validation comparison."""
    mape_gap = abs(rolling_cv["mean_mape_pct"] - random_cv["mean_mape_pct"])
    return {
        "absolute_mean_mape_gap_pct_points": float(mape_gap),
        "substantial_divergence": bool(mape_gap >= 5.0),
        "criterion": "Absolute difference in mean MAPE >= 5 percentage points.",
    }


def _limitations(include_year_built: bool) -> list[str]:
    """Limitations the current helper."""
    base_limitations = [
        "United Kingdom transactions constitute 58 percent of the training sample, so the fitted coefficients are UK-dominated and country fixed effects mainly shift levels for smaller markets.",
        "Size elasticity is assumed homogeneous across countries and asset types; no interaction specification was tested in this version.",
        "Mixed Use and Niche are heterogeneous buckets in the Preqin taxonomy and are likely to contribute higher residual variance.",
        "The Other Europe dummy pools roughly 20 countries with small sample sizes.",
        "Commercial property level indices were not used in this version; a chained quarterly proxy was built from HICP/CPI inflation-rate series rebased to 2021Q1 = 100.",
    ]
    if include_year_built:
        base_limitations.append(
            "Year built is only partially observed after Capital IQ enrichment; any retained coefficient must be interpreted with that missing-data caveat."
        )
    else:
        base_limitations.append(
            "Year built was excluded from the model because the Capital IQ enrichment produced fewer than 150 confident matches."
        )
    return base_limitations


def _prepare_change_d_deployment_frame(dataset: pd.DataFrame, pipeline_metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare change d deployment frame."""
    frame = dataset.copy()
    frame = frame.loc[~frame["primary_asset_type"].astype(str).isin(["Land", "Niche"])].copy()

    lower_bound = float(frame["price_per_sqm_eur"].quantile(0.025))
    upper_bound = float(frame["price_per_sqm_eur"].quantile(0.975))
    frame["price_per_sqm_winsorized_eur"] = frame["price_per_sqm_eur"].clip(lower_bound, upper_bound)
    frame["deal_size_winsorized_eur_mn"] = (
        frame["price_per_sqm_winsorized_eur"] * frame["TOTAL SIZE (SQ. M.)"] / 1_000_000
    )
    frame["actual_deal_size_eur_mn"] = frame["deal_size_winsorized_eur_mn"]
    frame["actual_price_per_sqm_eur"] = frame["price_per_sqm_winsorized_eur"]
    frame["log_total_size_sqm"] = np.log(frame["TOTAL SIZE (SQ. M.)"])
    frame["log_deal_size_eur_mn"] = np.log(frame["deal_size_winsorized_eur_mn"])
    frame["model_year_effect"] = frame["transaction_year"].astype(int).clip(upper=2025)
    frame["primary_asset_type"] = pd.Categorical(
        frame["primary_asset_type"].astype(str),
        categories=[level for level in ASSET_TYPE_LEVELS if level not in {"Land", "Niche"}],
    )
    frame["country_group"] = pd.Categorical(frame["country_group"].astype(str), categories=COUNTRY_GROUP_LEVELS)
    frame = frame.loc[np.isfinite(frame["log_total_size_sqm"]) & np.isfinite(frame["log_deal_size_eur_mn"])].copy()
    frame = frame.reset_index(drop=True)

    prep_metadata = {
        "include_year_built": False,
        "rows_removed_for_missing_year_built": 0,
        "rows_available_for_model": int(len(frame)),
        "asset_type_levels": [level for level in ASSET_TYPE_LEVELS if level not in {"Land", "Niche"}],
        "country_group_levels": COUNTRY_GROUP_LEVELS,
        "winsorization": {
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "lower_bound_eur_per_sqm": lower_bound,
            "upper_bound_eur_per_sqm": upper_bound,
        },
        "specification_name": "change_d",
        "specification_description": "Change C plus size-by-asset-type interactions.",
        "model_formula": CHANGE_D_FORMULA,
    }
    return frame, prep_metadata


def _load_change_d_metrics() -> dict[str, Any]:
    """Load change d metrics."""
    if not CHANGE_D_METRICS_PATH.exists():
        raise FileNotFoundError(
            "The change D metrics artifact is missing. Run the stage-two refit diagnostics before exporting deployment artifacts."
        )
    with CHANGE_D_METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_reference_benchmarks(model_frame: pd.DataFrame) -> dict[str, Any]:
    """Build reference benchmarks."""
    benchmark_cells = (
        model_frame.assign(
            asset_type=model_frame["primary_asset_type"].astype(str),
            country=model_frame["country_group"].astype(str),
        )
        .groupby(["asset_type", "country"], observed=True)["actual_price_per_sqm_eur"]
        .agg(median_price_per_sqm_eur="median", sample_size="size")
        .reset_index()
        .sort_values(["asset_type", "country"])
    )
    records = [
        {
            "asset_type": str(row["asset_type"]),
            "country": str(row["country"]),
            "median_price_per_sqm_eur": float(row["median_price_per_sqm_eur"]),
            "sample_size": int(row["sample_size"]),
        }
        for _, row in benchmark_cells.iterrows()
    ]
    return {
        "definition": (
            "Median price per square metre in the deployed change D training sample for each asset-type by country-group cell. "
            "This is a reference benchmark, not a prediction."
        ),
        "cells": records,
        "global_median_price_per_sqm_eur": float(model_frame["actual_price_per_sqm_eur"].median()),
    }


def _build_methodology_note(change_d_metrics: dict[str, Any]) -> str:
    """Build methodology note."""
    rolling_folds = change_d_metrics["rolling_origin"]["folds"]
    fold_bits = [
        f"{fold['test_year']}: {fold['model_metrics']['mape_pct']:.1f}%"
        for fold in rolling_folds
    ]
    return (
        "A hedonic point-valuation specification was evaluated and rejected for deployment because it did not beat the "
        "naive country-and-type median benchmark out of sample. Rolling-origin MAPE by fold was "
        + ", ".join(fold_bits)
        + f" (mean {change_d_metrics['rolling_origin']['mean_mape_pct']:.1f}%). "
        + f"The naive benchmark recorded {change_d_metrics['rolling_origin']['headline_fold']['baseline_metrics']['mape_pct']:.1f}% "
        "MAPE on the 2026 headline fold."
    )


def _build_retrieval_metadata(
    model_frame: pd.DataFrame,
    pipeline_metadata: dict[str, Any],
    prep_metadata: dict[str, Any],
    change_d_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build retrieval metadata."""
    reference_benchmarks = _build_reference_benchmarks(model_frame)
    rolling_origin = change_d_metrics["rolling_origin"]
    random_5_fold = change_d_metrics["random_5_fold"]
    headline_fold = rolling_origin["headline_fold"]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_framing": {
            "tool_type": "Comparable retrieval tool",
            "deployment_decision": "Point valuation evaluated and rejected on out-of-sample grounds.",
            "selected_stage_two_specification": "change_d",
            "selected_stage_two_metrics_source": str(CHANGE_D_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "deployed_artifacts": ["model/artifacts/comps_sample.parquet", "model/artifacts/metadata.json"],
        },
        "model_formula_evaluated": prep_metadata["model_formula"],
        "features_evaluated": [
            "log_total_size_sqm",
            "primary_asset_type",
            "country_group",
            "model_year_effect",
            "log_total_size_sqm x primary_asset_type",
        ],
        "asset_type_levels": prep_metadata["asset_type_levels"],
        "country_group_levels": prep_metadata["country_group_levels"],
        "training_sample_size": int(len(model_frame)),
        "country_group_counts": {
            str(key): int(value)
            for key, value in model_frame["country_group"].value_counts(dropna=False).sort_index().items()
        },
        "asset_type_counts": {
            str(key): int(value)
            for key, value in model_frame["primary_asset_type"].value_counts(dropna=False).sort_index().items()
        },
        "other_europe_bucket_composition": _other_europe_composition(model_frame),
        "winsorization": prep_metadata["winsorization"],
        "year_built": {
            **pipeline_metadata["year_built"],
            "included_in_evaluated_model": False,
        },
        "retrieval_scoring": {
            "description": (
                "Comparables are ranked by a transparent similarity score combining log-size proximity with bonuses for "
                "same asset type, same country group, and same transaction year."
            ),
            "weights": {
                "log_size_similarity_scale": 100.0,
                "log_size_similarity_penalty_multiplier": 4.0,
                "same_asset_type_bonus": 70.0,
                "same_country_bonus": 20.0,
                "same_year_bonus": 10.0,
            },
        },
        "reference_benchmark": reference_benchmarks,
        "valuation_evaluation": {
            "decision": "Rejected for deployment as a point valuation tool.",
            "reason": (
                "The evaluated hedonic specification did not beat the naive country-and-type median benchmark on "
                "rolling-origin out-of-sample prediction."
            ),
            "rolling_origin": {
                "fold_mapes_pct": [
                    {
                        "test_year": int(fold["test_year"]),
                        "model_mape_pct": float(fold["model_metrics"]["mape_pct"]),
                        "baseline_mape_pct": float(fold["baseline_metrics"]["mape_pct"]),
                    }
                    for fold in rolling_origin["folds"]
                ],
                "mean_mape_pct": float(rolling_origin["mean_mape_pct"]),
                "std_mape_pct": float(rolling_origin["std_mape_pct"]),
                "headline_fold_test_year": int(headline_fold["test_year"]),
                "headline_fold_mape_pct": float(headline_fold["model_metrics"]["mape_pct"]),
                "headline_fold_baseline_mape_pct": float(headline_fold["baseline_metrics"]["mape_pct"]),
            },
            "random_5_fold": {
                "mean_mape_pct": float(random_5_fold["mean_mape_pct"]),
                "std_mape_pct": float(random_5_fold["std_mape_pct"]),
                "baseline_mean_mape_pct": float(random_5_fold["baseline_mean_mape_pct"]),
            },
            "final_model_fit": {
                "rsquared": float(change_d_metrics["rsquared"]),
                "rsquared_adj": float(change_d_metrics["rsquared_adj"]),
            },
        },
        "methodology_note": _build_methodology_note(change_d_metrics),
        "limitations": [
            "This deployed application is a comparable-retrieval tool, not an automated valuation model.",
            "United Kingdom transactions remain a large share of the deployed sample, so the comparable universe is still UK-heavy.",
            "Mixed Use remains a heterogeneous Preqin bucket and will widen dispersion in the retrieved sample.",
            "The Other Europe country group pools many small-sample markets into one bucket.",
            "Year built was excluded because the Capital IQ enrichment produced fewer than 150 confident matches.",
        ],
    }


def _export_retrieval_artifacts(comps_sample: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Export retrieval artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comps_sample.to_parquet(COMPS_SAMPLE_PATH, index=False)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    MODEL_PATH.unlink(missing_ok=True)
    RESIDUALS_PATH.unlink(missing_ok=True)


def prepare_change_d_analysis_frame() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Prepare change d analysis frame."""
    dataset, pipeline_metadata = build_training_frame()
    model_frame, prep_metadata = _prepare_change_d_deployment_frame(dataset, pipeline_metadata)
    return model_frame, pipeline_metadata, prep_metadata


def _build_metadata(
    model_frame: pd.DataFrame,
    pipeline_metadata: dict[str, Any],
    prep_metadata: dict[str, Any],
    formula: str,
    final_model,
    rolling_cv: dict[str, Any],
    random_cv: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_formula": formula,
        "target_variable": "log_deal_size_eur_mn",
        "features": [
            "log_total_size_sqm",
            "primary_asset_type",
            "country_group",
            "log_index_value",
        ]
        + (["year_built"] if prep_metadata["include_year_built"] else []),
        "asset_type_levels": ASSET_TYPE_LEVELS,
        "country_group_levels": COUNTRY_GROUP_LEVELS,
        "training_sample_size": int(len(model_frame)),
        "country_group_counts": pipeline_metadata["country_group_counts"],
        "asset_type_counts": pipeline_metadata["asset_type_counts"],
        "other_europe_bucket_composition": _other_europe_composition(model_frame),
        "winsorization": pipeline_metadata["winsorization"],
        "year_built": {
            **pipeline_metadata["year_built"],
            "included_in_model": prep_metadata["include_year_built"],
            "rows_removed_for_missing_year_built": prep_metadata["rows_removed_for_missing_year_built"],
        },
        "index_inputs": {
            "country_series": pipeline_metadata["index_sources"],
            "proxy_note": pipeline_metadata["macro_proxy_note"],
            "residential_substitution_used": False,
        },
        "validation": {
            "rolling_origin": rolling_cv,
            "random_5_fold": random_cv,
            "comparison": _cross_validation_comparison(rolling_cv, random_cv),
        },
        "final_model_fit": {
            "nobs": int(final_model.nobs),
            "rsquared": float(final_model.rsquared),
            "rsquared_adj": float(final_model.rsquared_adj),
            "aic": float(final_model.aic),
            "bic": float(final_model.bic),
        },
        "prediction_interval": {
            "method": "Residual bootstrap on the log scale using 1,000 resampled residuals from the final fitted model.",
            "residual_pool_size": BOOTSTRAP_RESIDUAL_COUNT,
            "lower_percentile": 5,
            "upper_percentile": 95,
        },
        "limitations": _limitations(prep_metadata["include_year_built"]),
    }


def _export_artifacts(
    final_model,
    residual_pool: np.ndarray,
    comps_sample: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Export artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_model, handle)
    np.save(RESIDUALS_PATH, residual_pool)
    comps_sample.to_parquet(COMPS_SAMPLE_PATH, index=False)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_and_export_artifacts() -> TrainingOutputs:
    """Train and export artifacts."""
    dataset, pipeline_metadata = build_training_frame()
    model_frame, prep_metadata = _prepare_change_d_deployment_frame(dataset, pipeline_metadata)
    change_d_metrics = _load_change_d_metrics()
    comps_sample = build_anonymised_comps_sample(model_frame)
    metadata = _build_retrieval_metadata(
        model_frame=model_frame,
        pipeline_metadata=pipeline_metadata,
        prep_metadata=prep_metadata,
        change_d_metrics=change_d_metrics,
    )
    _export_retrieval_artifacts(comps_sample, metadata)
    return TrainingOutputs(model=None, model_frame=model_frame, metadata=metadata)


def main() -> None:
    """Run the module entry point."""
    outputs = train_and_export_artifacts()
    rolling = outputs.metadata["valuation_evaluation"]["rolling_origin"]
    print(f"Training rows: {outputs.metadata['training_sample_size']}")
    print(f"Deployment framing: {outputs.metadata['app_framing']['tool_type']}")
    print(f"Evaluated formula: {outputs.metadata['model_formula_evaluated']}")
    print(f"Rolling mean MAPE: {rolling['mean_mape_pct']:.2f}%")
    print(f"Headline fold MAPE: {rolling['headline_fold_mape_pct']:.2f}%")
    print(f"Headline fold naive baseline MAPE: {rolling['headline_fold_baseline_mape_pct']:.2f}%")
    print(f"Artifacts written to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
