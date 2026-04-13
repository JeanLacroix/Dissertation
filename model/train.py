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

BASE_FORMULA = (
    "log_deal_size_eur_mn ~ log_total_size_sqm + C(primary_asset_type) + "
    "C(country_group) + log_index_value"
)
BOOTSTRAP_RESIDUAL_COUNT = 1_000
RANDOM_SEED = 42

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
    model: Any
    model_frame: pd.DataFrame
    metadata: dict[str, Any]


def _build_formula(include_year_built: bool) -> str:
    if include_year_built:
        return f"{BASE_FORMULA} + year_built"
    return BASE_FORMULA


def _prepare_model_frame(dataset: pd.DataFrame, pipeline_metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    return smf.ols(formula=formula, data=train_frame).fit()


def _predict_deal_size_eur_mn(model, frame: pd.DataFrame) -> np.ndarray:
    log_predictions = model.predict(frame)
    return np.exp(np.asarray(log_predictions, dtype=float))


def _evaluate_predictions(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    percentage_errors = np.abs((actual_array - predicted_array) / actual_array)
    rmse = float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))
    return {
        "mape_pct": float(np.mean(percentage_errors) * 100.0),
        "rmse_eur_mn": rmse,
    }


def _baseline_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
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
    return {
        "mape_reduction_pct": float(
            ((baseline_metrics["mape_pct"] - model_metrics["mape_pct"]) / baseline_metrics["mape_pct"]) * 100.0
        ),
        "rmse_reduction_pct": float(
            ((baseline_metrics["rmse_eur_mn"] - model_metrics["rmse_eur_mn"]) / baseline_metrics["rmse_eur_mn"]) * 100.0
        ),
    }


def _run_fold(train_frame: pd.DataFrame, test_frame: pd.DataFrame, formula: str, fold_name: str) -> dict[str, Any]:
    model = _fit_ols(train_frame, formula)
    model_predictions = _predict_deal_size_eur_mn(model, test_frame)
    baseline_predictions = _baseline_predict(train_frame, test_frame)

    model_metrics = _evaluate_predictions(test_frame["actual_deal_size_eur_mn"], model_predictions)
    baseline_metrics = _evaluate_predictions(test_frame["actual_deal_size_eur_mn"], baseline_predictions)

    return {
        "fold": fold_name,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "lift_vs_baseline": _compute_lift(model_metrics, baseline_metrics),
    }


def run_rolling_origin_cv(model_frame: pd.DataFrame, formula: str) -> dict[str, Any]:
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
    labels: list[str] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        if np.isinf(upper):
            labels.append(f"{int(lower):,}+ {unit}")
        else:
            labels.append(f"{int(lower):,}-{int(upper):,} {unit}")
    return labels


def _build_bucket_series(series: pd.Series, bins: list[float], unit: str) -> tuple[pd.Series, pd.Series]:
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


def build_anonymised_comps_sample(model_frame: pd.DataFrame) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            "asset_type": model_frame["primary_asset_type"].astype(str),
            "country": model_frame["country_group"].astype(str),
            "transaction_year": model_frame["transaction_year"].astype("Int64"),
            "size_bucket": size_bucket,
            "size_bucket_order": size_bucket_order,
            "price_bucket": price_bucket,
            "price_bucket_order": price_bucket_order,
            "price_per_sqm_bucket": ppsqm_bucket,
            "price_per_sqm_bucket_order": ppsqm_bucket_order,
        }
    )


def _other_europe_composition(model_frame: pd.DataFrame) -> dict[str, int]:
    other_europe = model_frame.loc[model_frame["country_group"].astype(str).eq("Other Europe"), "country"]
    counts = other_europe.value_counts().sort_values(ascending=False)
    return {str(country): int(count) for country, count in counts.items()}


def _build_bootstrap_residual_pool(model, n_samples: int = BOOTSTRAP_RESIDUAL_COUNT) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    residuals = np.asarray(model.resid, dtype=float)
    return rng.choice(residuals, size=n_samples, replace=True)


def _cross_validation_comparison(rolling_cv: dict[str, Any], random_cv: dict[str, Any]) -> dict[str, Any]:
    mape_gap = abs(rolling_cv["mean_mape_pct"] - random_cv["mean_mape_pct"])
    return {
        "absolute_mean_mape_gap_pct_points": float(mape_gap),
        "substantial_divergence": bool(mape_gap >= 5.0),
        "criterion": "Absolute difference in mean MAPE >= 5 percentage points.",
    }


def _limitations(include_year_built: bool) -> list[str]:
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


def _build_metadata(
    model_frame: pd.DataFrame,
    pipeline_metadata: dict[str, Any],
    prep_metadata: dict[str, Any],
    formula: str,
    final_model,
    rolling_cv: dict[str, Any],
    random_cv: dict[str, Any],
) -> dict[str, Any]:
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_model, handle)
    np.save(RESIDUALS_PATH, residual_pool)
    comps_sample.to_parquet(COMPS_SAMPLE_PATH, index=False)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_and_export_artifacts() -> TrainingOutputs:
    dataset, pipeline_metadata = build_training_frame()
    model_frame, prep_metadata = _prepare_model_frame(dataset, pipeline_metadata)
    formula = _build_formula(include_year_built=prep_metadata["include_year_built"])

    rolling_cv = run_rolling_origin_cv(model_frame, formula)
    random_cv = run_random_cv(model_frame, formula)
    final_model = _fit_ols(model_frame, formula)
    residual_pool = _build_bootstrap_residual_pool(final_model)
    comps_sample = build_anonymised_comps_sample(model_frame)
    metadata = _build_metadata(
        model_frame=model_frame,
        pipeline_metadata=pipeline_metadata,
        prep_metadata=prep_metadata,
        formula=formula,
        final_model=final_model,
        rolling_cv=rolling_cv,
        random_cv=random_cv,
    )
    _export_artifacts(final_model, residual_pool, comps_sample, metadata)
    return TrainingOutputs(model=final_model, model_frame=model_frame, metadata=metadata)


def main() -> None:
    outputs = train_and_export_artifacts()
    rolling = outputs.metadata["validation"]["rolling_origin"]
    headline = rolling["headline_fold"]
    print(f"Training rows: {outputs.metadata['training_sample_size']}")
    print(f"Formula: {outputs.metadata['model_formula']}")
    print(f"Rolling mean MAPE: {rolling['mean_mape_pct']:.2f}%")
    print(f"Rolling mean RMSE: {rolling['mean_rmse_eur_mn']:.2f} EUR mn")
    print(f"Headline fold MAPE: {headline['model_metrics']['mape_pct']:.2f}%")
    print(f"Headline fold RMSE: {headline['model_metrics']['rmse_eur_mn']:.2f} EUR mn")
    print(f"Artifacts written to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
