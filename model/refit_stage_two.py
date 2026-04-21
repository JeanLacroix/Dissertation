from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import ASSET_TYPE_LEVELS, COUNTRY_GROUP_LEVELS, build_training_frame
from .train import (
    ARTIFACTS_DIR,
    BOOTSTRAP_RESIDUAL_COUNT,
    COMPS_SAMPLE_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    RESIDUALS_PATH,
    _baseline_predict,
    _build_bootstrap_residual_pool,
    _compute_lift,
    _cross_validation_comparison,
    _evaluate_predictions,
    _export_artifacts,
    _fit_ols,
    _predict_deal_size_eur_mn,
    build_anonymised_comps_sample,
)

REFIT_DIAGNOSTICS_DIR = ARTIFACTS_DIR / "stage_two_refits"
YEAR_FE_CAP = 2025
YEAR_FE_LEVELS = [2021, 2022, 2023, 2024, 2025]

ROLLING_FOLD_SPECS = [
    ("fold_1", [2021, 2022], 2023),
    ("fold_2", [2021, 2022, 2023], 2024),
    ("fold_3", [2021, 2022, 2023, 2024], 2025),
    ("fold_4", [2021, 2022, 2023, 2024, 2025], 2026),
]


@dataclass(frozen=True)
class RefitSpec:
    name: str
    description: str
    exclude_asset_types: tuple[str, ...] = ()
    exclude_country_groups: tuple[str, ...] = ()
    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    temporal_mode: str = "index"
    use_size_asset_interactions: bool = False
    allow_year_built: bool = True


@dataclass(frozen=True)
class RefitResult:
    spec: RefitSpec
    model: Any
    model_frame: pd.DataFrame
    formula: str
    coefficient_table: pd.DataFrame
    rolling_origin: dict[str, Any]
    random_5_fold: dict[str, Any]
    rsquared: float
    rsquared_adj: float
    prep_metadata: dict[str, Any]
    pipeline_metadata: dict[str, Any]
    implausibility_notes: list[str]


BASELINE_SPEC = RefitSpec(
    name="baseline",
    description="Original stage-two specification with 1st/99th winsorisation and log(index_value).",
)

CHANGE_A_SPEC = RefitSpec(
    name="change_a",
    description="Drop Land and Niche observations from the sample.",
    exclude_asset_types=("Land", "Niche"),
)

CHANGE_B_SPEC = RefitSpec(
    name="change_b",
    description="Change A plus 2.5th/97.5th percentile winsorisation.",
    exclude_asset_types=("Land", "Niche"),
    winsor_lower_quantile=0.025,
    winsor_upper_quantile=0.975,
)

CHANGE_C_SPEC = RefitSpec(
    name="change_c",
    description="Change B plus year fixed effects instead of log(index_value).",
    exclude_asset_types=("Land", "Niche"),
    winsor_lower_quantile=0.025,
    winsor_upper_quantile=0.975,
    temporal_mode="year_fe",
)

CHANGE_D_SPEC = RefitSpec(
    name="change_d",
    description="Change C plus size-by-asset-type interactions.",
    exclude_asset_types=("Land", "Niche"),
    winsor_lower_quantile=0.025,
    winsor_upper_quantile=0.975,
    temporal_mode="year_fe",
    use_size_asset_interactions=True,
)

CHANGE_E_SPEC = RefitSpec(
    name="change_e",
    description="Sensitivity run using change D on the non-UK sample.",
    exclude_asset_types=("Land", "Niche"),
    exclude_country_groups=("United Kingdom",),
    winsor_lower_quantile=0.025,
    winsor_upper_quantile=0.975,
    temporal_mode="year_fe",
    use_size_asset_interactions=True,
)


def _active_levels(all_levels: list[str], excluded_levels: tuple[str, ...]) -> list[str]:
    excluded = set(excluded_levels)
    return [level for level in all_levels if level not in excluded]


def _build_formula(spec: RefitSpec, include_year_built: bool) -> str:
    if spec.use_size_asset_interactions:
        size_and_asset = "log_total_size_sqm * C(primary_asset_type)"
    else:
        size_and_asset = "log_total_size_sqm + C(primary_asset_type)"

    rhs = [size_and_asset, "C(country_group)"]
    if spec.temporal_mode == "index":
        rhs.append("log_index_value")
    elif spec.temporal_mode == "year_fe":
        rhs.append("C(model_year_effect)")
    else:
        raise ValueError(f"Unsupported temporal_mode: {spec.temporal_mode}")

    if include_year_built:
        rhs.append("year_built")
    return "log_deal_size_eur_mn ~ " + " + ".join(rhs)


def _apply_spec_filters(
    dataset: pd.DataFrame,
    pipeline_metadata: dict[str, Any],
    spec: RefitSpec,
) -> tuple[pd.DataFrame, list[str], list[str], bool, int]:
    frame = dataset.copy()
    if spec.exclude_asset_types:
        frame = frame.loc[~frame["primary_asset_type"].astype(str).isin(spec.exclude_asset_types)].copy()
    if spec.exclude_country_groups:
        frame = frame.loc[~frame["country_group"].astype(str).isin(spec.exclude_country_groups)].copy()

    frame["log_total_size_sqm"] = np.log(frame["TOTAL SIZE (SQ. M.)"])
    frame = frame.loc[np.isfinite(frame["log_total_size_sqm"])].copy()

    include_year_built = bool(pipeline_metadata["year_built"]["year_built_enabled"] and spec.allow_year_built)
    rows_removed_for_missing_year_built = 0
    if include_year_built:
        rows_removed_for_missing_year_built = int(frame["year_built"].isna().sum())
        frame = frame.dropna(subset=["year_built"]).copy()

    active_asset_levels = _active_levels(ASSET_TYPE_LEVELS, spec.exclude_asset_types)
    active_country_levels = _active_levels(COUNTRY_GROUP_LEVELS, spec.exclude_country_groups)
    frame["primary_asset_type"] = pd.Categorical(frame["primary_asset_type"].astype(str), categories=active_asset_levels)
    frame["country_group"] = pd.Categorical(frame["country_group"].astype(str), categories=active_country_levels)
    frame = frame.reset_index(drop=True)
    return frame, active_asset_levels, active_country_levels, include_year_built, rows_removed_for_missing_year_built


def _apply_winsor_bounds(frame: pd.DataFrame, lower_bound: float, upper_bound: float) -> pd.DataFrame:
    out = frame.copy()
    out["price_per_sqm_winsorized_eur"] = out["price_per_sqm_eur"].clip(lower_bound, upper_bound)
    out["deal_size_winsorized_eur_mn"] = out["price_per_sqm_winsorized_eur"] * out["TOTAL SIZE (SQ. M.)"] / 1_000_000
    out["actual_deal_size_eur_mn"] = out["deal_size_winsorized_eur_mn"]
    out["actual_price_per_sqm_eur"] = out["price_per_sqm_winsorized_eur"]
    out["log_deal_size_eur_mn"] = np.log(out["deal_size_winsorized_eur_mn"])
    out = out.loc[np.isfinite(out["log_deal_size_eur_mn"])].copy().reset_index(drop=True)
    return out


def _winsor_bounds_from(frame: pd.DataFrame, spec: RefitSpec) -> tuple[float, float]:
    lower_bound = float(frame["price_per_sqm_eur"].quantile(spec.winsor_lower_quantile))
    upper_bound = float(frame["price_per_sqm_eur"].quantile(spec.winsor_upper_quantile))
    return lower_bound, upper_bound


def _apply_spec_frame(dataset: pd.DataFrame, pipeline_metadata: dict[str, Any], spec: RefitSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    filtered, asset_levels, country_levels, include_year_built, rows_removed_for_missing_year_built = _apply_spec_filters(
        dataset, pipeline_metadata, spec
    )
    lower_bound, upper_bound = _winsor_bounds_from(filtered, spec)
    frame = _apply_winsor_bounds(filtered, lower_bound, upper_bound)
    frame["primary_asset_type"] = pd.Categorical(frame["primary_asset_type"].astype(str), categories=asset_levels)
    frame["country_group"] = pd.Categorical(frame["country_group"].astype(str), categories=country_levels)
    return frame, {
        "spec": asdict(spec),
        "include_year_built": include_year_built,
        "rows_removed_for_missing_year_built": rows_removed_for_missing_year_built,
        "winsorization": {
            "lower_quantile": spec.winsor_lower_quantile,
            "upper_quantile": spec.winsor_upper_quantile,
            "lower_bound_eur_per_sqm": lower_bound,
            "upper_bound_eur_per_sqm": upper_bound,
        },
        "asset_type_levels": asset_levels,
        "country_group_levels": country_levels,
    }


def _map_prediction_year_effect(year_value: int, available_years: list[int]) -> int:
    capped_year = min(int(year_value), YEAR_FE_CAP)
    eligible_years = [year for year in available_years if year <= capped_year]
    if eligible_years:
        return max(eligible_years)
    return min(available_years)


def _prepare_frames_for_fit(train_frame: pd.DataFrame, test_frame: pd.DataFrame, spec: RefitSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_frame.copy()
    test = test_frame.copy()
    if spec.temporal_mode == "year_fe":
        train["model_year_effect"] = train["transaction_year"].astype(int).clip(upper=YEAR_FE_CAP)
        available_years = sorted(train["model_year_effect"].dropna().astype(int).unique().tolist())
        test["model_year_effect"] = test["transaction_year"].astype(int).map(
            lambda year_value: _map_prediction_year_effect(year_value, available_years)
        )
        train["model_year_effect"] = pd.Categorical(train["model_year_effect"], categories=available_years, ordered=True)
        test["model_year_effect"] = pd.Categorical(test["model_year_effect"], categories=available_years, ordered=True)
    return train, test


def _prepare_full_fit_frame(model_frame: pd.DataFrame, spec: RefitSpec) -> pd.DataFrame:
    frame = model_frame.copy()
    if spec.temporal_mode == "year_fe":
        frame["model_year_effect"] = frame["transaction_year"].astype(int).clip(upper=YEAR_FE_CAP)
        frame["model_year_effect"] = pd.Categorical(frame["model_year_effect"], categories=YEAR_FE_LEVELS, ordered=True)
    return frame


def _run_fold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    spec: RefitSpec,
    formula: str,
    fold_name: str,
    fold_aware_winsor: bool = True,
) -> dict[str, Any]:
    if fold_aware_winsor:
        lower_bound, upper_bound = _winsor_bounds_from(train_frame, spec)
        train_win = _apply_winsor_bounds(train_frame, lower_bound, upper_bound)
        test_win = _apply_winsor_bounds(test_frame, lower_bound, upper_bound)
    else:
        train_win = train_frame
        test_win = test_frame
    prepared_train, prepared_test = _prepare_frames_for_fit(train_win, test_win, spec)
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


def _run_rolling_origin_cv(
    model_frame: pd.DataFrame,
    spec: RefitSpec,
    formula: str,
    fold_aware_winsor: bool = True,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold_name, train_years, test_year in ROLLING_FOLD_SPECS:
        train_frame = model_frame.loc[model_frame["transaction_year"].isin(train_years)].copy()
        test_frame = model_frame.loc[model_frame["transaction_year"].eq(test_year)].copy()
        if train_frame.empty or test_frame.empty:
            continue
        folds.append(_run_fold(train_frame, test_frame, spec, formula, fold_name, fold_aware_winsor=fold_aware_winsor))

    mapes = [fold["model_metrics"]["mape_pct"] for fold in folds]
    rmses = [fold["model_metrics"]["rmse_eur_mn"] for fold in folds]
    baseline_mapes = [fold["baseline_metrics"]["mape_pct"] for fold in folds]
    baseline_rmses = [fold["baseline_metrics"]["rmse_eur_mn"] for fold in folds]
    return {
        "strategy": "rolling_origin_expanding_window",
        "folds": folds,
        "mean_mape_pct": float(np.mean(mapes)),
        "std_mape_pct": float(np.std(mapes, ddof=0)),
        "mean_rmse_eur_mn": float(np.mean(rmses)),
        "std_rmse_eur_mn": float(np.std(rmses, ddof=0)),
        "baseline_mean_mape_pct": float(np.mean(baseline_mapes)),
        "baseline_mean_rmse_eur_mn": float(np.mean(baseline_rmses)),
        "headline_fold": folds[-1] if folds else None,
    }


def _run_random_5_fold_cv(
    model_frame: pd.DataFrame,
    spec: RefitSpec,
    formula: str,
    fold_aware_winsor: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    order = np.arange(len(model_frame))
    rng.shuffle(order)
    folds = np.array_split(order, 5)
    fold_results: list[dict[str, Any]] = []

    for fold_index, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(order, test_idx, assume_unique=False)
        train_frame = model_frame.iloc[train_idx].copy()
        test_frame = model_frame.iloc[test_idx].copy()
        fold_results.append(
            _run_fold(train_frame, test_frame, spec, formula, f"random_fold_{fold_index}", fold_aware_winsor=fold_aware_winsor)
        )

    mapes = [fold["model_metrics"]["mape_pct"] for fold in fold_results]
    rmses = [fold["model_metrics"]["rmse_eur_mn"] for fold in fold_results]
    baseline_mapes = [fold["baseline_metrics"]["mape_pct"] for fold in fold_results]
    baseline_rmses = [fold["baseline_metrics"]["rmse_eur_mn"] for fold in fold_results]
    return {
        "strategy": "random_5_fold",
        "folds": fold_results,
        "mean_mape_pct": float(np.mean(mapes)),
        "std_mape_pct": float(np.std(mapes, ddof=0)),
        "mean_rmse_eur_mn": float(np.mean(rmses)),
        "std_rmse_eur_mn": float(np.std(rmses, ddof=0)),
        "baseline_mean_mape_pct": float(np.mean(baseline_mapes)),
        "baseline_mean_rmse_eur_mn": float(np.mean(baseline_rmses)),
    }


def _coefficient_table(model) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "p_value": model.pvalues.values,
        }
    )


def _size_slopes_by_asset_type(coefficient_table: pd.DataFrame, asset_levels: list[str]) -> dict[str, float]:
    lookup = coefficient_table.set_index("term")["coef"].to_dict()
    base_slope = float(lookup.get("log_total_size_sqm", np.nan))
    reference_asset = asset_levels[0] if asset_levels else None
    slopes: dict[str, float] = {}
    for asset in asset_levels:
        interaction_term = f"log_total_size_sqm:C(primary_asset_type)[T.{asset}]"
        interaction_coef = float(lookup.get(interaction_term, 0.0))
        if asset == reference_asset:
            interaction_coef = 0.0
        slopes[asset] = base_slope + interaction_coef
    return slopes


def _implausibility_notes(
    spec: RefitSpec,
    coefficient_table: pd.DataFrame,
    asset_levels: list[str],
    country_levels: list[str],
) -> list[str]:
    notes: list[str] = []
    table = coefficient_table.set_index("term")

    if "log_index_value" in table.index:
        coef = float(table.loc["log_index_value", "coef"])
        p_value = float(table.loc["log_index_value", "p_value"])
        if coef < 0.3 or coef > 2.0:
            notes.append(f"log(index_value) coefficient {coef:.3f} lies outside the 0.3 to 2.0 sanity range.")
        if p_value >= 0.05:
            notes.append(f"log(index_value) is statistically insignificant (p={p_value:.3f}).")

    if spec.use_size_asset_interactions:
        slopes = _size_slopes_by_asset_type(coefficient_table, asset_levels)
        bad_slopes = {asset: slope for asset, slope in slopes.items() if slope <= 0.0}
        if bad_slopes:
            notes.append(f"Non-positive size slopes detected: {bad_slopes}.")
    elif "log_total_size_sqm" in table.index:
        size_coef = float(table.loc["log_total_size_sqm", "coef"])
        if size_coef <= 0.0 or size_coef < 0.7 or size_coef > 1.0:
            notes.append(f"log(total_size_sqm) coefficient {size_coef:.3f} lies outside the expected 0.7 to 1.0 range.")

    if "United Kingdom" in country_levels:
        bad_country_effects = {
            term: float(table.loc[term, "coef"])
            for term in table.index
            if term.startswith("C(country_group)") and (float(table.loc[term, "coef"]) < -1.0 or float(table.loc[term, "coef"]) > 0.5)
        }
        if bad_country_effects:
            notes.append(f"Country fixed effects outside the expected -1.0 to +0.5 range: {bad_country_effects}.")
    return notes


def evaluate_spec(
    dataset: pd.DataFrame,
    pipeline_metadata: dict[str, Any],
    spec: RefitSpec,
    fold_aware_winsor: bool = True,
) -> RefitResult:
    model_frame, prep_metadata = _apply_spec_frame(dataset, pipeline_metadata, spec)
    formula = _build_formula(spec, include_year_built=prep_metadata["include_year_built"])
    if fold_aware_winsor:
        filtered_frame, _, _, _, _ = _apply_spec_filters(dataset, pipeline_metadata, spec)
        cv_frame = filtered_frame
    else:
        cv_frame = model_frame
    rolling_origin = _run_rolling_origin_cv(cv_frame, spec, formula, fold_aware_winsor=fold_aware_winsor)
    random_5_fold = _run_random_5_fold_cv(cv_frame, spec, formula, fold_aware_winsor=fold_aware_winsor)
    final_fit_frame = _prepare_full_fit_frame(model_frame, spec)
    final_model = _fit_ols(final_fit_frame, formula)
    coefficient_table = _coefficient_table(final_model)
    implausibility_notes = _implausibility_notes(
        spec,
        coefficient_table,
        prep_metadata["asset_type_levels"],
        prep_metadata["country_group_levels"],
    )
    return RefitResult(
        spec=spec,
        model=final_model,
        model_frame=model_frame,
        formula=formula,
        coefficient_table=coefficient_table,
        rolling_origin=rolling_origin,
        random_5_fold=random_5_fold,
        rsquared=float(final_model.rsquared),
        rsquared_adj=float(final_model.rsquared_adj),
        prep_metadata=prep_metadata,
        pipeline_metadata=pipeline_metadata,
        implausibility_notes=implausibility_notes,
    )


def _sign_flips(current_table: pd.DataFrame, previous_table: pd.DataFrame, compare_country_terms: bool = True) -> list[str]:
    current = current_table.set_index("term")["coef"]
    previous = previous_table.set_index("term")["coef"]
    flips: list[str] = []
    for term in sorted(set(current.index) & set(previous.index)):
        if not compare_country_terms and term.startswith("C(country_group)"):
            continue
        current_coef = float(current.loc[term])
        previous_coef = float(previous.loc[term])
        if np.isclose(current_coef, 0.0) or np.isclose(previous_coef, 0.0):
            continue
        if np.sign(current_coef) != np.sign(previous_coef):
            flips.append(term)
    return flips


def _other_europe_composition(model_frame: pd.DataFrame) -> dict[str, int]:
    other_europe = model_frame.loc[model_frame["country_group"].astype(str).eq("Other Europe"), "country"]
    counts = other_europe.value_counts().sort_values(ascending=False)
    return {str(country): int(count) for country, count in counts.items()}


def _build_final_metadata(result: RefitResult) -> dict[str, Any]:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specification_name": result.spec.name,
        "specification_description": result.spec.description,
        "specification_config": result.prep_metadata["spec"],
        "model_formula": result.formula,
        "features": [
            "log_total_size_sqm",
            "primary_asset_type",
            "country_group",
        ]
        + (["log_index_value"] if result.spec.temporal_mode == "index" else ["model_year_effect"])
        + (["year_built"] if result.prep_metadata["include_year_built"] else []),
        "asset_type_levels": result.prep_metadata["asset_type_levels"],
        "country_group_levels": result.prep_metadata["country_group_levels"],
        "training_sample_size": int(len(result.model_frame)),
        "country_group_counts": {
            str(key): int(value)
            for key, value in result.model_frame["country_group"].value_counts(dropna=False).sort_index().items()
        },
        "asset_type_counts": {
            str(key): int(value)
            for key, value in result.model_frame["primary_asset_type"].value_counts(dropna=False).sort_index().items()
        },
        "other_europe_bucket_composition": _other_europe_composition(result.model_frame),
        "winsorization": result.prep_metadata["winsorization"],
        "year_built": {
            **result.pipeline_metadata["year_built"],
            "included_in_model": result.prep_metadata["include_year_built"],
            "rows_removed_for_missing_year_built": result.prep_metadata["rows_removed_for_missing_year_built"],
        },
        "index_inputs": {
            "country_series": result.pipeline_metadata["index_sources"],
            "proxy_note": result.pipeline_metadata["macro_proxy_note"],
            "included_in_model": result.spec.temporal_mode == "index",
            "residential_substitution_used": False,
        },
        "validation": {
            "rolling_origin": result.rolling_origin,
            "random_5_fold": result.random_5_fold,
            "comparison": _cross_validation_comparison(result.rolling_origin, result.random_5_fold),
        },
        "final_model_fit": {
            "nobs": int(result.model.nobs),
            "rsquared": result.rsquared,
            "rsquared_adj": result.rsquared_adj,
            "aic": float(result.model.aic),
            "bic": float(result.model.bic),
        },
        "prediction_interval": {
            "method": "Residual bootstrap on the log scale using 1,000 resampled residuals from the final fitted model.",
            "residual_pool_size": BOOTSTRAP_RESIDUAL_COUNT,
            "lower_percentile": 5,
            "upper_percentile": 95,
        },
        "implausibility_notes": result.implausibility_notes,
        "limitations": [
            "United Kingdom transactions constitute 58 percent of the original filtered sample, so coefficients can be UK-dominated unless the UK is explicitly excluded.",
            "Mixed Use remains a heterogeneous bucket in the Preqin taxonomy and is likely to contribute higher residual variance.",
            "The Other Europe dummy pools roughly 20 countries with small sample sizes.",
            (
                "Year fixed effects replace the macro index, and out-of-sample future years inherit the most recent available training-year effect."
                if result.spec.temporal_mode == "year_fe"
                else "Commercial property level indices were not used in this version; a chained quarterly proxy was built from HICP/CPI inflation-rate series rebased to 2021Q1 = 100."
            ),
            (
                "Year built was excluded from the model because the Capital IQ enrichment produced fewer than 150 confident matches."
                if not result.prep_metadata["include_year_built"]
                else "Year built is only partially observed after Capital IQ enrichment and must be interpreted with caution."
            ),
        ],
    }
    if result.spec.use_size_asset_interactions:
        metadata["size_slopes_by_asset_type"] = _size_slopes_by_asset_type(
            result.coefficient_table,
            result.prep_metadata["asset_type_levels"],
        )
    return metadata


def _headline_fold_extra_metrics(result: RefitResult) -> dict[str, Any]:
    metrics = {
        "headline_fold_mape_pct": float(result.rolling_origin["headline_fold"]["model_metrics"]["mape_pct"]),
        "headline_fold_baseline_mape_pct": float(result.rolling_origin["headline_fold"]["baseline_metrics"]["mape_pct"]),
    }
    if result.spec.name == "change_e":
        train_frame = result.model_frame.loc[result.model_frame["transaction_year"].isin([2021, 2022, 2023, 2024, 2025])].copy()
        test_frame = result.model_frame.loc[result.model_frame["transaction_year"].eq(2026)].copy()
        prepared_train, prepared_test = _prepare_frames_for_fit(train_frame, test_frame, result.spec)
        model = _fit_ols(prepared_train, result.formula)
        predictions = _predict_deal_size_eur_mn(model, prepared_test)
        prepared_test = prepared_test.copy()
        prepared_test["ape_pct"] = np.abs((prepared_test["actual_deal_size_eur_mn"] - predictions) / prepared_test["actual_deal_size_eur_mn"]) * 100.0
        french_rows = prepared_test.loc[prepared_test["country_group"].astype(str).eq("France")]
        metrics["headline_fold_french_mape_pct"] = float(french_rows["ape_pct"].mean()) if not french_rows.empty else None
    return metrics


def _save_refit_outputs(result: RefitResult, sign_flips_vs_previous: list[str], previous_spec_name: str) -> None:
    REFIT_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    coefficient_path = REFIT_DIAGNOSTICS_DIR / f"{result.spec.name}_coefficients.csv"
    metrics_path = REFIT_DIAGNOSTICS_DIR / f"{result.spec.name}_metrics.json"
    summary_path = REFIT_DIAGNOSTICS_DIR / f"{result.spec.name}_summary.txt"
    result.coefficient_table.to_csv(coefficient_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(result.model.summary().as_text())
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "spec": asdict(result.spec),
                "previous_spec": previous_spec_name,
                "training_sample_size": int(len(result.model_frame)),
                "rolling_origin": result.rolling_origin,
                "random_5_fold": result.random_5_fold,
                "rsquared": result.rsquared,
                "rsquared_adj": result.rsquared_adj,
                "sign_flips_vs_previous": sign_flips_vs_previous,
                "implausibility_notes": result.implausibility_notes,
                **_headline_fold_extra_metrics(result),
            },
            handle,
            indent=2,
        )


def _export_promoted_result(result: RefitResult) -> None:
    metadata = _build_final_metadata(result)
    residual_pool = _build_bootstrap_residual_pool(result.model)
    comps_sample = build_anonymised_comps_sample(result.model_frame)
    _export_artifacts(result.model, residual_pool, comps_sample, metadata)


def run_change_series() -> list[dict[str, Any]]:
    dataset, pipeline_metadata = build_training_frame()
    baseline = evaluate_spec(dataset, pipeline_metadata, BASELINE_SPEC)
    previous_result = baseline
    collected_results: list[dict[str, Any]] = []

    for spec in [CHANGE_A_SPEC, CHANGE_B_SPEC, CHANGE_C_SPEC, CHANGE_D_SPEC]:
        current_result = evaluate_spec(dataset, pipeline_metadata, spec)
        sign_flips = _sign_flips(current_result.coefficient_table, previous_result.coefficient_table, compare_country_terms=True)
        payload = {
            "spec_name": spec.name,
            "previous_spec": previous_result.spec.name,
            "result": current_result,
            "sign_flips_vs_previous": sign_flips,
        }
        _save_refit_outputs(current_result, sign_flips, previous_result.spec.name)
        collected_results.append(payload)
        previous_result = current_result

        if spec.name == "change_d":
            # Promotion gate: headline fold has only 17 test rows so this threshold is noisy.
            # See model/artifacts/refit_audit/ for fold standard errors.
            headline_mape = float(current_result.rolling_origin["headline_fold"]["model_metrics"]["mape_pct"])
            rolling_mean_mape = float(current_result.rolling_origin["mean_mape_pct"])
            if headline_mape <= 55.0 and rolling_mean_mape <= 70.0:
                _export_promoted_result(current_result)
                sensitivity_result = evaluate_spec(dataset, pipeline_metadata, CHANGE_E_SPEC)
                sensitivity_flips = _sign_flips(
                    sensitivity_result.coefficient_table,
                    current_result.coefficient_table,
                    compare_country_terms=False,
                )
                sensitivity_payload = {
                    "spec_name": CHANGE_E_SPEC.name,
                    "previous_spec": current_result.spec.name,
                    "result": sensitivity_result,
                    "sign_flips_vs_previous": sensitivity_flips,
                }
                _save_refit_outputs(sensitivity_result, sensitivity_flips, current_result.spec.name)
                collected_results.append(sensitivity_payload)
            break

    return collected_results


if __name__ == "__main__":
    results = run_change_series()
    for item in results:
        result = item["result"]
        print(result.spec.name, round(result.rolling_origin["mean_mape_pct"], 2), round(result.rolling_origin["headline_fold"]["model_metrics"]["mape_pct"], 2))
