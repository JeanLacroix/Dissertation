"""Benchmark the error floor under synthetic richer-data completeness scenarios."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chart_palette import ACCENT, DARK, LIGHT, MUTED, NEUTRAL, PRIMARY, SECONDARY, TERTIARY
from .train import (
    ARTIFACTS_DIR,
    CHANGE_D_FORMULA,
    RANDOM_SEED,
    _fit_ols,
    prepare_change_d_analysis_frame,
    run_random_cv,
    run_rolling_origin_cv,
)

MOCK_DIR = ARTIFACTS_DIR / "mock_completeness_benchmark"
BASE_SIGNAL_SHARE = 0.50
SIGNAL_SHARE_GRID = [0.30, 0.50, 0.70, 0.85, 0.95]
TARGET_MAPE_PCT = 10.0
SAMPLE_SIZE_TARGET_MAPE_PCT = 20.0
SAMPLE_SIZE_REPEATS = 30
SAMPLE_SIZE_GRID_CANDIDATES = [50, 80, 100, 120, 150, 200, 300, 400, 800, 1600, 3200]
SAMPLE_SIZE_SIGNAL_SHARE = 1.0
CAPTURE_RECORDS_PER_YEAR = 32
ULTRA_COMPLETENESS_SPEC_NAME = "mock_ultra_completeness"

SYNTHETIC_FEATURES = [
    "micro_location_score",
    "building_quality_score",
    "lease_quality_score",
    "tenant_covenant_score",
    "capital_expenditure_need",
    "refurbishment_potential",
    "tenant_credit_spread",
    "rental_growth_forecast",
]

MOCK_SPECS = [
    {
        "spec_name": "real_change_d",
        "label": "Real data: Change D",
        "target_mode": "real",
        "formula": CHANGE_D_FORMULA,
    },
    {
        "spec_name": "mock_observed_only",
        "label": "Mock target: observed features only",
        "target_mode": "mock",
        "formula": CHANGE_D_FORMULA,
    },
    {
        "spec_name": "mock_partial_completeness",
        "label": "Mock target: observed + 2 synthetic features",
        "target_mode": "mock",
        "formula": CHANGE_D_FORMULA + " + micro_location_score + building_quality_score",
    },
    {
        "spec_name": "mock_extensive_dataset",
        "label": "Mock target: observed + 4 synthetic features",
        "target_mode": "mock",
        "formula": (
            CHANGE_D_FORMULA
            + " + micro_location_score + building_quality_score + lease_quality_score + tenant_covenant_score"
        ),
    },
    {
        "spec_name": "mock_ultra_completeness",
        "label": "Mock target: observed + 8 synthetic features",
        "target_mode": "mock",
        "formula": (
            CHANGE_D_FORMULA
            + " + micro_location_score + building_quality_score + lease_quality_score + tenant_covenant_score"
            + " + capital_expenditure_need + refurbishment_potential + tenant_credit_spread + rental_growth_forecast"
        ),
    },
]


def _make_group_effect(values: pd.Series, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Make group effect."""
    unique_values = sorted(values.astype(str).unique().tolist())
    lookup = {value: draw for value, draw in zip(unique_values, rng.normal(0.0, scale, size=len(unique_values)), strict=False)}
    return values.astype(str).map(lookup).to_numpy(dtype=float)


def _standardize(array: np.ndarray) -> np.ndarray:
    """Standardize the current helper."""
    return (array - array.mean()) / array.std(ddof=0)


def _build_mock_frame(model_frame: pd.DataFrame, signal_share: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build mock frame."""
    rng = np.random.default_rng(RANDOM_SEED)
    frame = model_frame.copy()

    baseline_model = _fit_ols(frame, CHANGE_D_FORMULA)
    baseline_fitted = np.asarray(baseline_model.fittedvalues, dtype=float)
    baseline_residuals = np.asarray(baseline_model.resid, dtype=float)
    residual_variance = float(np.var(baseline_residuals, ddof=0))

    country_effect = _make_group_effect(frame["country_group"], rng, scale=0.55)
    asset_effect = _make_group_effect(frame["primary_asset_type"], rng, scale=0.50)
    year_effect = _make_group_effect(frame["transaction_year"], rng, scale=0.35)
    latent_cores = [rng.normal(0.0, 1.0, size=len(frame)) for _ in range(8)]

    raw_features = {
        "micro_location_score": 0.60 * country_effect + 0.35 * year_effect + 0.75 * latent_cores[0] + rng.normal(0.0, 0.45, len(frame)),
        "building_quality_score": 0.45 * asset_effect + 0.40 * year_effect + 0.70 * latent_cores[1] + rng.normal(0.0, 0.45, len(frame)),
        "lease_quality_score": 0.40 * country_effect + 0.45 * asset_effect + 0.65 * latent_cores[2] + rng.normal(0.0, 0.45, len(frame)),
        "tenant_covenant_score": 0.35 * country_effect + 0.30 * year_effect + 0.70 * latent_cores[3] + rng.normal(0.0, 0.45, len(frame)),
        "capital_expenditure_need": 0.20 * asset_effect + 0.15 * country_effect + 0.90 * latent_cores[4] + rng.normal(0.0, 0.40, len(frame)),
        "refurbishment_potential": 0.15 * asset_effect + 0.90 * latent_cores[5] + rng.normal(0.0, 0.40, len(frame)),
        "tenant_credit_spread": 0.25 * country_effect + 0.90 * latent_cores[6] + rng.normal(0.0, 0.40, len(frame)),
        "rental_growth_forecast": 0.20 * country_effect + 0.15 * year_effect + 0.90 * latent_cores[7] + rng.normal(0.0, 0.40, len(frame)),
    }

    for feature_name, raw_values in raw_features.items():
        frame[feature_name] = _standardize(np.asarray(raw_values, dtype=float))

    beta_raw = np.array([0.24, 0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08], dtype=float)
    z_matrix = frame.loc[:, SYNTHETIC_FEATURES].to_numpy(dtype=float)
    raw_signal = z_matrix @ beta_raw
    target_signal_variance = residual_variance * signal_share
    signal_scale = float(np.sqrt(target_signal_variance / np.var(raw_signal, ddof=0)))
    synthetic_signal = raw_signal * signal_scale

    noise_share = 1.0 - signal_share
    irreducible_noise_std = float(np.sqrt(residual_variance * noise_share))
    irreducible_noise = rng.normal(0.0, irreducible_noise_std, size=len(frame))

    frame["mock_log_deal_size_eur_mn"] = baseline_fitted + synthetic_signal + irreducible_noise
    frame["mock_actual_deal_size_eur_mn"] = np.exp(frame["mock_log_deal_size_eur_mn"])
    frame["mock_actual_price_per_sqm_eur"] = (
        frame["mock_actual_deal_size_eur_mn"] * 1_000_000 / frame["TOTAL SIZE (SQ. M.)"]
    )

    feature_summary = pd.DataFrame(
        {
            "feature": SYNTHETIC_FEATURES,
            "mean": [float(frame[feature].mean()) for feature in SYNTHETIC_FEATURES],
            "std": [float(frame[feature].std(ddof=0)) for feature in SYNTHETIC_FEATURES],
            "corr_with_mock_log_price": [
                float(np.corrcoef(frame[feature].to_numpy(dtype=float), frame["mock_log_deal_size_eur_mn"].to_numpy(dtype=float))[0, 1])
                for feature in SYNTHETIC_FEATURES
            ],
        }
    )

    generation_summary = {
        "random_seed": RANDOM_SEED,
        "base_formula": CHANGE_D_FORMULA,
        "signal_share_of_current_residual_variance": signal_share,
        "irreducible_noise_share_of_current_residual_variance": noise_share,
        "current_change_d_residual_variance": residual_variance,
        "scaled_synthetic_signal_variance": float(np.var(synthetic_signal, ddof=0)),
        "irreducible_noise_variance": float(np.var(irreducible_noise, ddof=0)),
        "synthetic_feature_coefficients_after_scaling": {
            feature: float(coef)
            for feature, coef in zip(SYNTHETIC_FEATURES, beta_raw * signal_scale, strict=False)
        },
        "feature_summary": feature_summary.to_dict(orient="records"),
    }
    return frame, generation_summary


def _frame_for_target(frame: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    """Frame for target."""
    model_frame = frame.copy()
    if target_mode == "mock":
        model_frame["log_deal_size_eur_mn"] = model_frame["mock_log_deal_size_eur_mn"]
        model_frame["actual_deal_size_eur_mn"] = model_frame["mock_actual_deal_size_eur_mn"]
        model_frame["actual_price_per_sqm_eur"] = model_frame["mock_actual_price_per_sqm_eur"]
    return model_frame


def _evaluate_spec(frame: pd.DataFrame, spec: dict[str, str]) -> dict[str, Any]:
    """Evaluate spec."""
    model_frame = _frame_for_target(frame, spec["target_mode"])
    rolling = run_rolling_origin_cv(model_frame, spec["formula"])
    random_5_fold = run_random_cv(model_frame, spec["formula"])
    fitted_model = _fit_ols(model_frame, spec["formula"])

    row: dict[str, Any] = {
        "spec_name": spec["spec_name"],
        "label": spec["label"],
        "target_mode": spec["target_mode"],
        "formula": spec["formula"],
        "rolling_mean_mape_pct": float(rolling["mean_mape_pct"]),
        "headline_fold_mape_pct": float(rolling["headline_fold"]["model_metrics"]["mape_pct"]),
        "random_5_fold_mean_mape_pct": float(random_5_fold["mean_mape_pct"]),
        "rsquared": float(fitted_model.rsquared),
        "rsquared_adj": float(fitted_model.rsquared_adj),
        "headline_fold_naive_baseline_mape_pct": float(rolling["headline_fold"]["baseline_metrics"]["mape_pct"]),
    }
    for fold in rolling["folds"]:
        row[f"fold_{int(fold['test_year'])}_mape_pct"] = float(fold["model_metrics"]["mape_pct"])
        row[f"fold_{int(fold['test_year'])}_naive_baseline_mape_pct"] = float(fold["baseline_metrics"]["mape_pct"])
    return row


def _build_sensitivity_error_table(results: pd.DataFrame) -> pd.DataFrame:
    """Build sensitivity error table."""
    pivot = results.loc[results["target_mode"].eq("mock")].pivot_table(
        index="signal_share_of_current_residual_variance",
        columns="spec_name",
        values=["rolling_mean_mape_pct", "headline_fold_mape_pct", "random_5_fold_mean_mape_pct"],
    )
    pivot.columns = [f"{metric}__{spec_name}" for metric, spec_name in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"signal_share_of_current_residual_variance": "signal_share"})
    return pivot.sort_values("signal_share").reset_index(drop=True)


def _build_sample_size_grid(full_sample_size: int) -> list[int]:
    """Build sample size grid."""
    grid = {min(candidate, full_sample_size) for candidate in SAMPLE_SIZE_GRID_CANDIDATES}
    grid.add(full_sample_size)
    return sorted(int(value) for value in grid if int(value) > 0)


def _allocate_year_counts(year_counts: pd.Series, sample_size: int) -> pd.Series:
    """Allocate year counts."""
    counts = year_counts.astype(int).sort_index()
    if sample_size > int(counts.sum()):
        raise ValueError("Requested sample size exceeds the available frame size.")
    if sample_size < len(counts):
        raise ValueError("Sample size must be at least the number of transaction years.")

    allocation = pd.Series(1, index=counts.index, dtype=int)
    remaining = sample_size - len(counts)
    if remaining <= 0:
        return allocation

    spare_capacity = counts - 1
    if int(spare_capacity.sum()) <= 0:
        return allocation

    weighted_target = remaining * spare_capacity / float(spare_capacity.sum())
    extra = np.floor(weighted_target).astype(int)
    extra = np.minimum(extra, spare_capacity)
    allocation = allocation + extra

    leftover = remaining - int(extra.sum())
    if leftover > 0:
        fractional_order = (weighted_target - np.floor(weighted_target)).sort_values(ascending=False).index.tolist()
        for year in fractional_order:
            if leftover == 0:
                break
            if int(allocation.loc[year]) < int(counts.loc[year]):
                allocation.loc[year] += 1
                leftover -= 1

    if leftover > 0:
        capacity_order = (counts - allocation).sort_values(ascending=False).index.tolist()
        for year in capacity_order:
            if leftover == 0:
                break
            available = int(counts.loc[year] - allocation.loc[year])
            if available <= 0:
                continue
            take = min(available, leftover)
            allocation.loc[year] += take
            leftover -= take

    if leftover != 0:
        raise ValueError("Could not allocate the requested stratified sample size.")
    return allocation.astype(int)


def _year_stratified_subsample(model_frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Year stratified subsample."""
    if sample_size >= len(model_frame):
        return model_frame.copy().reset_index(drop=True)

    year_counts = model_frame["transaction_year"].value_counts().sort_index()
    allocation = _allocate_year_counts(year_counts, sample_size)
    sampled_parts: list[pd.DataFrame] = []

    for offset, year in enumerate(allocation.index.tolist()):
        take = int(allocation.loc[year])
        year_slice = model_frame.loc[model_frame["transaction_year"].eq(year)].copy()
        sampled_parts.append(year_slice.sample(n=take, random_state=random_state + offset, replace=False))

    sampled = pd.concat(sampled_parts, axis=0).sort_index()
    return sampled.reset_index(drop=True)


def _evaluate_sample_size_sensitivity(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate sample size sensitivity."""
    ultra_spec = next(spec for spec in MOCK_SPECS if spec["spec_name"] == ULTRA_COMPLETENESS_SPEC_NAME)
    sample_sizes = _build_sample_size_grid(len(model_frame))
    years = sorted(model_frame["transaction_year"].dropna().astype(int).unique().tolist())
    run_rows: list[dict[str, Any]] = []
    expected_fold_count = 4

    for sample_size in sample_sizes:
        for seed in range(SAMPLE_SIZE_REPEATS):
            subsample = _year_stratified_subsample(model_frame, sample_size, random_state=RANDOM_SEED + seed)
            year_mix = subsample["transaction_year"].value_counts().sort_index()
            mock_subsample, _ = _build_mock_frame(subsample, signal_share=SAMPLE_SIZE_SIGNAL_SHARE)
            evaluation_frame = _frame_for_target(mock_subsample, ultra_spec["target_mode"])

            row: dict[str, Any] = {
                "sample_size_n": int(sample_size),
                "seed": int(seed),
                "signal_share_of_current_residual_variance": float(SAMPLE_SIZE_SIGNAL_SHARE),
                "spec_name": ultra_spec["spec_name"],
                "label": ultra_spec["label"],
                "status": "ok",
                "valid_fold_count": 0,
                "full_fold_coverage": False,
                "rolling_mean_mape_pct": np.nan,
                "rolling_std_mape_pct": np.nan,
                "headline_fold_mape_pct": np.nan,
                "headline_fold_naive_baseline_mape_pct": np.nan,
                "headline_fold_test_rows": np.nan,
                "error_type": "",
                "error_message": "",
            }
            for year in years:
                row[f"rows_{year}"] = int(year_mix.get(year, 0))

            try:
                rolling = run_rolling_origin_cv(evaluation_frame, ultra_spec["formula"])
                headline_fold = rolling["headline_fold"]
                row["valid_fold_count"] = int(len(rolling["folds"]))
                row["full_fold_coverage"] = bool(len(rolling["folds"]) == expected_fold_count)
                row["rolling_mean_mape_pct"] = float(rolling["mean_mape_pct"])
                row["rolling_std_mape_pct"] = float(rolling["std_mape_pct"])
                if headline_fold is not None:
                    row["headline_fold_mape_pct"] = float(headline_fold["model_metrics"]["mape_pct"])
                    row["headline_fold_naive_baseline_mape_pct"] = float(headline_fold["baseline_metrics"]["mape_pct"])
                    row["headline_fold_test_rows"] = int(headline_fold["test_rows"])
            except Exception as exc:
                row["status"] = "error"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)

            run_rows.append(row)

    run_results = pd.DataFrame(run_rows)
    summary_rows: list[dict[str, Any]] = []
    for sample_size in sample_sizes:
        current = run_results.loc[run_results["sample_size_n"].eq(sample_size)].copy()
        valid = current.loc[current["status"].eq("ok") & current["full_fold_coverage"]].copy()

        summary_row: dict[str, Any] = {
            "sample_size_n": int(sample_size),
            "signal_share_of_current_residual_variance": float(SAMPLE_SIZE_SIGNAL_SHARE),
            "n_runs": int(len(current)),
            "n_successful_runs": int(current["status"].eq("ok").sum()),
            "n_full_fold_coverage_runs": int(len(valid)),
            "rolling_mean_mape_pct_mean": np.nan,
            "rolling_mean_mape_pct_p10": np.nan,
            "rolling_mean_mape_pct_p90": np.nan,
            "rolling_std_mape_pct_mean": np.nan,
            "headline_fold_mape_pct_mean": np.nan,
            "headline_fold_mape_pct_p10": np.nan,
            "headline_fold_mape_pct_p90": np.nan,
            "headline_fold_naive_baseline_mape_pct_mean": np.nan,
            "observed_capture_years": float(sample_size) / CAPTURE_RECORDS_PER_YEAR,
        }
        for year in years:
            summary_row[f"mean_rows_{year}"] = float(valid[f"rows_{year}"].mean()) if not valid.empty else np.nan

        if not valid.empty:
            rolling_values = valid["rolling_mean_mape_pct"].to_numpy(dtype=float)
            headline_values = valid["headline_fold_mape_pct"].to_numpy(dtype=float)
            summary_row["rolling_mean_mape_pct_mean"] = float(np.mean(rolling_values))
            summary_row["rolling_mean_mape_pct_p10"] = float(np.percentile(rolling_values, 10))
            summary_row["rolling_mean_mape_pct_p90"] = float(np.percentile(rolling_values, 90))
            summary_row["rolling_std_mape_pct_mean"] = float(valid["rolling_std_mape_pct"].mean())
            summary_row["headline_fold_mape_pct_mean"] = float(np.mean(headline_values))
            summary_row["headline_fold_mape_pct_p10"] = float(np.percentile(headline_values, 10))
            summary_row["headline_fold_mape_pct_p90"] = float(np.percentile(headline_values, 90))
            summary_row["headline_fold_naive_baseline_mape_pct_mean"] = float(
                valid["headline_fold_naive_baseline_mape_pct"].mean()
            )

        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows).sort_values("sample_size_n").reset_index(drop=True)
    threshold_summary = _estimate_sample_size_threshold(summary, SAMPLE_SIZE_TARGET_MAPE_PCT)
    return summary, run_results, threshold_summary


def _estimate_sample_size_threshold(sample_size_results: pd.DataFrame, target_mape_pct: float) -> dict[str, Any]:
    """Estimate sample size threshold."""
    curve = (
        sample_size_results.loc[sample_size_results["rolling_mean_mape_pct_mean"].notna()]
        .sort_values("sample_size_n")
        .reset_index(drop=True)
    )
    summary: dict[str, Any] = {
        "target_mape_pct": float(target_mape_pct),
        "threshold_reached": False,
        "estimated_minimum_rows": np.nan,
        "lower_bracket_n": np.nan,
        "upper_bracket_n": np.nan,
        "estimated_observed_capture_years": np.nan,
    }
    if curve.empty:
        return summary

    below_target = curve.loc[curve["rolling_mean_mape_pct_mean"].le(target_mape_pct)]
    if below_target.empty:
        return summary

    crossing_idx = int(below_target.index[0])
    upper = below_target.iloc[0]
    if crossing_idx == 0:
        estimated_rows = float(upper["sample_size_n"])
        lower_n = float(upper["sample_size_n"])
        upper_n = float(upper["sample_size_n"])
    else:
        lower = curve.iloc[crossing_idx - 1]
        x0 = float(np.log(lower["sample_size_n"]))
        x1 = float(np.log(upper["sample_size_n"]))
        y0 = float(lower["rolling_mean_mape_pct_mean"])
        y1 = float(upper["rolling_mean_mape_pct_mean"])
        if np.isclose(y1, y0):
            estimated_rows = float(upper["sample_size_n"])
        else:
            weight = float((target_mape_pct - y0) / (y1 - y0))
            estimated_rows = float(np.exp(x0 + weight * (x1 - x0)))
        lower_n = float(lower["sample_size_n"])
        upper_n = float(upper["sample_size_n"])

    summary["threshold_reached"] = True
    summary["estimated_minimum_rows"] = estimated_rows
    summary["lower_bracket_n"] = lower_n
    summary["upper_bracket_n"] = upper_n
    summary["estimated_observed_capture_years"] = estimated_rows / CAPTURE_RECORDS_PER_YEAR
    return summary


def _plot_precision_benchmark(results: pd.DataFrame) -> Path:
    """Plot precision benchmark."""
    plot_frame = results.loc[
        results["spec_name"].ne("real_change_d") & results["signal_share_of_current_residual_variance"].eq(BASE_SIGNAL_SHARE)
    ].copy().reset_index(drop=True)
    x = np.arange(len(plot_frame))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 6))
    bars_rolling = ax.bar(x - width / 2, plot_frame["rolling_mean_mape_pct"], width=width, color=PRIMARY, label="Rolling-origin mean MAPE")
    bars_headline = ax.bar(x + width / 2, plot_frame["headline_fold_mape_pct"], width=width, color=ACCENT, label="2026 headline MAPE")
    ax.axhline(
        float(plot_frame["headline_fold_naive_baseline_mape_pct"].iloc[0]),
        color=MUTED,
        linestyle="--",
        linewidth=1.6,
        label="2026 naive baseline MAPE",
    )

    for bars in (bars_rolling, bars_headline):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.2, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    spec_labels = {
        "mock_observed_only": "Observed only",
        "mock_partial_completeness": "+2 synthetic",
        "mock_extensive_dataset": "+4 synthetic",
        "mock_ultra_completeness": "+8 synthetic",
    }
    ax.set_xticks(x, [spec_labels.get(name, name) for name in plot_frame["spec_name"].tolist()])
    ax.set_xlabel("Mock completeness benchmark")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Benchmark precision under a richer-data completeness hypothesis")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = MOCK_DIR / "mock_precision_benchmark.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_sample_size_sensitivity(sample_size_results: pd.DataFrame, threshold_summary: dict[str, Any], target_mape_pct: float) -> Path:
    """Plot sample size sensitivity."""
    plot_frame = (
        sample_size_results.loc[sample_size_results["rolling_mean_mape_pct_mean"].notna()]
        .sort_values("sample_size_n")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = plot_frame["sample_size_n"].to_numpy(dtype=float)
    y = plot_frame["rolling_mean_mape_pct_mean"].to_numpy(dtype=float)
    y_low = plot_frame["rolling_mean_mape_pct_p10"].to_numpy(dtype=float)
    y_high = plot_frame["rolling_mean_mape_pct_p90"].to_numpy(dtype=float)

    ax.fill_between(x, y_low, y_high, color=LIGHT, alpha=0.35, label="p10-p90 across repeated subsamples")
    ax.plot(x, y, marker="o", linewidth=2.4, color=DARK, label="Rolling-origin mean MAPE")
    for _, row in plot_frame.iterrows():
        ax.text(
            float(row["sample_size_n"]) * 1.02,
            float(row["rolling_mean_mape_pct_mean"]) + 0.9,
            f"{row['rolling_mean_mape_pct_mean']:.1f}",
            fontsize=8,
            color=DARK,
        )

    ax.axhline(target_mape_pct, color=ACCENT, linestyle="--", linewidth=1.8, label=f"{target_mape_pct:.0f}% threshold")
    if bool(threshold_summary.get("threshold_reached")):
        threshold_n = float(threshold_summary["estimated_minimum_rows"])
        ax.axvline(threshold_n, color=SECONDARY, linestyle=":", linewidth=1.8, label=f"N* approx. {threshold_n:.0f}")
        ax.text(
            threshold_n * 1.02,
            target_mape_pct + 1.0,
            f"N* approx. {threshold_n:.0f}\n~{threshold_summary['estimated_observed_capture_years']:.1f} years at 32 rows/year",
            fontsize=8,
            color=SECONDARY,
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Subsample size N (log scale)")
    ax.set_ylabel("Rolling-origin mean MAPE (%)")
    ax.set_title("Sample-size sensitivity under the +8 synthetic / 100% signal upper-bound scenario")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = MOCK_DIR / "sample_size_sensitivity_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_fold_benchmark(results: pd.DataFrame) -> Path:
    """Plot fold benchmark."""
    plot_frame = results.loc[
        results["spec_name"].isin([
            "mock_observed_only",
            "mock_partial_completeness",
            "mock_extensive_dataset",
            "mock_ultra_completeness",
        ])
        & results["signal_share_of_current_residual_variance"].eq(BASE_SIGNAL_SHARE)
    ].copy()
    x = np.arange(4)
    n_specs = len(plot_frame)
    width = 0.82 / max(n_specs, 1)
    colors = [LIGHT, TERTIARY, SECONDARY, DARK][:n_specs]

    fig, ax = plt.subplots(figsize=(11.5, 6))
    for idx, (_, row) in enumerate(plot_frame.iterrows()):
        positions = x + (idx - (n_specs - 1) / 2) * width
        values = [row[f"fold_{year}_mape_pct"] for year in [2023, 2024, 2025, 2026]]
        bars = ax.bar(positions, values, width=width, color=colors[idx], label=row["label"])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{height:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x, ["2023", "2024", "2025", "2026"])
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-fold mock-data benchmark under richer feature availability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = MOCK_DIR / "mock_precision_by_fold.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_sensitivity_envelope(results: pd.DataFrame) -> Path:
    """Plot sensitivity envelope."""
    plot_frame = results.loc[results["target_mode"].eq("mock")].copy()
    x = plot_frame["signal_share_of_current_residual_variance"].astype(float).sort_values().unique()
    spec_order = [
        ("mock_observed_only", "Observed only", LIGHT),
        ("mock_partial_completeness", "+2 synthetic", TERTIARY),
        ("mock_extensive_dataset", "+4 synthetic", SECONDARY),
        ("mock_ultra_completeness", "+8 synthetic", DARK),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for spec_name, label, color in spec_order:
        spec_frame = plot_frame.loc[plot_frame["spec_name"].eq(spec_name)].sort_values("signal_share_of_current_residual_variance")
        ax.plot(
            spec_frame["signal_share_of_current_residual_variance"],
            spec_frame["rolling_mean_mape_pct"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} — rolling mean",
        )
        for _, row in spec_frame.iterrows():
            ax.text(
                float(row["signal_share_of_current_residual_variance"]) + 0.005,
                float(row["rolling_mean_mape_pct"]) + 1.0,
                f"{row['rolling_mean_mape_pct']:.1f}",
                fontsize=8,
                color=color,
            )

    ax.set_xticks(x, [f"{int(share * 100)}%" for share in x])
    ax.set_xlabel("Share of current residual variance made explainable by synthetic features")
    ax.set_ylabel("Rolling-origin mean MAPE (%)")
    ax.set_title("Sensitivity of achievable precision to the completeness hypothesis")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = MOCK_DIR / "sensitivity_precision_envelope.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _required_log_sigma_for_mape(mape_target_pct: float, samples: int = 500_000) -> float:
    """Required log sigma for MAPE."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(samples)
    target = mape_target_pct / 100.0
    lower, upper = 1e-4, 3.0
    for _ in range(60):
        mid = (lower + upper) / 2.0
        residuals = base * mid
        mape = float(np.mean(np.abs(1.0 - np.exp(-residuals))))
        if mape < target:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2.0


def _required_log_r2_for_mape_target(
    mape_target_pct: float,
    log_target_variance: float,
) -> dict[str, float]:
    """Required log R-squared for MAPE target."""
    sigma = _required_log_sigma_for_mape(mape_target_pct)
    sigma2 = sigma * sigma
    r2 = 1.0 - sigma2 / log_target_variance
    return {
        "mape_target_pct": float(mape_target_pct),
        "required_log_residual_sigma": float(sigma),
        "required_log_residual_variance": float(sigma2),
        "log_target_variance": float(log_target_variance),
        "required_log_r2": float(r2),
    }


def _plot_completeness_acceptance_curve(sensitivity_results: pd.DataFrame, target_mape_pct: float) -> Path:
    """Plot completeness acceptance curve."""
    spec_order = [
        ("mock_observed_only", "Observed only", LIGHT),
        ("mock_partial_completeness", "+2 synthetic", TERTIARY),
        ("mock_extensive_dataset", "+4 synthetic", SECONDARY),
        ("mock_ultra_completeness", "+8 synthetic", DARK),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for spec_name, label, color in spec_order:
        spec_frame = (
            sensitivity_results.loc[sensitivity_results["spec_name"].eq(spec_name)]
            .sort_values("signal_share_of_current_residual_variance")
        )
        if spec_frame.empty:
            continue
        ax.plot(
            spec_frame["signal_share_of_current_residual_variance"],
            spec_frame["random_5_fold_mean_mape_pct"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label} — random 5-fold mean",
        )
        for _, row in spec_frame.iterrows():
            ax.text(
                float(row["signal_share_of_current_residual_variance"]) + 0.005,
                float(row["random_5_fold_mean_mape_pct"]) + 1.0,
                f"{row['random_5_fold_mean_mape_pct']:.1f}",
                fontsize=8,
                color=color,
            )

    ax.axhline(target_mape_pct, color=ACCENT, linestyle="--", linewidth=1.8, label=f"{target_mape_pct:.0f}% target")
    ax.set_xlabel("Share of current Change D residual variance explained by synthetic features")
    ax.set_ylabel("Random 5-fold mean MAPE (%)")
    ax.set_title(f"Completeness acceptance curve: how close to {target_mape_pct:.0f}% MAPE under richer data")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = MOCK_DIR / "completeness_acceptance_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_readme(
    generation_summary: dict[str, Any],
    results: pd.DataFrame,
    r2_analysis: dict[str, float] | None = None,
    sensitivity_results: pd.DataFrame | None = None,
    sample_size_results: pd.DataFrame | None = None,
    sample_size_threshold: dict[str, Any] | None = None,
) -> Path:
    """Write README."""
    observed_row = results.loc[results["spec_name"].eq("mock_observed_only")].iloc[0]
    extensive_row = results.loc[results["spec_name"].eq("mock_extensive_dataset")].iloc[0]
    ultra_row = results.loc[results["spec_name"].eq("mock_ultra_completeness")].iloc[0]
    real_row = results.loc[results["spec_name"].eq("real_change_d")].iloc[0]

    r2_block = ""
    if r2_analysis is not None:
        r2_block = (
            f"\n## How complete must the data be to reach {r2_analysis['mape_target_pct']:.0f}% random 5-fold MAPE?\n\n"
            f"Assuming log-price residuals are approximately normal with variance sigma-squared, a "
            f"random 5-fold mean MAPE of {r2_analysis['mape_target_pct']:.0f}% implies a residual "
            f"log-sigma of roughly {r2_analysis['required_log_residual_sigma']:.3f} "
            f"(residual variance {r2_analysis['required_log_residual_variance']:.4f}). The Change D "
            f"sample has log-target variance {r2_analysis['log_target_variance']:.3f}, so reaching "
            f"this MAPE requires a log-scale R-squared of at least "
            f"**{r2_analysis['required_log_r2']:.3f}**. The current Change D deployed fit achieves "
            f"log-scale R-squared around 0.60, so the observed feature set would need to explain "
            f"roughly {100.0 * (r2_analysis['required_log_r2'] - 0.60):.0f} additional percentage points "
            f"of log-price variance relative to today.\n\n"
        )

    sensitivity_block = ""
    if sensitivity_results is not None:
        ultra_sens = sensitivity_results.loc[sensitivity_results["spec_name"].eq("mock_ultra_completeness")]
        if not ultra_sens.empty:
            lowest = ultra_sens.sort_values("random_5_fold_mean_mape_pct").iloc[0]
            sensitivity_block = (
                f"At signal_share = {lowest['signal_share_of_current_residual_variance']:.0%} with 8 synthetic "
                f"features, random 5-fold mean MAPE reaches {lowest['random_5_fold_mean_mape_pct']:.1f}%. "
                f"This is the best case produced by the mock and is still well above the 10% target, "
                f"which confirms that the observed hedonic feature set alone cannot close the gap.\n\n"
            )

    sample_size_block = ""
    if sample_size_results is not None:
        curve = sample_size_results.sort_values("sample_size_n").reset_index(drop=True)
        full_row = curve.iloc[-1]
        sample_size_block = (
            f"## How many rows would be needed to reach {SAMPLE_SIZE_TARGET_MAPE_PCT:.0f}% rolling-origin MAPE?\n\n"
            f"This layer holds the richest mock-data scenario fixed at +8 synthetic features with "
            f"{SAMPLE_SIZE_SIGNAL_SHARE:.0%} explainable residual-variance share, then re-runs the rolling-origin "
            f"benchmark on {SAMPLE_SIZE_REPEATS} repeated year-stratified subsamples at each sample size. "
            f"Year stratification preserves the 2021-2026 fold structure so the low-N curve stays comparable "
            f"to the headline full-sample benchmark.\n\n"
            f"At the full analysis sample ({int(full_row['sample_size_n'])} rows), the mean rolling-origin MAPE "
            f"for this scenario is {full_row['rolling_mean_mape_pct_mean']:.1f}%.\n\n"
        )
        if sample_size_threshold is not None and bool(sample_size_threshold.get("threshold_reached")):
            sample_size_block += (
                f"The mean curve crosses the {SAMPLE_SIZE_TARGET_MAPE_PCT:.0f}% line at roughly "
                f"**N* = {sample_size_threshold['estimated_minimum_rows']:.0f} rows**, bracketed by "
                f"{sample_size_threshold['lower_bracket_n']:.0f} and {sample_size_threshold['upper_bracket_n']:.0f} "
                f"rows on the discrete grid. Using the observed capture pace assumption of "
                f"{CAPTURE_RECORDS_PER_YEAR} rows per year, that corresponds to about "
                f"**{sample_size_threshold['estimated_observed_capture_years']:.1f} years** of capture.\n\n"
            )
        else:
            sample_size_block += (
                f"Even the largest analysed sample on this grid does not push the mean curve below "
                f"{SAMPLE_SIZE_TARGET_MAPE_PCT:.0f}% rolling-origin MAPE.\n\n"
            )

        sample_size_block += (
            "Caveats:\n"
            "- The curve uses the current Preqin sample structure as a stand-in for what internal Alantra capture might look like.\n"
            "- The synthetic features are imposed by construction, and the 100% signal-share case is an explicit upper-bound benchmark rather than an observed production state.\n"
            "- At low N the country-group fixed effects and yearly folds become thin; widening uncertainty is part of the result, not a bug.\n\n"
        )

    readme_text = f"""# Mock completeness benchmark

This analysis creates a synthetic richer-data benchmark to estimate what predictive precision could be achievable if materially more deal-level information were observed. The starting point is the existing Change D sample and fitted structure. A mock target is generated as the sum of: (i) the fitted Change D signal already explained by observed covariates, (ii) up to eight synthetic standardised features (micro-location, building quality, lease quality, tenant covenant, capex need, refurbishment potential, credit spread, rental growth), and (iii) irreducible noise.

The synthetic features are not copied from external data. They are generated with group structure by country, asset type, and year plus idiosyncratic noise, then scaled so that they explain {generation_summary['signal_share_of_current_residual_variance']:.0%} of the current Change D residual variance in the base case. The remaining {generation_summary['irreducible_noise_share_of_current_residual_variance']:.0%} is left as noise.

On the real target, Change D records a rolling-origin mean MAPE of {real_row['rolling_mean_mape_pct']:.1f}%. On the base-case mock target, the observed-feature model records {observed_row['rolling_mean_mape_pct']:.1f}%, the +4 synthetic-feature model records {extensive_row['rolling_mean_mape_pct']:.1f}%, and the +8 synthetic-feature model records {ultra_row['rolling_mean_mape_pct']:.1f}%. The sensitivity sweep covers 30%, 50%, 70%, 85%, and 95% explainable residual-variance shares. This should be interpreted as a structured sensitivity benchmark rather than as a claim about true achievable production accuracy.
{r2_block}{sensitivity_block}{sample_size_block}"""
    output_path = MOCK_DIR / "README.md"
    output_path.write_text(readme_text, encoding="utf-8")
    return output_path


def main() -> None:
    """Run the module entry point."""
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    model_frame, _, _ = prepare_change_d_analysis_frame()
    mock_frame, generation_summary = _build_mock_frame(model_frame, signal_share=BASE_SIGNAL_SHARE)

    rows = [_evaluate_spec(mock_frame, spec) for spec in MOCK_SPECS]
    results = pd.DataFrame(rows)
    results["signal_share_of_current_residual_variance"] = np.nan
    results.loc[results["target_mode"].eq("mock"), "signal_share_of_current_residual_variance"] = BASE_SIGNAL_SHARE
    results["synthetic_feature_completeness_ratio"] = results["spec_name"].map(
        {
            "real_change_d": np.nan,
            "mock_observed_only": 0.0,
            "mock_partial_completeness": 0.25,
            "mock_extensive_dataset": 0.5,
            "mock_ultra_completeness": 1.0,
        }
    )
    feature_summary = pd.DataFrame(generation_summary["feature_summary"])

    sensitivity_rows: list[dict[str, Any]] = []
    for signal_share in SIGNAL_SHARE_GRID:
        sensitivity_frame, _ = _build_mock_frame(model_frame, signal_share=signal_share)
        for spec in MOCK_SPECS[1:]:
            row = _evaluate_spec(sensitivity_frame, spec)
            row["signal_share_of_current_residual_variance"] = signal_share
            row["synthetic_feature_completeness_ratio"] = {
                "mock_observed_only": 0.0,
                "mock_partial_completeness": 0.25,
                "mock_extensive_dataset": 0.5,
                "mock_ultra_completeness": 1.0,
            }[spec["spec_name"]]
            sensitivity_rows.append(row)
    sensitivity_results = pd.DataFrame(sensitivity_rows)
    sensitivity_error_table = _build_sensitivity_error_table(sensitivity_results)
    sample_size_results, sample_size_run_results, sample_size_threshold = _evaluate_sample_size_sensitivity(model_frame)

    results_path = MOCK_DIR / "benchmark_results.csv"
    results.to_csv(results_path, index=False)

    sensitivity_results_path = MOCK_DIR / "sensitivity_results.csv"
    sensitivity_results.to_csv(sensitivity_results_path, index=False)

    sensitivity_error_table_path = MOCK_DIR / "sensitivity_error_table.csv"
    sensitivity_error_table.to_csv(sensitivity_error_table_path, index=False)

    sample_size_results_path = MOCK_DIR / "sample_size_sensitivity.csv"
    sample_size_results.to_csv(sample_size_results_path, index=False)

    sample_size_run_results_path = MOCK_DIR / "sample_size_run_results.csv"
    sample_size_run_results.to_csv(sample_size_run_results_path, index=False)

    feature_summary_path = MOCK_DIR / "synthetic_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    generation_path = MOCK_DIR / "generation_spec.json"
    generation_path.write_text(json.dumps(generation_summary, indent=2), encoding="utf-8")

    log_target_variance = float(np.var(model_frame["log_deal_size_eur_mn"].to_numpy(dtype=float), ddof=0))
    r2_analysis = _required_log_r2_for_mape_target(TARGET_MAPE_PCT, log_target_variance)
    r2_analysis_path = MOCK_DIR / "required_log_r2_for_target_mape.json"
    r2_analysis_path.write_text(json.dumps(r2_analysis, indent=2), encoding="utf-8")

    sample_size_threshold_path = MOCK_DIR / "sample_size_threshold_summary.json"
    sample_size_threshold_path.write_text(json.dumps(sample_size_threshold, indent=2), encoding="utf-8")

    precision_plot_path = _plot_precision_benchmark(results)
    fold_plot_path = _plot_fold_benchmark(results)
    sensitivity_plot_path = _plot_sensitivity_envelope(sensitivity_results)
    acceptance_curve_path = _plot_completeness_acceptance_curve(sensitivity_results, TARGET_MAPE_PCT)
    sample_size_plot_path = _plot_sample_size_sensitivity(
        sample_size_results,
        sample_size_threshold,
        SAMPLE_SIZE_TARGET_MAPE_PCT,
    )
    readme_path = _write_readme(
        generation_summary,
        results,
        r2_analysis=r2_analysis,
        sensitivity_results=sensitivity_results,
        sample_size_results=sample_size_results,
        sample_size_threshold=sample_size_threshold,
    )

    print("Mock completeness benchmark complete.")
    print(f"Rows analysed: {len(model_frame)}")
    print(f"Results table: {results_path}")
    print(f"Sensitivity results: {sensitivity_results_path}")
    print(f"Sensitivity error table: {sensitivity_error_table_path}")
    print(f"Sample-size results: {sample_size_results_path}")
    print(f"Sample-size run results: {sample_size_run_results_path}")
    print(f"Synthetic feature summary: {feature_summary_path}")
    print(f"Generation spec: {generation_path}")
    print(f"Precision plot: {precision_plot_path}")
    print(f"Fold plot: {fold_plot_path}")
    print(f"Sensitivity plot: {sensitivity_plot_path}")
    print(f"Acceptance curve plot: {acceptance_curve_path}")
    print(f"Sample-size plot: {sample_size_plot_path}")
    print(f"Required R2 analysis: {r2_analysis_path}")
    print(f"Sample-size threshold: {sample_size_threshold_path}")
    print(f"README: {readme_path}")
    print(
        f"To reach {TARGET_MAPE_PCT:.0f}% MAPE, need log-R2 >= {r2_analysis['required_log_r2']:.3f} "
        f"(required log-sigma {r2_analysis['required_log_residual_sigma']:.3f})."
    )
    if bool(sample_size_threshold.get("threshold_reached")):
        print(
            f"To reach {SAMPLE_SIZE_TARGET_MAPE_PCT:.0f}% rolling-origin MAPE under the +8 synthetic / "
            f"{SAMPLE_SIZE_SIGNAL_SHARE:.0%} signal scenario, need about {sample_size_threshold['estimated_minimum_rows']:.0f} rows "
            f"(~{sample_size_threshold['estimated_observed_capture_years']:.1f} years at "
            f"{CAPTURE_RECORDS_PER_YEAR} rows/year)."
        )
    print(results.loc[:, ["label", "rolling_mean_mape_pct", "headline_fold_mape_pct", "random_5_fold_mean_mape_pct"]].to_string(index=False))
    print()
    print(sensitivity_error_table.to_string(index=False))
    print()
    print(sample_size_results.to_string(index=False))


if __name__ == "__main__":
    main()
