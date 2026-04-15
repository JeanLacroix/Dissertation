from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
SIGNAL_SHARE_GRID = [0.30, 0.50, 0.70]

SYNTHETIC_FEATURES = [
    "micro_location_score",
    "building_quality_score",
    "lease_quality_score",
    "tenant_covenant_score",
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
]


def _make_group_effect(values: pd.Series, rng: np.random.Generator, scale: float) -> np.ndarray:
    unique_values = sorted(values.astype(str).unique().tolist())
    lookup = {value: draw for value, draw in zip(unique_values, rng.normal(0.0, scale, size=len(unique_values)), strict=False)}
    return values.astype(str).map(lookup).to_numpy(dtype=float)


def _standardize(array: np.ndarray) -> np.ndarray:
    return (array - array.mean()) / array.std(ddof=0)


def _build_mock_frame(model_frame: pd.DataFrame, signal_share: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    frame = model_frame.copy()

    baseline_model = _fit_ols(frame, CHANGE_D_FORMULA)
    baseline_fitted = np.asarray(baseline_model.fittedvalues, dtype=float)
    baseline_residuals = np.asarray(baseline_model.resid, dtype=float)
    residual_variance = float(np.var(baseline_residuals, ddof=0))

    country_effect = _make_group_effect(frame["country_group"], rng, scale=0.55)
    asset_effect = _make_group_effect(frame["primary_asset_type"], rng, scale=0.50)
    year_effect = _make_group_effect(frame["transaction_year"], rng, scale=0.35)
    latent_core_1 = rng.normal(0.0, 1.0, size=len(frame))
    latent_core_2 = rng.normal(0.0, 1.0, size=len(frame))
    latent_core_3 = rng.normal(0.0, 1.0, size=len(frame))
    latent_core_4 = rng.normal(0.0, 1.0, size=len(frame))

    raw_features = {
        "micro_location_score": 0.60 * country_effect + 0.35 * year_effect + 0.75 * latent_core_1 + rng.normal(0.0, 0.45, len(frame)),
        "building_quality_score": 0.45 * asset_effect + 0.40 * year_effect + 0.70 * latent_core_2 + rng.normal(0.0, 0.45, len(frame)),
        "lease_quality_score": 0.40 * country_effect + 0.45 * asset_effect + 0.65 * latent_core_3 + rng.normal(0.0, 0.45, len(frame)),
        "tenant_covenant_score": 0.35 * country_effect + 0.30 * year_effect + 0.70 * latent_core_4 + rng.normal(0.0, 0.45, len(frame)),
    }

    for feature_name, raw_values in raw_features.items():
        frame[feature_name] = _standardize(np.asarray(raw_values, dtype=float))

    beta_raw = np.array([0.24, 0.18, 0.16, 0.14], dtype=float)
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
    model_frame = frame.copy()
    if target_mode == "mock":
        model_frame["log_deal_size_eur_mn"] = model_frame["mock_log_deal_size_eur_mn"]
        model_frame["actual_deal_size_eur_mn"] = model_frame["mock_actual_deal_size_eur_mn"]
        model_frame["actual_price_per_sqm_eur"] = model_frame["mock_actual_price_per_sqm_eur"]
    return model_frame


def _evaluate_spec(frame: pd.DataFrame, spec: dict[str, str]) -> dict[str, Any]:
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
    pivot = results.loc[results["target_mode"].eq("mock")].pivot_table(
        index="signal_share_of_current_residual_variance",
        columns="spec_name",
        values=["rolling_mean_mape_pct", "headline_fold_mape_pct", "random_5_fold_mean_mape_pct"],
    )
    pivot.columns = [f"{metric}__{spec_name}" for metric, spec_name in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"signal_share_of_current_residual_variance": "signal_share"})
    return pivot.sort_values("signal_share").reset_index(drop=True)


def _plot_precision_benchmark(results: pd.DataFrame) -> Path:
    plot_frame = results.loc[
        results["spec_name"].ne("real_change_d") & results["signal_share_of_current_residual_variance"].eq(BASE_SIGNAL_SHARE)
    ].copy().reset_index(drop=True)
    x = np.arange(len(plot_frame))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars_rolling = ax.bar(x - width / 2, plot_frame["rolling_mean_mape_pct"], width=width, color="#1f77b4", label="Rolling-origin mean MAPE")
    bars_headline = ax.bar(x + width / 2, plot_frame["headline_fold_mape_pct"], width=width, color="#ff7f0e", label="2026 headline MAPE")
    ax.axhline(
        float(plot_frame["headline_fold_naive_baseline_mape_pct"].iloc[0]),
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.6,
        label="2026 naive baseline MAPE",
    )

    for bars in (bars_rolling, bars_headline):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.2, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, ["Observed only", "+2 synthetic", "+4 synthetic"])
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


def _plot_fold_benchmark(results: pd.DataFrame) -> Path:
    plot_frame = results.loc[
        results["spec_name"].isin(["mock_observed_only", "mock_partial_completeness", "mock_extensive_dataset"])
        & results["signal_share_of_current_residual_variance"].eq(BASE_SIGNAL_SHARE)
    ].copy()
    x = np.arange(4)
    width = 0.22
    colors = ["#9ecae1", "#3182bd", "#08519c"]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for idx, (_, row) in enumerate(plot_frame.iterrows()):
        positions = x + (idx - 1) * width
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
    plot_frame = results.loc[results["target_mode"].eq("mock")].copy()
    x = plot_frame["signal_share_of_current_residual_variance"].astype(float).sort_values().unique()
    spec_order = [
        ("mock_observed_only", "Observed only", "#9ecae1"),
        ("mock_partial_completeness", "+2 synthetic", "#3182bd"),
        ("mock_extensive_dataset", "+4 synthetic", "#08519c"),
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


def _write_readme(generation_summary: dict[str, Any], results: pd.DataFrame) -> Path:
    observed_row = results.loc[results["spec_name"].eq("mock_observed_only")].iloc[0]
    extensive_row = results.loc[results["spec_name"].eq("mock_extensive_dataset")].iloc[0]
    real_row = results.loc[results["spec_name"].eq("real_change_d")].iloc[0]

    readme_text = f"""# Mock completeness benchmark

This analysis creates a synthetic richer-data benchmark to estimate what predictive precision could be achievable if materially more deal-level information were observed. The starting point is the existing Change D sample and fitted structure. A mock target is generated as the sum of: (i) the fitted Change D signal already explained by observed covariates, (ii) four synthetic standardised features representing micro-location quality, building quality, lease quality, and tenant covenant strength, and (iii) irreducible noise.

The synthetic features are not copied from external data. They are generated with group structure by country, asset type, and year plus idiosyncratic noise, then scaled so that they explain {generation_summary['signal_share_of_current_residual_variance']:.0%} of the current Change D residual variance in the base case. The remaining {generation_summary['irreducible_noise_share_of_current_residual_variance']:.0%} is left as noise.

On the real target, Change D records a rolling-origin mean MAPE of {real_row['rolling_mean_mape_pct']:.1f}%. On the base-case mock target, the observed-feature model records {observed_row['rolling_mean_mape_pct']:.1f}%, while the extensive-data model with all four synthetic features records {extensive_row['rolling_mean_mape_pct']:.1f}%. The script also runs a sensitivity sweep over 30%, 50%, and 70% explainable residual-variance shares. This should be interpreted as a structured sensitivity benchmark rather than as a claim about true achievable production accuracy.
"""
    output_path = MOCK_DIR / "README.md"
    output_path.write_text(readme_text, encoding="utf-8")
    return output_path


def main() -> None:
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
            "mock_partial_completeness": 0.5,
            "mock_extensive_dataset": 1.0,
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
                "mock_partial_completeness": 0.5,
                "mock_extensive_dataset": 1.0,
            }[spec["spec_name"]]
            sensitivity_rows.append(row)
    sensitivity_results = pd.DataFrame(sensitivity_rows)
    sensitivity_error_table = _build_sensitivity_error_table(sensitivity_results)

    results_path = MOCK_DIR / "benchmark_results.csv"
    results.to_csv(results_path, index=False)

    sensitivity_results_path = MOCK_DIR / "sensitivity_results.csv"
    sensitivity_results.to_csv(sensitivity_results_path, index=False)

    sensitivity_error_table_path = MOCK_DIR / "sensitivity_error_table.csv"
    sensitivity_error_table.to_csv(sensitivity_error_table_path, index=False)

    feature_summary_path = MOCK_DIR / "synthetic_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    generation_path = MOCK_DIR / "generation_spec.json"
    generation_path.write_text(json.dumps(generation_summary, indent=2), encoding="utf-8")

    precision_plot_path = _plot_precision_benchmark(results)
    fold_plot_path = _plot_fold_benchmark(results)
    sensitivity_plot_path = _plot_sensitivity_envelope(sensitivity_results)
    readme_path = _write_readme(generation_summary, results)

    print("Mock completeness benchmark complete.")
    print(f"Rows analysed: {len(model_frame)}")
    print(f"Results table: {results_path}")
    print(f"Sensitivity results: {sensitivity_results_path}")
    print(f"Sensitivity error table: {sensitivity_error_table_path}")
    print(f"Synthetic feature summary: {feature_summary_path}")
    print(f"Generation spec: {generation_path}")
    print(f"Precision plot: {precision_plot_path}")
    print(f"Fold plot: {fold_plot_path}")
    print(f"Sensitivity plot: {sensitivity_plot_path}")
    print(f"README: {readme_path}")
    print(results.loc[:, ["label", "rolling_mean_mape_pct", "headline_fold_mape_pct", "random_5_fold_mean_mape_pct"]].to_string(index=False))
    print()
    print(sensitivity_error_table.to_string(index=False))


if __name__ == "__main__":
    main()
