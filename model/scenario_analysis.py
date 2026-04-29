"""Evaluate scenario-level retrieval diagnostics for the dissertation workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chart_palette import ACCENT, DARK, LIGHT, PALE, PRIMARY, SECONDARY
from .train import (
    ARTIFACTS_DIR,
    CHANGE_D_FORMULA,
    RANDOM_SEED,
    _fit_ols,
    prepare_change_d_analysis_frame,
    run_random_cv,
    run_rolling_origin_cv,
)

SCENARIO_DIR = ARTIFACTS_DIR / "scenario_analysis"
SCENARIOS = [
    {
        "scenario": "A",
        "label": "A — Full",
        "formula": CHANGE_D_FORMULA,
    },
    {
        "scenario": "B",
        "label": "B — Partial",
        "formula": "log_deal_size_eur_mn ~ C(primary_asset_type) + C(country_group) + C(model_year_effect)",
    },
    {
        "scenario": "C",
        "label": "C — Minimal",
        "formula": "log_deal_size_eur_mn ~ C(primary_asset_type) + C(country_group)",
    },
]


def _training_sample_hash(model_frame: pd.DataFrame) -> str:
    """Training sample hash."""
    hashed = pd.util.hash_pandas_object(model_frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _evaluate_scenario(model_frame: pd.DataFrame, scenario: dict[str, str]) -> dict[str, Any]:
    """Evaluate scenario."""
    formula = scenario["formula"]
    rolling = run_rolling_origin_cv(model_frame, formula)
    random_5_fold = run_random_cv(model_frame, formula)
    fitted_model = _fit_ols(model_frame, formula)

    row: dict[str, Any] = {
        "scenario": scenario["scenario"],
        "scenario_label": scenario["label"],
        "formula": formula,
        "rolling_mean_mape_pct": float(rolling["mean_mape_pct"]),
        "headline_fold_mape_pct": float(rolling["headline_fold"]["model_metrics"]["mape_pct"]),
        "random_5_fold_mean_mape_pct": float(random_5_fold["mean_mape_pct"]),
        "rsquared": float(fitted_model.rsquared),
        "rsquared_adj": float(fitted_model.rsquared_adj),
        "rolling_naive_baseline_mean_mape_pct": float(rolling["baseline_mean_mape_pct"]),
        "headline_fold_naive_baseline_mape_pct": float(rolling["headline_fold"]["baseline_metrics"]["mape_pct"]),
        "n_coefficients": int(len(fitted_model.params)),
    }

    for fold in rolling["folds"]:
        year = int(fold["test_year"])
        row[f"fold_{year}_mape_pct"] = float(fold["model_metrics"]["mape_pct"])
        row[f"fold_{year}_naive_baseline_mape_pct"] = float(fold["baseline_metrics"]["mape_pct"])

    return row


def _build_results(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build results."""
    rows = [_evaluate_scenario(model_frame, scenario) for scenario in SCENARIOS]
    results = pd.DataFrame(rows)
    full_coefficients = float(results.loc[results["scenario"].eq("A"), "n_coefficients"].iloc[0])
    results["completeness_ratio"] = results["n_coefficients"] / full_coefficients
    results = results[
        [
            "scenario",
            "scenario_label",
            "formula",
            "completeness_ratio",
            "n_coefficients",
            "rolling_mean_mape_pct",
            "headline_fold_mape_pct",
            "random_5_fold_mean_mape_pct",
            "rsquared",
            "rsquared_adj",
            "rolling_naive_baseline_mean_mape_pct",
            "headline_fold_naive_baseline_mape_pct",
            "fold_2023_mape_pct",
            "fold_2024_mape_pct",
            "fold_2025_mape_pct",
            "fold_2026_mape_pct",
            "fold_2023_naive_baseline_mape_pct",
            "fold_2024_naive_baseline_mape_pct",
            "fold_2025_naive_baseline_mape_pct",
            "fold_2026_naive_baseline_mape_pct",
        ]
    ]

    per_fold = results[
        [
            "scenario",
            "scenario_label",
            "fold_2023_mape_pct",
            "fold_2024_mape_pct",
            "fold_2025_mape_pct",
            "fold_2026_mape_pct",
            "fold_2023_naive_baseline_mape_pct",
            "fold_2024_naive_baseline_mape_pct",
            "fold_2025_naive_baseline_mape_pct",
            "fold_2026_naive_baseline_mape_pct",
        ]
    ].copy()
    return results, per_fold


def _plot_completeness_vs_error(results: pd.DataFrame) -> Path:
    """Plot completeness vs error."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = results["completeness_ratio"].to_numpy(dtype=float)

    ax.plot(x, results["rolling_mean_mape_pct"], marker="o", linewidth=2.2, color=SECONDARY, label="Rolling-origin mean MAPE")
    ax.plot(x, results["headline_fold_mape_pct"], marker="o", linewidth=2.2, color=ACCENT, label="2026 headline MAPE")
    ax.axhline(
        float(results["headline_fold_naive_baseline_mape_pct"].iloc[0]),
        color=PRIMARY,
        linestyle="--",
        linewidth=1.8,
        label="2026 naive baseline MAPE",
    )

    for _, row in results.iterrows():
        ax.text(row["completeness_ratio"] + 0.01, row["rolling_mean_mape_pct"] + 1.2, row["scenario"], fontsize=10)
        ax.text(row["completeness_ratio"] + 0.01, row["headline_fold_mape_pct"] - 3.0, row["scenario"], fontsize=10)

    ax.set_xlabel("Completeness ratio (coefficient-count definition)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Model error as a function of feature completeness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = SCENARIO_DIR / "completeness_vs_error.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_per_scenario_fold_mape(results: pd.DataFrame) -> Path:
    """Plot per scenario fold MAPE."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(results))
    width = 0.18
    fold_columns = [
        ("fold_2023_mape_pct", "2023"),
        ("fold_2024_mape_pct", "2024"),
        ("fold_2025_mape_pct", "2025"),
        ("fold_2026_mape_pct", "2026"),
    ]
    colors = [PALE, LIGHT, SECONDARY, DARK]

    for offset, ((column, label), color) in enumerate(zip(fold_columns, colors)):
        positions = x + (offset - 1.5) * width
        bars = ax.bar(positions, results[column], width=width, color=color, label=label)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.2, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, results["scenario"])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-fold MAPE by feature-completeness scenario")
    ax.legend(title="Test year", frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = SCENARIO_DIR / "per_scenario_fold_mape.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_manifest(model_frame: pd.DataFrame, pipeline_metadata: dict[str, Any], prep_metadata: dict[str, Any]) -> Path:
    """Write manifest."""
    manifest = {
        "inputs_used": [
            "data/raw/Preqin_RealEstate_Deals-13_04_2026.xlsx",
            "data/raw/Data_test_2_cap_IQ_since_2021.xls",
            "data/indices/ecb_hicp_housing_water_electricity.csv",
            "data/indices/uk_cpi_inflation_rate.csv",
        ],
        "training_sample_hash": _training_sample_hash(model_frame),
        "training_sample_rows": int(len(model_frame)),
        "pipeline_filtered_rows": int(pipeline_metadata["filtered_row_count"]),
        "analysis_rows": int(prep_metadata["rows_available_for_model"]),
        "random_seed": RANDOM_SEED,
        "completeness_definition": "Number of fitted coefficients in the scenario specification divided by the number of fitted coefficients in scenario A.",
        "scenario_order": [scenario["scenario"] for scenario in SCENARIOS],
    }
    output_path = SCENARIO_DIR / "manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    """Run the module entry point."""
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    model_frame, pipeline_metadata, prep_metadata = prepare_change_d_analysis_frame()
    results, per_fold = _build_results(model_frame)

    scenario_results_path = SCENARIO_DIR / "scenario_results.csv"
    per_fold_path = SCENARIO_DIR / "per_fold_mape.csv"
    results.to_csv(scenario_results_path, index=False)
    per_fold.to_csv(per_fold_path, index=False)

    completeness_plot_path = _plot_completeness_vs_error(results)
    fold_plot_path = _plot_per_scenario_fold_mape(results)
    manifest_path = _write_manifest(model_frame, pipeline_metadata, prep_metadata)

    print("Scenario analysis complete.")
    print(f"Rows analysed: {len(model_frame)}")
    print(f"Scenario results: {scenario_results_path}")
    print(f"Per-fold MAPE: {per_fold_path}")
    print(f"Completeness plot: {completeness_plot_path}")
    print(f"Per-scenario fold plot: {fold_plot_path}")
    print(f"Manifest: {manifest_path}")
    print(results.loc[:, ["scenario", "completeness_ratio", "rolling_mean_mape_pct", "headline_fold_mape_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
