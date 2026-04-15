from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from .train import (
    ARTIFACTS_DIR,
    CHANGE_D_FORMULA,
    RANDOM_SEED,
    ROLLING_FOLD_SPECS,
    _baseline_predict,
    _evaluate_predictions,
    _fit_ols,
    _prepare_formula_frames,
    _predict_deal_size_eur_mn,
    prepare_change_d_analysis_frame,
)

RF_DIR = ARTIFACTS_DIR / "rf_test"


def _prepare_rf_matrices(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_columns = ["primary_asset_type", "country_group", "model_year_effect"]
    numeric_columns = ["log_total_size_sqm"]

    train_features = train_frame.loc[:, numeric_columns + categorical_columns].copy()
    test_features = test_frame.loc[:, numeric_columns + categorical_columns].copy()

    for column in categorical_columns:
        train_features[column] = train_features[column].astype(str)
        test_features[column] = test_features[column].astype(str)

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined, columns=categorical_columns, dtype=float)
    x_train = encoded.iloc[: len(train_frame)].reset_index(drop=True)
    x_test = encoded.iloc[len(train_frame) :].reset_index(drop=True)
    return x_train, x_test


def _rolling_origin_rf_results(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []

    for fold_name, train_years, test_year in ROLLING_FOLD_SPECS:
        train_frame = model_frame.loc[model_frame["transaction_year"].isin(train_years)].copy()
        test_frame = model_frame.loc[model_frame["transaction_year"].eq(test_year)].copy()
        x_train, x_test = _prepare_rf_matrices(train_frame, test_frame)

        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        rf_model.fit(x_train, train_frame["log_deal_size_eur_mn"].to_numpy(dtype=float))

        rf_log_predictions = rf_model.predict(x_test)
        rf_predictions = np.exp(rf_log_predictions)
        prepared_train, prepared_test = _prepare_formula_frames(train_frame, test_frame, CHANGE_D_FORMULA)
        hedonic_model = _fit_ols(prepared_train, CHANGE_D_FORMULA)
        hedonic_predictions = _predict_deal_size_eur_mn(hedonic_model, prepared_test)
        hedonic_log_predictions = hedonic_model.predict(prepared_test)
        baseline_predictions = _baseline_predict(prepared_train, prepared_test)
        baseline_log_predictions = np.log(np.clip(baseline_predictions, a_min=1e-9, a_max=None))
        actual_actual = prepared_test["actual_deal_size_eur_mn"].to_numpy(dtype=float)
        actual_log = prepared_test["log_deal_size_eur_mn"].to_numpy(dtype=float)

        fold_rows.append(
            {
                "fold": fold_name,
                "test_year": int(test_year),
                "rf_mape_pct": float(_evaluate_predictions(actual_actual, rf_predictions)["mape_pct"]),
                "hedonic_mape_pct": float(_evaluate_predictions(actual_actual, hedonic_predictions)["mape_pct"]),
                "baseline_mape_pct": float(_evaluate_predictions(actual_actual, baseline_predictions)["mape_pct"]),
                "rf_r2_log_target": float(r2_score(actual_log, rf_log_predictions)),
                "hedonic_r2_log_target": float(r2_score(actual_log, hedonic_log_predictions)),
                "baseline_r2_log_target": float(r2_score(actual_log, baseline_log_predictions)),
            }
        )

    fold_results = pd.DataFrame(fold_rows)
    comparison = pd.DataFrame(
        [
            {
                "model": "Random forest",
                "rolling_origin_mean_mape_pct": float(fold_results["rf_mape_pct"].mean()),
                "headline_fold_mape_pct": float(fold_results.loc[fold_results["test_year"].eq(2026), "rf_mape_pct"].iloc[0]),
                "mean_oof_r2_log_target": float(fold_results["rf_r2_log_target"].mean()),
            },
            {
                "model": "Change D hedonic",
                "rolling_origin_mean_mape_pct": float(fold_results["hedonic_mape_pct"].mean()),
                "headline_fold_mape_pct": float(fold_results.loc[fold_results["test_year"].eq(2026), "hedonic_mape_pct"].iloc[0]),
                "mean_oof_r2_log_target": float(fold_results["hedonic_r2_log_target"].mean()),
            },
            {
                "model": "Naive baseline",
                "rolling_origin_mean_mape_pct": float(fold_results["baseline_mape_pct"].mean()),
                "headline_fold_mape_pct": float(fold_results.loc[fold_results["test_year"].eq(2026), "baseline_mape_pct"].iloc[0]),
                "mean_oof_r2_log_target": float(fold_results["baseline_r2_log_target"].mean()),
            },
        ]
    )
    return fold_results, comparison.iloc[0].to_dict(), comparison


def _fit_full_rf(model_frame: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    x_full, _ = _prepare_rf_matrices(model_frame, model_frame.iloc[0:0].copy())
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    rf_model.fit(x_full, model_frame["log_deal_size_eur_mn"].to_numpy(dtype=float))
    importances = pd.DataFrame(
        {
            "feature": x_full.columns,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return rf_model, importances


def _plot_fold_comparison(fold_results: pd.DataFrame) -> Path:
    x = np.arange(len(fold_results))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6))

    series = [
        ("rf_mape_pct", "Random forest", "#756bb1"),
        ("hedonic_mape_pct", "Change D hedonic", "#d95f0e"),
        ("baseline_mape_pct", "Naive baseline", "#6baed6"),
    ]
    for offset, (column, label, color) in enumerate(series):
        positions = x + (offset - 1) * width
        bars = ax.bar(positions, fold_results[column], width=width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, fold_results["test_year"].astype(str))
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Random forest vs hedonic vs naive baseline by rolling-origin fold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = RF_DIR / "rf_vs_hedonic_by_fold.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_readme(comparison: pd.DataFrame) -> Path:
    rf_row = comparison.loc[comparison["model"].eq("Random forest")].iloc[0]
    hedonic_row = comparison.loc[comparison["model"].eq("Change D hedonic")].iloc[0]
    baseline_row = comparison.loc[comparison["model"].eq("Naive baseline")].iloc[0]
    readme_text = (
        "This diagnostic tested whether a tree-based ensemble could materially improve predictive accuracy relative to the "
        "Change D hedonic specification on the same European transaction sample and the same rolling-origin folds. "
        f"The random forest produced a rolling-origin mean MAPE of {rf_row['rolling_origin_mean_mape_pct']:.1f}% and a "
        f"2026 headline-fold MAPE of {rf_row['headline_fold_mape_pct']:.1f}%. The corresponding figures for the Change D "
        f"hedonic were {hedonic_row['rolling_origin_mean_mape_pct']:.1f}% and {hedonic_row['headline_fold_mape_pct']:.1f}%, "
        f"while the naive country-by-asset-type median benchmark recorded {baseline_row['rolling_origin_mean_mape_pct']:.1f}% "
        f"and {baseline_row['headline_fold_mape_pct']:.1f}%. In this current out-of-sample test, the random forest does not materially rescue the "
        "result relative to the hedonic baseline. The interpretation is therefore that the performance ceiling is largely set "
        "by the available feature set and sample structure rather than by the choice between a linear model and a standard "
        "tree-based ensemble."
    )
    output_path = RF_DIR / "README.md"
    output_path.write_text(readme_text, encoding="utf-8")
    return output_path


def main() -> None:
    RF_DIR.mkdir(parents=True, exist_ok=True)
    model_frame, _, _ = prepare_change_d_analysis_frame()

    fold_results, rf_summary, comparison = _rolling_origin_rf_results(model_frame)
    _, importances = _fit_full_rf(model_frame)

    results_path = RF_DIR / "rf_results.csv"
    comparison.to_csv(results_path, index=False)

    importances_path = RF_DIR / "feature_importances.csv"
    importances.to_csv(importances_path, index=False)

    plot_path = _plot_fold_comparison(fold_results)
    readme_path = _write_readme(comparison)

    print("Random forest test complete.")
    print(f"Rows analysed: {len(model_frame)}")
    print(f"Results table: {results_path}")
    print(f"Fold comparison plot: {plot_path}")
    print(f"Feature importances: {importances_path}")
    print(f"README: {readme_path}")
    print(
        comparison.loc[:, ["model", "rolling_origin_mean_mape_pct", "headline_fold_mape_pct", "mean_oof_r2_log_target"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
