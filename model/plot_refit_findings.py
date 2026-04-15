from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chart_palette import ACCENT, DARK, GRID, LIGHT, MUTED, NEUTRAL, PALE, PRIMARY, SECONDARY, TERTIARY, green_cmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFIT_DIR = PROJECT_ROOT / "model" / "artifacts" / "stage_two_refits"

SPEC_FILES = [
    ("A", "change_a_metrics.json"),
    ("B", "change_b_metrics.json"),
    ("C", "change_c_metrics.json"),
    ("D", "change_d_metrics.json"),
    ("E", "refit_e_metrics.json"),
    ("F", "refit_f_metrics.json"),
]

COLORS = {
    "rolling": SECONDARY,
    "headline": ACCENT,
    "baseline": PRIMARY,
    "random": TERTIARY,
    "retained": PRIMARY,
    "collapsed": LIGHT,
    "actual": DARK,
    "model": SECONDARY,
    "naive": LIGHT,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_refit_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if "rolling_origin" in metrics:
        folds = metrics["rolling_origin"]["folds"]
        fold_mapes = [fold["model_metrics"]["mape_pct"] for fold in folds]
        headline_fold = metrics["rolling_origin"]["headline_fold"]
        return {
            "label": label,
            "sample_size": metrics["training_sample_size"],
            "rolling_mean_mape_pct": metrics["rolling_origin"]["mean_mape_pct"],
            "random_mean_mape_pct": metrics["random_5_fold"]["mean_mape_pct"],
            "headline_mape_pct": headline_fold["model_metrics"]["mape_pct"],
            "headline_baseline_mape_pct": headline_fold["baseline_metrics"]["mape_pct"],
            "pooled_french_mape_pct": np.nan,
            "fold_mapes": fold_mapes,
        }

    sample_size = metrics.get("sample_size_after_uk_exclusion", metrics.get("sample_size_after_filters"))
    pooled_french = metrics.get("pooled_french_test_mape_pct", metrics.get("pooled_french_office_test_mape_pct"))
    return {
        "label": label,
        "sample_size": sample_size,
        "rolling_mean_mape_pct": metrics["rolling_mean_mape_pct"],
        "random_mean_mape_pct": metrics["random_5_fold_mean_mape_pct"],
        "headline_mape_pct": metrics["headline_fold_mape_pct"],
        "headline_baseline_mape_pct": metrics["headline_fold_baseline_mape_pct"],
        "pooled_french_mape_pct": pooled_french,
        "fold_mapes": metrics["rolling_fold_mapes_pct"],
    }


def _load_refit_summary() -> pd.DataFrame:
    rows = []
    for label, filename in SPEC_FILES:
        metrics = _load_json(REFIT_DIR / filename)
        rows.append(_extract_refit_row(label, metrics))
    return pd.DataFrame(rows)


def _plot_refit_path(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(summary))

    ax.plot(
        x,
        summary["rolling_mean_mape_pct"],
        marker="o",
        linewidth=2.2,
        color=COLORS["rolling"],
        label="Rolling-origin mean MAPE",
    )
    ax.plot(
        x,
        summary["headline_mape_pct"],
        marker="o",
        linewidth=2.2,
        color=COLORS["headline"],
        label="2026 headline MAPE",
    )
    ax.plot(
        x,
        summary["headline_baseline_mape_pct"],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=COLORS["baseline"],
        label="2026 naive baseline MAPE",
    )
    ax.plot(
        x,
        summary["random_mean_mape_pct"],
        marker="o",
        linewidth=2.0,
        linestyle=":",
        color=COLORS["random"],
        label="Random 5-fold mean MAPE",
    )

    for metric_column, offset in [
        ("rolling_mean_mape_pct", 2.0),
        ("headline_mape_pct", -4.0),
    ]:
        for x_pos, y_val in zip(x, summary[metric_column]):
            ax.text(x_pos, y_val + offset, f"{y_val:.1f}", ha="center", va="center", fontsize=9)

    ax.axhline(70, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.axhline(60, color=GRID, linestyle="--", linewidth=1)
    ax.axhline(55, color=GRID, linestyle=":", linewidth=1)
    ax.axhline(50, color=NEUTRAL, linestyle=":", linewidth=1)
    ax.text(len(summary) - 0.6, 71.5, "70%", fontsize=9, color=MUTED)
    ax.text(len(summary) - 0.6, 61.5, "60%", fontsize=9, color=MUTED)
    ax.text(len(summary) - 0.6, 56.5, "55%", fontsize=9, color=MUTED)
    ax.text(len(summary) - 0.6, 51.5, "50%", fontsize=9, color=MUTED)

    ax.set_xticks(x, summary["label"])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Stage-two refit path: sample narrowing improved the headline fold, not the full time path")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_performance_path.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_fold_heatmap(summary: pd.DataFrame) -> Path:
    fold_labels = ["2023", "2024", "2025", "2026"]
    heatmap = np.array(summary["fold_mapes"].tolist(), dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    image = ax.imshow(heatmap, cmap=green_cmap("refit_fold_mape"), aspect="auto")

    ax.set_xticks(np.arange(len(fold_labels)), fold_labels)
    ax.set_yticks(np.arange(len(summary)), summary["label"])
    ax.set_xlabel("Test year")
    ax.set_ylabel("Specification")
    ax.set_title("Rolling-origin fold MAPE heatmap")

    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax.text(col, row, f"{heatmap[row, col]:.1f}", ha="center", va="center", fontsize=9, color="black")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("MAPE (%)")
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_fold_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_e_f_dashboard(summary: pd.DataFrame) -> Path:
    subset = summary.loc[summary["label"].isin(["E", "F"])].copy()
    metrics = [
        ("rolling_mean_mape_pct", "Rolling mean"),
        ("headline_mape_pct", "2026 headline"),
        ("pooled_french_mape_pct", "Pooled France"),
        ("headline_baseline_mape_pct", "2026 baseline"),
    ]

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 6))

    e_vals = [float(subset.loc[subset["label"].eq("E"), key].iloc[0]) for key, _ in metrics]
    f_vals = [float(subset.loc[subset["label"].eq("F"), key].iloc[0]) for key, _ in metrics]

    bars_e = ax.bar(x - width / 2, e_vals, width=width, color=SECONDARY, label="Refit E")
    bars_f = ax.bar(x + width / 2, f_vals, width=width, color=PRIMARY, label="Refit F")

    for bars in (bars_e, bars_f):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(60, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.axhline(55, color=GRID, linestyle=":", linewidth=1)
    ax.axhline(50, color=NEUTRAL, linestyle=":", linewidth=1)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("E and F diagnostics: good-looking headline folds did not survive broader checks")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_e_f_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_f_country_scope() -> Path:
    metrics = _load_json(REFIT_DIR / "refit_f_metrics.json")
    counts = pd.Series(metrics["office_country_counts_before_grouping"]).sort_values()
    retained = set(metrics["retained_country_groups"])
    colors = [COLORS["retained"] if country in retained else COLORS["collapsed"] for country in counts.index]

    fig_height = max(6.0, len(counts) * 0.26)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(counts.index, counts.values, color=colors)
    ax.axvline(15, color=DARK, linestyle="--", linewidth=1.2)
    ax.text(15.3, len(counts) - 0.5, "15-office threshold", fontsize=9, va="top")

    ax.set_xlabel("Office transactions in non-UK sample")
    ax.set_title("Refit F country composition before regrouping")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_f_country_scope.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_f_headline_single_deal() -> Path:
    headline = pd.read_csv(REFIT_DIR / "refit_f_headline_top10_residuals.csv")
    row = headline.iloc[0]
    labels = ["Actual", "Model", "Naive baseline"]
    values = [
        float(row["actual_deal_size_eur_mn"]),
        float(row["predicted_eur_mn"]),
        float(row["baseline_predicted_eur_mn"]),
    ]
    colors = [COLORS["actual"], COLORS["model"], COLORS["naive"]]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10)

    title = f"Refit F headline fold was a single Dublin office deal: {row['DEAL NAME']}"
    ax.set_title(title)
    ax.set_ylabel("Deal size (EUR mn)")
    ax.text(
        0.02,
        0.96,
        f"2026 Office-only non-UK test count: 1\nAPE: {float(row['ape_pct']):.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": PALE, "edgecolor": LIGHT},
    )
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_f_single_deal_headline.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_sample_size_tradeoff(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    scatter = ax.scatter(
        summary["sample_size"],
        summary["headline_mape_pct"],
        s=summary["rolling_mean_mape_pct"] * 2.0,
        c=np.arange(len(summary)),
        cmap=green_cmap("refit_sample_size"),
        alpha=0.85,
    )

    for _, row in summary.iterrows():
        ax.text(row["sample_size"] + 4, row["headline_mape_pct"] + 1.2, row["label"], fontsize=10)

    ax.set_xlabel("Training sample size")
    ax.set_ylabel("2026 headline MAPE (%)")
    ax.set_title("Lower headline MAPE came with narrower samples, not stronger time-path stability")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.9)
    colorbar.set_label("Specification order")
    ax.text(
        0.02,
        0.98,
        "Bubble size scales with rolling-origin mean MAPE",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": PALE, "edgecolor": LIGHT},
    )
    fig.tight_layout()

    output_path = REFIT_DIR / "refit_sample_size_tradeoff.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    summary = _load_refit_summary()
    outputs = [
        _plot_refit_path(summary),
        _plot_fold_heatmap(summary),
        _plot_e_f_dashboard(summary),
        _plot_f_country_scope(),
        _plot_f_headline_single_deal(),
        _plot_sample_size_tradeoff(summary),
    ]

    manifest = {"generated_files": [str(path.relative_to(PROJECT_ROOT)) for path in outputs]}
    (REFIT_DIR / "graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
