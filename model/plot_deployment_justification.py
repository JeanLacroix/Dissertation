from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chart_palette import ACCENT, DARK, LIGHT, MUTED, PALE, PRIMARY, SECONDARY, green_cmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
REFIT_DIR = ARTIFACTS_DIR / "stage_two_refits"
OUTPUT_DIR = ARTIFACTS_DIR / "deployment_justification"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
COMPS_SAMPLE_PATH = ARTIFACTS_DIR / "comps_sample.parquet"

SPEC_FILES = [
    ("A", "change_a_metrics.json"),
    ("B", "change_b_metrics.json"),
    ("C", "change_c_metrics.json"),
    ("D", "change_d_metrics.json"),
    ("E", "refit_e_metrics.json"),
    ("F", "refit_f_metrics.json"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_refit_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if "rolling_origin" in metrics:
        folds = metrics["rolling_origin"]["folds"]
        headline_fold = metrics["rolling_origin"]["headline_fold"]
        return {
            "label": label,
            "sample_size": int(metrics["training_sample_size"]),
            "rolling_mean_mape_pct": float(metrics["rolling_origin"]["mean_mape_pct"]),
            "headline_mape_pct": float(headline_fold["model_metrics"]["mape_pct"]),
            "headline_baseline_mape_pct": float(headline_fold["baseline_metrics"]["mape_pct"]),
        }

    return {
        "label": label,
        "sample_size": int(metrics.get("sample_size_after_uk_exclusion", metrics.get("sample_size_after_filters"))),
        "rolling_mean_mape_pct": float(metrics["rolling_mean_mape_pct"]),
        "headline_mape_pct": float(metrics["headline_fold_mape_pct"]),
        "headline_baseline_mape_pct": float(metrics["headline_fold_baseline_mape_pct"]),
    }


def _load_refit_summary() -> pd.DataFrame:
    rows = []
    for label, filename in SPEC_FILES:
        rows.append(_extract_refit_row(label, _load_json(REFIT_DIR / filename)))
    return pd.DataFrame(rows)


def _plot_change_d_fold_comparison(change_d_metrics: dict[str, Any]) -> Path:
    folds = change_d_metrics["rolling_origin"]["folds"]
    years = [str(fold["test_year"]) for fold in folds]
    model_mapes = [fold["model_metrics"]["mape_pct"] for fold in folds]
    baseline_mapes = [fold["baseline_metrics"]["mape_pct"] for fold in folds]

    x = np.arange(len(years))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    model_bars = ax.bar(x - width / 2, model_mapes, width=width, color=SECONDARY, label="Change D hedonic")
    base_bars = ax.bar(x + width / 2, baseline_mapes, width=width, color=PRIMARY, label="Naive benchmark")

    ax.axhline(change_d_metrics["rolling_origin"]["mean_mape_pct"], color=SECONDARY, linestyle="--", linewidth=1.5)
    ax.axhline(change_d_metrics["rolling_origin"]["baseline_mean_mape_pct"], color=PRIMARY, linestyle="--", linewidth=1.5)

    for bars in (model_bars, base_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.8, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, years)
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Change D did not beat the naive benchmark out of sample")
    ax.text(
        0.02,
        0.97,
        (
            f"Rolling mean MAPE: hedonic {change_d_metrics['rolling_origin']['mean_mape_pct']:.1f}% "
            f"vs naive {change_d_metrics['rolling_origin']['baseline_mean_mape_pct']:.1f}%\n"
            f"2026 headline fold: hedonic {model_mapes[-1]:.1f}% vs naive {baseline_mapes[-1]:.1f}%"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": PALE, "edgecolor": LIGHT},
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "change_d_model_vs_baseline_by_fold.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_refit_tradeoff(summary: pd.DataFrame) -> Path:
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, summary["rolling_mean_mape_pct"], marker="o", linewidth=2.2, color=SECONDARY, label="Rolling-origin mean MAPE")
    ax.plot(x, summary["headline_mape_pct"], marker="o", linewidth=2.2, color=ACCENT, label="2026 headline MAPE")
    ax.plot(
        x,
        summary["headline_baseline_mape_pct"],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=PRIMARY,
        label="2026 naive baseline MAPE",
    )

    for _, row in summary.iterrows():
        ax.text(row.name, row["rolling_mean_mape_pct"] + 1.8, f"{row['rolling_mean_mape_pct']:.1f}", ha="center", fontsize=9)
        ax.text(row.name, row["headline_mape_pct"] - 5.0, f"{row['headline_mape_pct']:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x, summary["label"])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Narrower specifications improved the headline fold, not the full time path")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "refit_tradeoff_path.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_retrieval_universe_heatmap(comps: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    asset_order = metadata["asset_type_levels"]
    country_order = metadata["country_group_levels"]
    counts = (
        comps.groupby(["asset_type", "country"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=asset_order, columns=country_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    image = ax.imshow(counts.to_numpy(), cmap=green_cmap("retrieval_universe"), aspect="auto")
    ax.set_xticks(np.arange(len(country_order)), country_order, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(asset_order)), asset_order)
    ax.set_xlabel("Country group")
    ax.set_ylabel("Asset type")
    ax.set_title("The deployed retrieval universe has breadth, but coverage is uneven")

    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            ax.text(col, row, str(int(counts.iat[row, col])), ha="center", va="center", fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Comparable count")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "retrieval_universe_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_benchmark_cell_sizes(metadata: dict[str, Any]) -> Path:
    benchmark_cells = pd.DataFrame(metadata["reference_benchmark"]["cells"])
    benchmark_cells["cell"] = benchmark_cells["asset_type"] + " | " + benchmark_cells["country"]
    benchmark_cells = benchmark_cells.sort_values("sample_size", ascending=True).reset_index(drop=True)

    fig_height = max(6.0, len(benchmark_cells) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = np.where(benchmark_cells["sample_size"] >= 10, PRIMARY, LIGHT)
    ax.barh(benchmark_cells["cell"], benchmark_cells["sample_size"], color=colors)
    ax.axvline(5, color=MUTED, linestyle=":", linewidth=1.2)
    ax.axvline(10, color=DARK, linestyle="--", linewidth=1.2)
    ax.text(10.3, len(benchmark_cells) - 1, "10 comps", fontsize=9, va="top")
    ax.text(5.3, len(benchmark_cells) - 4, "5 comps", fontsize=9, va="top")
    ax.set_xlabel("Sample size in deployed comparable universe")
    ax.set_title("Comparable density is strong in some cells and thin in others")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "benchmark_cell_sample_sizes.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_json(METADATA_PATH)
    comps = pd.read_parquet(COMPS_SAMPLE_PATH)
    change_d_metrics = _load_json(REFIT_DIR / "change_d_metrics.json")
    refit_summary = _load_refit_summary()

    outputs = [
        _plot_change_d_fold_comparison(change_d_metrics),
        _plot_refit_tradeoff(refit_summary),
        _plot_retrieval_universe_heatmap(comps, metadata),
        _plot_benchmark_cell_sizes(metadata),
    ]

    manifest = {"generated_files": [str(path.relative_to(PROJECT_ROOT)) for path in outputs]}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
