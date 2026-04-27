from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .chart_palette import ACCENT, DARK, LIGHT, MUTED, PALE, PRIMARY, SECONDARY, green_cmap
from .pipeline import assign_country_group, canonicalise_country, load_preqin_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
REFIT_DIR = ARTIFACTS_DIR / "stage_two_refits"
OUTPUT_DIR = ARTIFACTS_DIR / "deployment_justification"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
COMPS_SAMPLE_PATH = ARTIFACTS_DIR / "comps_sample.parquet"

KEY_PREQIN_MISSING_VALUE_FIELDS = [
    ("DEAL TYPE", "Used to keep only single-asset transactions in the deployed retrieval sample."),
    ("PRIMARY ASSET TYPE", "Defines the asset-type cells used in retrieval and benchmarking."),
    ("ASSET REGIONS", "Used to keep only European transactions before modelling."),
    ("ASSET COUNTRIES", "Needed to map each deal into the country-group heatmaps and benchmark cells."),
    ("ASSET CITIES", "Context field for city-level retrieval output; not a hard filter in the pipeline."),
    ("DEAL DATE", "Required to place each deal in the rolling-origin evaluation timeline."),
    ("DEAL SIZE (EUR MN)", "Model target and retrieval anchor; rows without it are dropped."),
    ("TOTAL SIZE (SQ. M.)", "Core size predictor and price-per-sqm denominator; rows without it are dropped."),
    (
        "INITIAL CAPITALIZATION RATE (%)",
        "Potential valuation feature, but effectively unusable because the raw workbook is almost entirely blank.",
    ),
]

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


def _format_int(value: Any) -> str:
    return f"{int(value):,}"


def _build_heatmap_counts(
    frame: pd.DataFrame,
    *,
    row_field: str,
    col_field: str,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    return (
        frame.groupby([row_field, col_field], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=row_order, columns=col_order, fill_value=0)
    )


def _draw_counts_heatmap(
    ax: plt.Axes,
    counts: pd.DataFrame,
    *,
    title: str,
    cmap_name: str,
    note: str | None = None,
) -> Any:
    image = ax.imshow(counts.to_numpy(), cmap=green_cmap(cmap_name), aspect="auto")
    ax.set_xticks(np.arange(len(counts.columns)), counts.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(counts.index)), counts.index.tolist())
    ax.set_xlabel("Country group")
    ax.set_ylabel("Asset type")
    ax.set_title(title)

    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            ax.text(col, row, str(int(counts.iat[row, col])), ha="center", va="center", fontsize=9)

    if note:
        ax.text(
            0.0,
            -0.18,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=MUTED,
        )
    return image


def _build_raw_preqin_heatmap_counts(raw: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    asset_order = metadata["asset_type_levels"]
    country_order = metadata["country_group_levels"]

    frame = raw.copy()
    frame["primary_asset_type"] = frame["PRIMARY ASSET TYPE"].astype(str).str.strip()
    unsupported_asset_mask = ~frame["primary_asset_type"].isin(asset_order)
    unsupported_asset_rows = int(unsupported_asset_mask.sum())
    frame = frame.loc[~unsupported_asset_mask].copy()
    frame = frame.loc[frame["ASSET COUNTRIES"].notna()].copy()
    frame["country"] = frame["ASSET COUNTRIES"].map(canonicalise_country)
    frame["country_group"] = frame["country"].map(assign_country_group)
    frame["primary_asset_type"] = pd.Categorical(frame["primary_asset_type"], categories=asset_order)
    frame["country_group"] = pd.Categorical(frame["country_group"], categories=country_order)

    counts = _build_heatmap_counts(
        frame,
        row_field="primary_asset_type",
        col_field="country_group",
        row_order=asset_order,
        col_order=country_order,
    )
    stats = {
        "raw_total_rows": int(len(raw)),
        "counted_rows": int(len(frame)),
        "missing_country_rows": int(raw["ASSET COUNTRIES"].isna().sum()),
        "unsupported_asset_rows": unsupported_asset_rows,
    }
    return counts, stats


def _build_filtered_retrieval_heatmap_counts(comps: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    return _build_heatmap_counts(
        comps,
        row_field="asset_type",
        col_field="country",
        row_order=metadata["asset_type_levels"],
        col_order=metadata["country_group_levels"],
    )


def _build_preqin_missing_value_summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_summary = pd.DataFrame(
        {
            "column": raw.columns,
            "non_missing_rows": raw.notna().sum().values,
            "missing_rows": raw.isna().sum().values,
            "missing_pct": (raw.isna().mean() * 100.0).values,
        }
    ).sort_values(["missing_pct", "missing_rows", "column"], ascending=[False, False, True]).reset_index(drop=True)

    key_spec = pd.DataFrame(KEY_PREQIN_MISSING_VALUE_FIELDS, columns=["column", "pipeline_role"])
    key_summary = key_spec.merge(full_summary, on="column", how="left")
    key_summary["pipeline_role"] = key_summary["pipeline_role"].map(lambda text: fill(text, width=44))
    return full_summary, key_summary


def _build_raw_country_density(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = raw.loc[raw["ASSET COUNTRIES"].notna()].copy()
    frame["country"] = frame["ASSET COUNTRIES"].map(lambda value: canonicalise_country(str(value).split(",")[0]))
    counts = frame["country"].value_counts().rename_axis("country").reset_index(name="deal_count")
    counts["share_pct"] = counts["deal_count"] / counts["deal_count"].sum() * 100.0
    counts = counts.sort_values(["deal_count", "country"], ascending=[True, True]).reset_index(drop=True)
    return counts, int(raw["ASSET COUNTRIES"].isna().sum())


def _write_preqin_missing_value_outputs(raw: pd.DataFrame) -> list[Path]:
    full_summary, _ = _build_preqin_missing_value_summaries(raw)

    csv_path = OUTPUT_DIR / "preqin_missing_values_summary.csv"
    full_summary.to_csv(csv_path, index=False)
    return [csv_path]


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
    counts = _build_filtered_retrieval_heatmap_counts(comps, metadata)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    image = _draw_counts_heatmap(
        ax,
        counts,
        title="The deployed retrieval universe has breadth, but coverage is uneven",
        cmap_name="retrieval_universe",
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Comparable count")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "retrieval_universe_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_raw_vs_retrieval_heatmaps(raw: pd.DataFrame, comps: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    raw_counts, raw_stats = _build_raw_preqin_heatmap_counts(raw, metadata)
    retrieval_counts = _build_filtered_retrieval_heatmap_counts(comps, metadata)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    raw_note = (
        f"Counted rows: {_format_int(raw_stats['counted_rows'])} / {_format_int(raw_stats['raw_total_rows'])}. "
        f"{_format_int(raw_stats['unsupported_asset_rows'])} raw rows sit in non-deployed asset types "
        f"(Niche or Land), and {_format_int(raw_stats['missing_country_rows'])} rows have missing asset country."
    )
    filtered_note = f"Counted rows: {_format_int(len(comps))}. Current filtered comparable universe."

    raw_image = _draw_counts_heatmap(
        axes[0],
        raw_counts,
        title="Original Preqin workbook before dissertation filters",
        cmap_name="preqin_raw_workbook",
        note=raw_note,
    )
    filtered_image = _draw_counts_heatmap(
        axes[1],
        retrieval_counts,
        title="Filtered dissertation retrieval universe",
        cmap_name="preqin_filtered_retrieval",
        note=filtered_note,
    )
    axes[1].set_ylabel("")

    raw_colorbar = fig.colorbar(raw_image, ax=axes[0], shrink=0.88)
    raw_colorbar.set_label("Raw deal count")
    filtered_colorbar = fig.colorbar(filtered_image, ax=axes[1], shrink=0.88)
    filtered_colorbar.set_label("Comparable count")

    fig.suptitle("Preqin coverage narrows materially once the dissertation retrieval filters are applied", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95], w_pad=2.5)

    output_path = OUTPUT_DIR / "preqin_raw_vs_retrieval_heatmaps.png"
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


def _plot_raw_country_density_pre_cut(raw: pd.DataFrame) -> Path:
    counts, missing_country_rows = _build_raw_country_density(raw)

    fig_height = max(8.0, len(counts) * 0.28)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(counts["country"], counts["share_pct"], color=PRIMARY, alpha=0.9)

    for bar, (_, row) in zip(bars, counts.iterrows(), strict=False):
        ax.text(
            float(bar.get_width()) + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{row['share_pct']:.1f}% ({_format_int(row['deal_count'])})",
            va="center",
            fontsize=8,
            color=DARK,
        )

    ax.set_xlabel("Share of raw pre-cut deals with a known country (%)")
    ax.set_ylabel("Country")
    ax.set_title("Raw Preqin workbook before filtering: deal density by country")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        -0.02,
        f"Known-country rows: {_format_int(int(counts['deal_count'].sum()))}. "
        f"Rows with missing asset country omitted: {_format_int(missing_country_rows)}. "
        f"If a raw cell lists multiple countries, the first listed country is used here.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_path = OUTPUT_DIR / "preqin_country_density_pre_cut.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_json(METADATA_PATH)
    comps = pd.read_parquet(COMPS_SAMPLE_PATH)
    raw_preqin = load_preqin_transactions()
    change_d_metrics = _load_json(REFIT_DIR / "change_d_metrics.json")
    refit_summary = _load_refit_summary()

    outputs = [
        _plot_change_d_fold_comparison(change_d_metrics),
        _plot_refit_tradeoff(refit_summary),
        _plot_retrieval_universe_heatmap(comps, metadata),
        _plot_raw_vs_retrieval_heatmaps(raw_preqin, comps, metadata),
        _plot_raw_country_density_pre_cut(raw_preqin),
        _plot_benchmark_cell_sizes(metadata),
    ]
    outputs.extend(_write_preqin_missing_value_outputs(raw_preqin))

    manifest = {"generated_files": [str(path.relative_to(PROJECT_ROOT)) for path in outputs]}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
