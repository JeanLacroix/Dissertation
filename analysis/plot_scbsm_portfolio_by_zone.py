"""Generate a zone-level composition chart from the SCBSM asset seed."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.chart_palette import DARK, GRID, LIGHT, MUTED, PALE, PRIMARY
from src.backend.paths import EXPORTS_DIR, SEED_ASSETS_PATH, ensure_outreach_dirs

DEFAULT_OUTPUT_PATH = EXPORTS_DIR / "scbsm_portfolio_composition_by_zone.png"
DEFAULT_SUMMARY_PATH = EXPORTS_DIR / "scbsm_portfolio_composition_by_zone.csv"
FIGURE_TITLE = "Figure X. SCBSM portfolio composition by zone based on extracted public disclosures"


def _build_zone_summary(assets: pd.DataFrame) -> pd.DataFrame:
    """Build zone summary."""
    frame = assets.copy()
    frame["fair_value_eur_mn"] = pd.to_numeric(frame["fair_value_eur_mn"], errors="coerce")

    summary = (
        frame.groupby("zone", dropna=False)
        .agg(
            asset_count=("asset_id", "size"),
            fair_value_eur_mn=("fair_value_eur_mn", "sum"),
        )
        .reset_index()
        .sort_values(["fair_value_eur_mn", "asset_count"], ascending=[False, False], ignore_index=True)
    )
    return summary


def _plot_zone_fair_value(summary: pd.DataFrame, output_path: Path) -> Path:
    """Plot zone fair value."""
    fig, ax = plt.subplots(figsize=(9.5, 6))

    x_labels = summary["zone"].astype(str).tolist()
    y_values = summary["fair_value_eur_mn"].astype(float).tolist()
    asset_counts = summary["asset_count"].astype(int).tolist()

    bars = ax.bar(
        x_labels,
        y_values,
        color=[PRIMARY, LIGHT, PALE][: len(summary)],
        edgecolor=DARK,
        linewidth=1.1,
        width=0.62,
    )

    max_value = max(y_values) if y_values else 0.0
    for bar, fair_value, count in zip(bars, y_values, asset_counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            fair_value + max(max_value * 0.02, 2.0),
            f"EUR {fair_value:,.1f}m\nn={count}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK,
        )

    ax.set_title(FIGURE_TITLE, fontsize=13, color=DARK, pad=16)
    ax.set_xlabel("Zone / geography", color=DARK)
    ax.set_ylabel("Portfolio fair value (EUR m)", color=DARK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=DARK)

    total_assets = int(summary["asset_count"].sum())
    total_value = float(summary["fair_value_eur_mn"].sum())
    fig.text(
        0.125,
        0.02,
        (
            f"Labels show fair value and asset count by zone. "
            f"Current extracted sample: {total_assets} assets, EUR {total_value:,.1f}m."
        ),
        fontsize=9,
        color=MUTED,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Plot SCBSM portfolio fair value by zone from the extracted public-disclosure asset seed."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SEED_ASSETS_PATH,
        help="CSV asset seed to read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="PNG file to write.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="CSV summary file to write alongside the chart.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the module entry point."""
    args = parse_args()
    ensure_outreach_dirs()

    assets = pd.read_csv(args.input)
    summary = _build_zone_summary(assets)
    summary.to_csv(args.summary_output, index=False)
    output_path = _plot_zone_fair_value(summary, args.output)

    print("SCBSM zone composition chart written.")
    print(f"  chart: {output_path}")
    print(f"  summary: {args.summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
