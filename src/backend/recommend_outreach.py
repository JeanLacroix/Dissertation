from __future__ import annotations

import argparse
from pathlib import Path

from .outreach_db import export_ranked_contacts
from .outreach_service import load_dashboard_context
from .paths import EXPORTS_DIR, ensure_outreach_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank outreach targets for a selected asset and optionally export the result."
    )
    parser.add_argument("--asset-id", type=str, default=None, help="Asset id to score against.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked contacts to print.")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional CSV path for the ranked contact list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_outreach_dirs()
    context = load_dashboard_context(asset_id=args.asset_id)
    ranked = context.ranked_contacts.head(args.top_k).copy()

    print("Outreach ranking ready.")
    print(f"  asset: {context.selected_asset['asset_name']} ({context.selected_asset['asset_id']})")
    print(f"  zone: {context.selected_asset['zone']}")
    print(f"  yield band: {context.selected_asset['cap_rate_range_pct']}")
    print("")
    for row in ranked.itertuples(index=False):
        print(
            f"{row.contact_id} | {row.full_name} | {row.company} | "
            f"score={row.outreach_score:.1f} | fit={row.fit_label} | action={row.recommended_action}"
        )

    if args.export is not None:
        export_path = export_ranked_contacts(ranked, args.export)
    else:
        export_path = export_ranked_contacts(
            ranked,
            EXPORTS_DIR / f"{context.selected_asset['asset_id']}_ranked_contacts.csv",
        )
    print("")
    print(f"  export: {export_path}")


if __name__ == "__main__":
    main()
