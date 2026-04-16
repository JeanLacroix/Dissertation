from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .outreach_db import export_ranked_contacts
from .outreach_service import build_deal_input, load_dashboard_context
from .paths import EXPORTS_DIR, ensure_outreach_dirs


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "mandate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a mandate against the SCBSM prototype and optionally export the result."
    )
    parser.add_argument("--mandate-name", type=str, default="New Paris office mandate")
    parser.add_argument("--asset-type", type=str, default="Office")
    parser.add_argument("--country", type=str, default="France")
    parser.add_argument("--zone", type=str, default="Paris")
    parser.add_argument("--city", type=str, default="Paris")
    parser.add_argument("--ticket-eur-mn", type=float, default=35.0)
    parser.add_argument("--cap-rate-pct", type=float, default=4.75)
    parser.add_argument("--size-sqm", type=float, default=5000.0)
    parser.add_argument("--transaction-date", type=str, default="")
    parser.add_argument("--export", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_outreach_dirs()
    deal_input = build_deal_input(
        {
            "mandate_name": args.mandate_name,
            "asset_type": args.asset_type,
            "country": args.country,
            "zone": args.zone,
            "city": args.city,
            "ticket_eur_mn": args.ticket_eur_mn,
            "cap_rate_pct": args.cap_rate_pct,
            "size_sqm": args.size_sqm,
            "transaction_date": args.transaction_date or None,
        }
    )
    context = load_dashboard_context(deal_input=deal_input.as_dict())
    evaluation = context.scbsm_evaluation
    export_frame = pd.DataFrame([evaluation])

    print("SCBSM mandate fit ready.")
    print(
        f"  mandate: {deal_input.mandate_name} | {deal_input.asset_type} | {deal_input.zone} | "
        f"EUR {deal_input.ticket_eur_mn:,.1f}m | {deal_input.cap_rate_pct:.2f}%"
    )
    print("")
    print(
        f"SCBSM | score={evaluation['outreach_score']:.1f} | fit={evaluation['fit_label']} | "
        f"summary={evaluation['match_summary']}"
    )

    if args.export is not None:
        export_path = export_ranked_contacts(export_frame, args.export)
    else:
        export_path = export_ranked_contacts(
            export_frame,
            EXPORTS_DIR / f"{_slugify(deal_input.mandate_name)}_scbsm_fit.csv",
        )
    print("")
    print(f"  export: {export_path}")


if __name__ == "__main__":
    main()
