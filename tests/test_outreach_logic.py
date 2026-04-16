from __future__ import annotations

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src.backend.outreach_scoring import (
    compute_fund_lifecycle_status,
    compute_resale_deadline_status,
    derive_scbsm_profile,
    score_scbsm_for_deal,
)
from src.backend.outreach_service import build_deal_input, log_touchpoint, validate_profile_payload
from src.backend.paths import SEED_ASSETS_PATH


def _load_assets_frame() -> pd.DataFrame:
    return pd.read_csv(SEED_ASSETS_PATH)


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "investor_id",
            "event_date",
            "created_at_utc",
            "event_id",
            "touchpoint_type",
            "status_value",
        ]
    )


class OutreachLogicTests(TestCase):
    def test_build_deal_input_swaps_price_range_and_derives_ticket(self) -> None:
        deal = build_deal_input(
            {
                "mandate_name": " Reversed Range ",
                "asset_type": "Office",
                "country": "France",
                "zone": "Paris",
                "city": " Paris ",
                "price_min_eur_mn": 45.0,
                "price_max_eur_mn": 35.0,
                "ticket_eur_mn": None,
                "cap_rate_pct": "",
                "size_sqm": 6200,
            }
        )

        self.assertEqual(deal.mandate_name, "Reversed Range")
        self.assertEqual(deal.city, "Paris")
        self.assertEqual(deal.price_min_eur_mn, 35.0)
        self.assertEqual(deal.price_max_eur_mn, 45.0)
        self.assertEqual(deal.ticket_eur_mn, 40.0)
        self.assertIsNone(deal.cap_rate_pct)

    def test_validate_profile_payload_flags_missing_mandatory_fields(self) -> None:
        payload = {
            "name": "",
            "firm": "SCBSM",
            "investor_type": "Listed Company",
            "sector_focus": [],
            "geographic_focus": "Paris intramuros",
            "ticket_min_eur_mn": 15.0,
            "ticket_max_eur_mn": 50.0,
            "portfolio_cap_rate_pct": 4.74,
            "portfolio_value_eur_mn": 473.8,
            "ltv_pct": 38.84,
            "last_updated": "2026-04-16",
            "source_tag": "",
        }

        missing = validate_profile_payload(payload)

        self.assertIn("name", missing)
        self.assertIn("sector_focus", missing)
        self.assertIn("source_tag", missing)
        self.assertNotIn("firm", missing)

    def test_log_touchpoint_sets_backdated_flag_for_old_entries(self) -> None:
        deal = build_deal_input()
        backdated_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=91)).date().isoformat()

        with patch("src.backend.outreach_service.append_outreach_event", return_value="evt_test") as mock_append:
            event_id = log_touchpoint(
                deal=deal,
                event_date=backdated_date,
                touchpoint_type="teaser_sent",
                status_value="sent",
                owner="Jean",
                notes="backdated test",
            )

        self.assertEqual(event_id, "evt_test")
        self.assertTrue(mock_append.call_args.kwargs["backdated_flag"])
        self.assertEqual(mock_append.call_args.kwargs["touchpoint_type"], "teaser_sent")

    def test_score_scbsm_for_default_deal_is_strong_match(self) -> None:
        assets = _load_assets_frame()
        profile = derive_scbsm_profile(assets)
        result = score_scbsm_for_deal(
            deal=build_deal_input(),
            assets=assets,
            events=_empty_events_frame(),
            profile=profile,
        )

        self.assertEqual(result["fit_label"], "Strong match")
        self.assertFalse(result["off_target_warning"])
        self.assertTrue(result["criteria"]["sector"]["match"])
        self.assertTrue(result["criteria"]["geography"]["match"])
        self.assertTrue(result["criteria"]["ticket"]["match"])
        self.assertTrue(result["yield_accretive"])

    def test_score_scbsm_for_clear_mismatch_flags_off_target(self) -> None:
        assets = _load_assets_frame()
        profile = derive_scbsm_profile(assets)
        mismatch_deal = build_deal_input(
            {
                "mandate_name": "Normandy hotel",
                "asset_type": "Hotel",
                "country": "France",
                "zone": "Normandy",
                "city": "Deauville",
                "price_min_eur_mn": 7.0,
                "price_max_eur_mn": 9.0,
                "cap_rate_pct": 6.5,
                "size_sqm": 5000.0,
            }
        )
        result = score_scbsm_for_deal(
            deal=mismatch_deal,
            assets=assets,
            events=_empty_events_frame(),
            profile=profile,
        )

        self.assertEqual(result["fit_label"], "Do not target")
        self.assertTrue(result["off_target_warning"])
        self.assertGreaterEqual(result["hard_mismatch_count"], 3)
        self.assertTrue(any("Hotel" in warning for warning in result["warning_messages"]))
        self.assertTrue(any("Deauville" in warning for warning in result["warning_messages"]))
        self.assertTrue(any("below SCBSM minimum" in warning for warning in result["warning_messages"]))

    def test_fund_lifecycle_status_flags_likely_seller_for_short_remaining_life(self) -> None:
        assets = _load_assets_frame()
        profile = derive_scbsm_profile(assets)
        current_year = pd.Timestamp.today().year
        fund_profile = replace(
            profile,
            investor_type="Fund",
            fund_vintage_year=current_year - 6,
            fund_life_years=7.0,
        )

        status = compute_fund_lifecycle_status(fund_profile)

        self.assertEqual(status["status"], "Likely seller")
        self.assertTrue(status["is_likely_seller"])
        self.assertLessEqual(status["years_remaining"], 2.0)

    def test_resale_deadline_status_approaching_when_within_18_months(self) -> None:
        assets = _load_assets_frame()
        profile = derive_scbsm_profile(assets)
        acquisition_date = (pd.Timestamp.today() - pd.DateOffset(years=4)).date().isoformat()
        marchand_profile = replace(
            profile,
            marchand_de_bien=True,
            acquisition_date=acquisition_date,
        )

        status = compute_resale_deadline_status(marchand_profile)

        self.assertEqual(status["status"], "Resale deadline approaching")
        self.assertIsNotNone(status["deadline"])
        self.assertLessEqual(status["months_remaining"], 18)
