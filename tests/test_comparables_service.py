from __future__ import annotations

from unittest import TestCase

from src.backend.comparables_service import (
    ComparableQuery,
    classify_comparable_scenario,
    format_comparable_results,
    retrieve_comparables,
)


class ComparablesServiceTests(TestCase):
    def test_classify_comparable_scenarios(self) -> None:
        scenario_a = classify_comparable_scenario(has_size=True, has_year=True, cap_rate_pct=4.74)
        scenario_b = classify_comparable_scenario(has_size=False, has_year=True, cap_rate_pct=None)
        scenario_c = classify_comparable_scenario(has_size=False, has_year=False, cap_rate_pct=None)

        self.assertEqual(scenario_a["scenario"], "A")
        self.assertTrue(scenario_a["label"].startswith("A"))
        self.assertTrue(scenario_a["enhanced_mode_note"])
        self.assertEqual(scenario_b["scenario"], "B")
        self.assertEqual(scenario_b["enhanced_mode_note"], "")
        self.assertEqual(scenario_c["scenario"], "C")

    def test_retrieve_comparables_keeps_exact_country_pool_when_large_enough(self) -> None:
        result = retrieve_comparables(
            ComparableQuery(
                asset_type="Office",
                country="France",
                city="Paris",
                size_sqm=6000.0,
                transaction_year=2026,
                cap_rate_pct=4.74,
            )
        )

        self.assertFalse(result["widened"])
        self.assertEqual(result["retrieval_scope"], "Type x country")
        self.assertGreaterEqual(result["exact_match_count"], 5)
        self.assertEqual(len(result["results"]), 10)
        self.assertTrue((result["results"]["primary_asset_type"] == "Office").all())
        self.assertGreaterEqual(result["benchmark"]["sample_size"], 5)

    def test_retrieve_comparables_widens_when_exact_pool_is_thin(self) -> None:
        result = retrieve_comparables(
            ComparableQuery(
                asset_type="Hotel",
                country="France",
                city="Paris",
                size_sqm=6000.0,
                transaction_year=2026,
                cap_rate_pct=None,
            )
        )

        self.assertTrue(result["widened"])
        self.assertLess(result["exact_match_count"], 5)
        self.assertNotEqual(result["retrieval_scope"], "Type x country")
        self.assertEqual(len(result["results"]), 10)
        self.assertTrue(result["benchmark"]["thin_cell"])

    def test_format_comparable_results_returns_expected_columns(self) -> None:
        result = retrieve_comparables(
            ComparableQuery(
                asset_type="Office",
                country="France",
                city="Paris",
                size_sqm=6000.0,
                transaction_year=2026,
                cap_rate_pct=4.74,
            )
        )
        display = format_comparable_results(result["results"].head(3))

        self.assertEqual(
            list(display.columns),
            [
                "Similarity score",
                "Deal date",
                "Country",
                "City",
                "Asset type",
                "Size (sqm)",
                "Deal size (EUR mn)",
                "Price / sqm (EUR)",
                "Initial cap rate (%)",
                "Scope",
            ],
        )
        self.assertEqual(len(display), 3)
