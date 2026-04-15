# SCBSM Yield Extraction

This dataset was built from the SCBSM 2024 universal registration document already scraped into `data/raw/scbsm/tables/scbsm-2024-06-30-fr/`.

## Source tables used

1. `table_134.csv` supplies the asset list, geography, valuation date, and the last Cushman & Wakefield visit date.
2. `table_137.csv` supplies the asset-level fair value (`Juste Valeur (€ HFA)`).
3. `table_135.csv` supplies the capitalisation-rate assumptions (`Taux de capitalisation`) by geographic zone: `Paris`, `IDF`, and `Province`.

## Join logic

1. I keyed the asset list on `asset_number` from tables `134` and `137`.
2. I then joined the zone-level capitalisation assumptions from `table_135` back onto each asset via the `zone` field.
3. Because the URD reports the yield as a **zone range**, not as a single asset-specific point estimate, I materialised three fields:
   - `yield_min_pct`
   - `yield_max_pct`
   - `yield_mid_pct`

## What the yield means here

The `yield_*` fields are **not bespoke expert yields for each asset**. They are the capitalisation-rate band disclosed for the asset's zone in the expertise assumptions table. That means:

- every Paris asset shares the Paris cap-rate band;
- every IDF asset shares the IDF band;
- every Province asset shares the Province band.

This is still useful for outreach prioritisation because it gives the selection algorithm a transparent benchmark to compare against each contact's target yield range. It should not be represented as a uniquely appraised yield per building.

## Current output summary

- Assets normalised: 18
- Zones covered: IDF, Paris, Province
- Fair value total (EUR mn): 411.4

## Practical consequence for the outreach app

The outreach algorithm uses `yield_mid_pct` as the comparable benchmark when ranking contacts, while still surfacing the full disclosed band (`cap_rate_range_pct`) in the contact fiche.
