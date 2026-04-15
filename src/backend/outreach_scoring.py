from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


RELATIONSHIP_STAGE_BONUS = {
    "new": 4.0,
    "warm": 11.0,
    "active": 15.0,
    "cooling_off": -6.0,
}

LAST_OUTCOME_BONUS = {
    "positive": 8.0,
    "neutral": 1.0,
    "no_reply": -7.0,
    "not_now": -5.0,
    "none": 0.0,
}


def _parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _days_since(value: Any) -> int | None:
    parsed = _parse_date(value)
    if parsed is None:
        return None
    return (date.today() - parsed).days


def _latest_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["contact_id", "latest_event_date", "latest_outcome", "latest_channel", "next_action_date"])

    ordered = events.copy()
    ordered["event_date"] = pd.to_datetime(ordered["event_date"], errors="coerce")
    ordered = ordered.sort_values(["contact_id", "event_date", "event_id"])
    latest = ordered.groupby("contact_id", as_index=False).tail(1).copy()
    latest = latest.rename(
        columns={
            "event_date": "latest_event_date",
            "outcome": "latest_outcome",
            "channel": "latest_channel",
        }
    )
    latest["latest_event_date"] = latest["latest_event_date"].dt.date.astype(str)
    return latest[["contact_id", "latest_event_date", "latest_outcome", "latest_channel", "next_action_date"]]


def _zone_match_score(asset_zone: str, asset_city: str, contact_zone_focus: str, contact_city: str) -> float:
    if contact_zone_focus == asset_zone:
        score = 24.0
    elif asset_zone == "Paris" and contact_zone_focus == "IDF":
        score = 14.0
    elif asset_zone == "IDF" and contact_zone_focus == "Paris":
        score = 8.0
    elif contact_zone_focus == "Nationwide":
        score = 12.0
    elif asset_zone == "Province" and contact_zone_focus == "IDF":
        score = 4.0
    else:
        score = 0.0

    if asset_city and contact_city and asset_city.lower() == contact_city.lower():
        score += 5.0
    return score


def _asset_focus_score(asset_class: str, contact_asset_focus: str) -> float:
    if asset_class == contact_asset_focus:
        return 18.0
    if "Mixed" in {asset_class, contact_asset_focus}:
        return 10.0
    if asset_class == "Office" and contact_asset_focus == "Retail":
        return 0.0
    if asset_class == "Retail" and contact_asset_focus == "Office":
        return 0.0
    return 6.0


def _range_fit_score(value: float, lower: float | None, upper: float | None, full_score: float) -> float:
    if lower is None or upper is None or np.isnan(lower) or np.isnan(upper):
        return full_score * 0.4

    if lower <= value <= upper:
        return full_score

    band = max(upper - lower, 1e-6)
    distance = min(abs(value - lower), abs(value - upper))
    penalty_ratio = min(distance / band, 2.0)
    return max(0.0, full_score * (1.0 - 0.5 * penalty_ratio))


def _cooldown_penalty(days_since_last_touch: int | None) -> float:
    if days_since_last_touch is None:
        return 0.0
    if days_since_last_touch <= 7:
        return -18.0
    if days_since_last_touch <= 21:
        return -9.0
    if days_since_last_touch <= 45:
        return -3.0
    return 0.0


def _recommended_action(latest_outcome: str, days_since_last_touch: int | None) -> str:
    if days_since_last_touch is None:
        return "First-touch email with the asset fiche and target-yield angle."
    if latest_outcome == "positive" and (days_since_last_touch or 0) >= 14:
        return "Follow up with a tighter asset memo and propose a short call."
    if latest_outcome == "neutral" and (days_since_last_touch or 0) >= 10:
        return "Re-engage with one concrete angle: timing, pricing, or income visibility."
    if latest_outcome == "no_reply" and (days_since_last_touch or 0) >= 14:
        return "Retry via an alternate channel and shorten the pitch to one paragraph."
    if latest_outcome == "not_now":
        return "Hold until the next action date before re-opening the conversation."
    return "Keep warm and avoid over-contacting before the cooldown has passed."


def _fit_label(total_score: float) -> str:
    if total_score >= 80:
        return "High"
    if total_score >= 60:
        return "Medium"
    return "Low"


def score_contacts_for_asset(
    *,
    asset_row: pd.Series,
    contacts: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    latest = _latest_events(events)
    frame = contacts.merge(latest, how="left", on="contact_id")

    asset_zone = str(asset_row["zone"])
    asset_city = str(asset_row["city"])
    asset_class = str(asset_row["asset_class"])
    asset_value_eur_mn = float(asset_row["fair_value_eur_mn"])
    asset_yield_mid = float(asset_row["yield_mid_pct"])
    asset_yield_band = str(asset_row["cap_rate_range_pct"])

    frame["zone_match_score"] = frame.apply(
        lambda row: _zone_match_score(asset_zone, asset_city, str(row["zone_focus"]), str(row["city"])),
        axis=1,
    )
    frame["asset_focus_score"] = frame.apply(
        lambda row: _asset_focus_score(asset_class, str(row["asset_focus"])),
        axis=1,
    )
    frame["ticket_fit_score"] = frame.apply(
        lambda row: _range_fit_score(
            asset_value_eur_mn,
            row["min_ticket_eur_mn"],
            row["max_ticket_eur_mn"],
            18.0,
        ),
        axis=1,
    )
    frame["yield_fit_score"] = frame.apply(
        lambda row: _range_fit_score(
            asset_yield_mid,
            row["min_target_yield_pct"],
            row["max_target_yield_pct"],
            20.0,
        ),
        axis=1,
    )

    frame["relationship_stage_bonus"] = frame["relationship_stage"].map(RELATIONSHIP_STAGE_BONUS).fillna(0.0)
    frame["response_bonus"] = frame["response_rate_score"].fillna(0.0) * 10.0
    frame["priority_bonus"] = frame["strategic_priority"].fillna(0.0) * 2.0
    frame["latest_outcome"] = frame["latest_outcome"].fillna(frame["last_outcome"]).fillna("none")
    frame["outcome_bonus"] = frame["latest_outcome"].map(LAST_OUTCOME_BONUS).fillna(0.0)
    frame["days_since_last_touch"] = frame["latest_event_date"].fillna(frame["last_contact_date"]).map(_days_since)
    frame["cooldown_penalty"] = frame["days_since_last_touch"].map(_cooldown_penalty)

    frame["outreach_score"] = (
        frame["zone_match_score"]
        + frame["asset_focus_score"]
        + frame["ticket_fit_score"]
        + frame["yield_fit_score"]
        + frame["relationship_stage_bonus"]
        + frame["response_bonus"]
        + frame["priority_bonus"]
        + frame["outcome_bonus"]
        + frame["cooldown_penalty"]
    )
    frame["fit_label"] = frame["outreach_score"].map(_fit_label)
    frame["recommended_action"] = frame.apply(
        lambda row: _recommended_action(str(row["latest_outcome"]), row["days_since_last_touch"]),
        axis=1,
    )
    frame["suggested_pitch"] = frame.apply(
        lambda row: (
            f"Lead with the {asset_zone} {asset_class.lower()} angle, "
            f"mention the disclosed cap-rate band around {asset_yield_band}, "
            f"and anchor the pitch on a value of EUR {asset_value_eur_mn:,.1f}m."
        ),
        axis=1,
    )
    frame["selected_asset_id"] = asset_row["asset_id"]
    frame["selected_asset_name"] = asset_row["asset_name"]
    frame["selected_asset_value_eur_mn"] = asset_value_eur_mn
    frame["selected_asset_yield_mid_pct"] = asset_yield_mid

    return frame.sort_values(
        ["outreach_score", "strategic_priority", "response_rate_score", "full_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
