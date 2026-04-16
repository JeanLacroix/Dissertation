from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from .paths import SCBSM_PROFILE_PATH


RELATIONSHIP_STAGE_BONUS = {
    "new": 4.0,
    "warm": 10.0,
    "active": 14.0,
    "cooling_off": -6.0,
}

LAST_OUTCOME_BONUS = {
    "positive": 8.0,
    "neutral": 1.0,
    "no_reply": -7.0,
    "not_now": -5.0,
    "none": 0.0,
}


@dataclass(frozen=True)
class DealInput:
    mandate_name: str
    asset_type: str
    country: str
    zone: str
    city: str
    ticket_eur_mn: float
    cap_rate_pct: float
    size_sqm: float
    transaction_date: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScbsmProfile:
    investor_id: str
    company: str
    display_name: str
    title: str
    country_focus: str
    preferred_channel: str
    owner: str
    qualitative_focus: str
    notes: str
    asset_count: int
    total_fair_value_eur_mn: float
    min_ticket_eur_mn: float
    max_ticket_eur_mn: float
    median_ticket_eur_mn: float
    weighted_cap_rate_pct: float
    min_cap_rate_pct: float
    max_cap_rate_pct: float
    zone_focus: str
    asset_focus: str
    top_zone: str
    top_asset_class: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _days_since(value: Any) -> int | None:
    parsed = _parse_date(value)
    if parsed is None:
        return None
    return (date.today() - parsed).days


def _slug_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _latest_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "contact_id",
                "latest_event_date",
                "latest_outcome",
                "latest_channel",
                "latest_mandate_name",
                "next_action_date",
            ]
        )

    ordered = events.copy()
    ordered["event_date"] = pd.to_datetime(ordered["event_date"], errors="coerce")
    ordered = ordered.sort_values(["contact_id", "event_date", "event_id"])
    latest = ordered.groupby("contact_id", as_index=False).tail(1).copy()
    latest = latest.rename(
        columns={
            "event_date": "latest_event_date",
            "outcome": "latest_outcome",
            "channel": "latest_channel",
            "mandate_name": "latest_mandate_name",
        }
    )
    latest["latest_event_date"] = latest["latest_event_date"].dt.date.astype(str)
    return latest[
        [
            "contact_id",
            "latest_event_date",
            "latest_outcome",
            "latest_channel",
            "latest_mandate_name",
            "next_action_date",
        ]
    ]


def _range_fit_score(value: float, lower: float | None, upper: float | None, full_score: float) -> float:
    if pd.isna(lower) or pd.isna(upper):
        return full_score * 0.4
    if lower <= value <= upper:
        return full_score
    band = max(float(upper) - float(lower), 1e-6)
    distance = min(abs(value - float(lower)), abs(value - float(upper)))
    penalty_ratio = min(distance / band, 2.0)
    return max(0.0, full_score * (1.0 - 0.5 * penalty_ratio))


def _geography_score(deal: DealInput, row: pd.Series) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if _slug_text(row.get("country_focus", "")) == _slug_text(deal.country):
        score += 6.0

    zone_focus = str(row.get("zone_focus", "")).strip()
    zone_focus_slug = _slug_text(zone_focus)
    deal_zone_slug = _slug_text(deal.zone)

    if zone_focus_slug and zone_focus_slug == deal_zone_slug:
        score += 18.0
        reasons.append(f"Direct geography fit: {deal.zone} mandate matches {zone_focus} focus.")
    elif deal_zone_slug == "paris" and zone_focus_slug == "idf":
        score += 12.0
        reasons.append("Good geography fit: an IDF buyer can still underwrite a Paris mandate.")
    elif deal_zone_slug == "idf" and zone_focus_slug == "paris":
        score += 7.0
        reasons.append("Partial geography fit: Paris-focused capital may still look at wider IDF situations.")
    elif zone_focus_slug == "nationwide":
        score += 10.0
        reasons.append("Nationwide mandate coverage keeps this investor in scope.")

    deal_city_token = _slug_text(deal.city)
    city_focus_token = _slug_text(row.get("city_focus", ""))
    if deal_city_token and city_focus_token and (
        deal_city_token in city_focus_token or city_focus_token in deal_city_token
    ):
        score += 5.0
        reasons.append(f"City-level angle is relevant: {row.get('city_focus')} overlaps with {deal.city}.")
    return score, reasons


def _asset_focus_score(deal: DealInput, row: pd.Series) -> tuple[float, list[str]]:
    investor_focus = str(row.get("asset_focus", ""))
    reasons: list[str] = []
    if investor_focus == deal.asset_type:
        return 18.0, [f"Sector focus matches the mandate exactly: {deal.asset_type}."]
    if "Mixed" in {investor_focus, deal.asset_type}:
        return 10.0, [f"Asset focus is adjacent enough for a mixed-use / flexible underwriting conversation."]
    return 0.0, reasons


def _portfolio_cap_rate_score(deal: DealInput, row: pd.Series) -> tuple[float, list[str]]:
    current_cap_rate = row.get("current_portfolio_cap_rate_pct")
    reasons: list[str] = []
    if pd.isna(current_cap_rate):
        return 0.0, reasons

    gap = abs(float(current_cap_rate) - float(deal.cap_rate_pct))
    if gap <= 0.5:
        reasons.append(
            f"Current portfolio cap rate ({current_cap_rate:.2f}%) is very close to the mandate estimate ({deal.cap_rate_pct:.2f}%)."
        )
        return 8.0, reasons
    if gap <= 1.0:
        reasons.append("Current portfolio cap rate is directionally aligned with the mandate.")
        return 5.0, reasons
    if gap <= 1.5:
        return 2.0, reasons
    if gap >= 2.5:
        return -4.0, ["Portfolio cap-rate profile looks materially different from the mandate."]
    return 0.0, reasons


def _activity_timing_score(row: pd.Series) -> tuple[float, list[str], list[str]]:
    reasons: list[str] = []
    cautions: list[str] = []
    score = 0.0

    if bool(row.get("marchand_de_bien", False)):
        score += 5.0
        reasons.append("Marchand-de-bien status suggests recurring acquisition / disposal activity.")

    vintage = row.get("fund_vintage_year")
    hold = row.get("holding_period_years")
    if pd.isna(vintage) or pd.isna(hold):
        return score, reasons, cautions

    years_elapsed = date.today().year - int(vintage)
    years_remaining = float(hold) - years_elapsed

    if years_remaining <= 0:
        score -= 8.0
        cautions.append("Fund appears at or beyond its stated holding horizon, so sell-side behaviour may dominate.")
    elif years_remaining <= 2:
        score -= 3.0
        cautions.append("Fund looks relatively late in life. Buyer appetite may be lower than the headline profile suggests.")
    elif years_remaining <= 4:
        score += 3.0
        reasons.append("Fund timing still supports acquisition activity.")
    else:
        score += 1.0

    return score, reasons, cautions


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


def _recommended_action(latest_outcome: str, days_since_last_touch: int | None, fit_label: str) -> str:
    if fit_label == "High" and days_since_last_touch is None:
        return "Send the teaser immediately with a two-line rationale tied to geography, size, and yield."
    if latest_outcome == "positive" and (days_since_last_touch or 0) >= 10:
        return "Follow up with a short asset fiche and ask for a reaction window."
    if latest_outcome == "neutral" and (days_since_last_touch or 0) >= 10:
        return "Re-engage with one specific angle: cap rate, ticket size, or local footprint."
    if latest_outcome == "no_reply" and (days_since_last_touch or 0) >= 14:
        return "Retry on a different channel and tighten the message to one paragraph."
    if latest_outcome == "not_now":
        return "Respect the hold request and wait for the next action date."
    if fit_label == "Low":
        return "Do not prioritise this investor in the first outreach wave."
    return "Keep warm and avoid over-contacting before the cooldown has passed."


def _fit_label(total_score: float) -> str:
    if total_score >= 80:
        return "High"
    if total_score >= 60:
        return "Medium"
    return "Low"


def _build_match_summary(reasons: list[str], cautions: list[str]) -> tuple[str, str]:
    summary = " | ".join(reasons[:3]) if reasons else "No strong explicit fit driver was detected."
    caution = " | ".join(cautions[:2]) if cautions else ""
    return summary, caution


def score_contacts_for_deal(
    *,
    deal: DealInput,
    contacts: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    latest = _latest_events(events)
    frame = contacts.merge(latest, how="left", on="contact_id")

    zone_scores: list[float] = []
    asset_scores: list[float] = []
    ticket_scores: list[float] = []
    yield_scores: list[float] = []
    portfolio_scores: list[float] = []
    activity_scores: list[float] = []
    match_summaries: list[str] = []
    risk_flags: list[str] = []
    reason_lists: list[str] = []

    for row in frame.itertuples(index=False):
        row_series = pd.Series(row._asdict())

        zone_score, zone_reasons = _geography_score(deal, row_series)
        asset_score, asset_reasons = _asset_focus_score(deal, row_series)
        ticket_score = _range_fit_score(
            deal.ticket_eur_mn,
            row_series.get("min_ticket_eur_mn"),
            row_series.get("max_ticket_eur_mn"),
            18.0,
        )
        ticket_reasons = []
        if ticket_score >= 18.0:
            ticket_reasons.append("Ticket size sits inside the stated underwriting range.")
        elif ticket_score >= 10.0:
            ticket_reasons.append("Ticket size is close enough to the stated range to justify a test.")

        yield_score = _range_fit_score(
            deal.cap_rate_pct,
            row_series.get("min_target_yield_pct"),
            row_series.get("max_target_yield_pct"),
            18.0,
        )
        yield_reasons = []
        if yield_score >= 18.0:
            yield_reasons.append("Mandate cap rate sits inside the investor's stated return band.")
        elif yield_score >= 10.0:
            yield_reasons.append("Mandate cap rate is close to the investor's return band.")

        portfolio_score, portfolio_reasons = _portfolio_cap_rate_score(deal, row_series)
        activity_score, activity_reasons, activity_cautions = _activity_timing_score(row_series)

        reasons = zone_reasons + asset_reasons + ticket_reasons + yield_reasons + portfolio_reasons + activity_reasons
        summary, caution = _build_match_summary(reasons, activity_cautions)

        zone_scores.append(zone_score)
        asset_scores.append(asset_score)
        ticket_scores.append(ticket_score)
        yield_scores.append(yield_score)
        portfolio_scores.append(portfolio_score)
        activity_scores.append(activity_score)
        match_summaries.append(summary)
        risk_flags.append(caution)
        reason_lists.append("\n".join(f"- {item}" for item in reasons + activity_cautions))

    frame["zone_match_score"] = zone_scores
    frame["asset_focus_score"] = asset_scores
    frame["ticket_fit_score"] = ticket_scores
    frame["yield_fit_score"] = yield_scores
    frame["portfolio_cap_rate_score"] = portfolio_scores
    frame["activity_timing_score"] = activity_scores
    frame["match_summary"] = match_summaries
    frame["risk_flags"] = risk_flags
    frame["explicit_reasons"] = reason_lists

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
        + frame["portfolio_cap_rate_score"]
        + frame["activity_timing_score"]
        + frame["relationship_stage_bonus"]
        + frame["response_bonus"]
        + frame["priority_bonus"]
        + frame["outcome_bonus"]
        + frame["cooldown_penalty"]
    )
    frame["fit_label"] = frame["outreach_score"].map(_fit_label)
    frame["recommended_action"] = frame.apply(
        lambda row: _recommended_action(str(row["latest_outcome"]), row["days_since_last_touch"], str(row["fit_label"])),
        axis=1,
    )
    frame["suggested_pitch"] = frame.apply(
        lambda row: (
            f"Lead with the {deal.zone} {deal.asset_type.lower()} fit, "
            f"anchor the mandate at EUR {deal.ticket_eur_mn:,.1f}m and {deal.cap_rate_pct:.2f}% cap rate, "
            f"then explain why {row['company']} already looks aligned."
        ),
        axis=1,
    )

    frame["selected_mandate_name"] = deal.mandate_name
    frame["selected_asset_type"] = deal.asset_type
    frame["selected_zone"] = deal.zone
    frame["selected_city"] = deal.city
    frame["selected_ticket_eur_mn"] = deal.ticket_eur_mn
    frame["selected_cap_rate_pct"] = deal.cap_rate_pct

    return frame.sort_values(
        ["outreach_score", "strategic_priority", "response_rate_score", "full_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _load_scbsm_profile_metadata() -> dict[str, Any]:
    if not SCBSM_PROFILE_PATH.exists():
        return {
            "investor_id": "scbsm",
            "company": "SCBSM",
            "display_name": "SCBSM Portfolio Prototype",
            "title": "Single-investor prototype derived from the disclosed SCBSM portfolio",
            "country_focus": "France",
            "preferred_channel": "email",
            "owner": "Jean",
            "qualitative_focus": "SCBSM fit is derived from the disclosed portfolio and public cap-rate context.",
            "notes": "",
        }
    return json.loads(SCBSM_PROFILE_PATH.read_text(encoding="utf-8"))


def derive_scbsm_profile(assets: pd.DataFrame) -> ScbsmProfile:
    metadata = _load_scbsm_profile_metadata()
    portfolio = assets.copy()
    portfolio["fair_value_eur_mn"] = portfolio["fair_value_eur_mn"].fillna(0.0)
    portfolio["yield_mid_pct"] = portfolio["yield_mid_pct"].fillna(portfolio["yield_mid_pct"].median())

    total_fair_value = float(portfolio["fair_value_eur_mn"].sum())
    weighted_cap_rate = (
        float((portfolio["fair_value_eur_mn"] * portfolio["yield_mid_pct"]).sum() / total_fair_value)
        if total_fair_value > 0
        else float(portfolio["yield_mid_pct"].mean())
    )
    zone_mix = (
        portfolio.groupby("zone", dropna=False)["fair_value_eur_mn"]
        .sum()
        .sort_values(ascending=False)
    )
    asset_mix = (
        portfolio.groupby("asset_class", dropna=False)["fair_value_eur_mn"]
        .sum()
        .sort_values(ascending=False)
    )

    return ScbsmProfile(
        investor_id=str(metadata.get("investor_id", "scbsm")),
        company=str(metadata.get("company", "SCBSM")),
        display_name=str(metadata.get("display_name", "SCBSM Portfolio Prototype")),
        title=str(metadata.get("title", "Single-investor prototype derived from the disclosed SCBSM portfolio")),
        country_focus=str(metadata.get("country_focus", "France")),
        preferred_channel=str(metadata.get("preferred_channel", "email")),
        owner=str(metadata.get("owner", "Jean")),
        qualitative_focus=str(metadata.get("qualitative_focus", "")),
        notes=str(metadata.get("notes", "")),
        asset_count=int(len(portfolio)),
        total_fair_value_eur_mn=total_fair_value,
        min_ticket_eur_mn=float(portfolio["fair_value_eur_mn"].min()),
        max_ticket_eur_mn=float(portfolio["fair_value_eur_mn"].max()),
        median_ticket_eur_mn=float(portfolio["fair_value_eur_mn"].median()),
        weighted_cap_rate_pct=weighted_cap_rate,
        min_cap_rate_pct=float(portfolio["yield_mid_pct"].min()),
        max_cap_rate_pct=float(portfolio["yield_mid_pct"].max()),
        zone_focus=" / ".join(zone_mix.index.astype(str).tolist()),
        asset_focus=" / ".join(asset_mix.index.astype(str).tolist()),
        top_zone=str(zone_mix.index[0]) if not zone_mix.empty else "",
        top_asset_class=str(asset_mix.index[0]) if not asset_mix.empty else "",
    )


def _latest_scbsm_event(events: pd.DataFrame, investor_id: str) -> pd.Series | None:
    if events.empty:
        return None
    frame = events.copy()
    if "investor_id" not in frame.columns:
        return None
    frame = frame.loc[frame["investor_id"].eq(investor_id)].copy()
    if frame.empty:
        return None
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.sort_values(["event_date", "event_id"], ascending=[False, False])
    return frame.iloc[0]


def score_scbsm_for_deal(
    *,
    deal: DealInput,
    assets: pd.DataFrame,
    events: pd.DataFrame,
    profile: ScbsmProfile,
) -> dict[str, Any]:
    portfolio = assets.copy()
    portfolio["asset_class"] = portfolio["asset_class"].fillna("")
    portfolio["zone"] = portfolio["zone"].fillna("")
    portfolio["city"] = portfolio["city"].fillna("")
    portfolio["fair_value_eur_mn"] = portfolio["fair_value_eur_mn"].fillna(0.0)
    portfolio["yield_mid_pct"] = portfolio["yield_mid_pct"].fillna(profile.weighted_cap_rate_pct)

    same_country = _slug_text(deal.country) == _slug_text(profile.country_focus)
    same_zone = portfolio.loc[portfolio["zone"].map(_slug_text).eq(_slug_text(deal.zone))].copy()
    same_city = portfolio.loc[portfolio["city"].map(_slug_text).eq(_slug_text(deal.city))].copy()
    same_asset = portfolio.loc[portfolio["asset_class"].map(_slug_text).eq(_slug_text(deal.asset_type))].copy()
    same_zone_asset = same_zone.loc[same_zone["asset_class"].map(_slug_text).eq(_slug_text(deal.asset_type))].copy()

    reasons: list[str] = []
    cautions: list[str] = []

    zone_score = 0.0
    if not same_city.empty:
        zone_score = 26.0
        reasons.append(f"SCBSM already owns disclosed assets in {deal.city}.")
    elif not same_zone.empty:
        zone_score = 22.0
        reasons.append(f"SCBSM already has portfolio exposure in {deal.zone}.")
    elif same_country:
        zone_score = 10.0
        reasons.append("The mandate remains inside SCBSM's French portfolio footprint.")
    else:
        cautions.append("The mandate sits outside the current SCBSM country footprint.")

    asset_value = float(same_asset["fair_value_eur_mn"].sum())
    portfolio_total = max(profile.total_fair_value_eur_mn, 1e-6)
    asset_share = asset_value / portfolio_total
    asset_score = 0.0
    if not same_zone_asset.empty:
        asset_score = min(28.0, 14.0 + 40.0 * asset_share)
        reasons.append(
            f"SCBSM holds {deal.asset_type.lower()} assets in the same zone, which strengthens product fit."
        )
    elif not same_asset.empty:
        asset_score = min(24.0, 10.0 + 35.0 * asset_share)
        reasons.append(
            f"{deal.asset_type} is already represented in the SCBSM portfolio."
        )
    else:
        cautions.append(f"No disclosed SCBSM assets currently sit in the `{deal.asset_type}` bucket.")

    ticket_score = _range_fit_score(
        deal.ticket_eur_mn,
        profile.min_ticket_eur_mn,
        profile.max_ticket_eur_mn,
        20.0,
    )
    if ticket_score >= 20.0:
        reasons.append("The mandate ticket sits inside the disclosed SCBSM portfolio range.")
    elif ticket_score >= 12.0:
        reasons.append("The mandate ticket is close to the current SCBSM portfolio range.")
    else:
        cautions.append("The mandate ticket is materially outside the current SCBSM disclosed range.")

    zone_cap_rate = (
        float(same_zone["yield_mid_pct"].mean())
        if not same_zone.empty
        else profile.weighted_cap_rate_pct
    )
    yield_score = max(0.0, 18.0 - 6.0 * abs(float(deal.cap_rate_pct) - zone_cap_rate))
    if yield_score >= 16.0:
        reasons.append("The mandate cap rate is closely aligned with SCBSM's zone-level public yield context.")
    elif yield_score >= 10.0:
        reasons.append("The mandate cap rate is directionally aligned with the public SCBSM yield context.")
    else:
        cautions.append("The mandate cap rate looks wide versus SCBSM's disclosed zone-level yield context.")

    portfolio_cap_rate_score, portfolio_reasons = _portfolio_cap_rate_score(
        deal,
        pd.Series({"current_portfolio_cap_rate_pct": profile.weighted_cap_rate_pct}),
    )
    reasons.extend(portfolio_reasons)

    latest_event = _latest_scbsm_event(events, profile.investor_id)
    latest_outcome = "none"
    latest_event_date: str | None = None
    history_bonus = 0.0
    cooldown_penalty = 0.0
    if latest_event is not None:
        latest_outcome = str(latest_event.get("outcome", "none"))
        latest_event_date = (
            pd.to_datetime(latest_event.get("event_date"), errors="coerce").date().isoformat()
            if pd.notna(latest_event.get("event_date"))
            else None
        )
        history_bonus = LAST_OUTCOME_BONUS.get(latest_outcome, 0.0)
        cooldown_penalty = _cooldown_penalty(_days_since(latest_event_date))
        if cooldown_penalty < 0:
            cautions.append("A recent SCBSM interaction is still inside the cooldown window.")

    overall_score = (
        zone_score
        + asset_score
        + ticket_score
        + yield_score
        + portfolio_cap_rate_score
        + history_bonus
        + cooldown_penalty
    )
    fit_label = _fit_label(overall_score)
    summary, caution_line = _build_match_summary(reasons, cautions)
    explicit_reasons = "\n".join(f"- {item}" for item in reasons + cautions)

    return {
        **profile.as_dict(),
        "zone_match_score": zone_score,
        "asset_focus_score": asset_score,
        "ticket_fit_score": ticket_score,
        "yield_fit_score": yield_score,
        "portfolio_cap_rate_score": portfolio_cap_rate_score,
        "history_bonus": history_bonus,
        "cooldown_penalty": cooldown_penalty,
        "outreach_score": overall_score,
        "fit_label": fit_label,
        "latest_outcome": latest_outcome,
        "latest_event_date": latest_event_date,
        "match_summary": summary,
        "risk_flags": caution_line,
        "explicit_reasons": explicit_reasons,
        "recommended_action": _recommended_action(latest_outcome, _days_since(latest_event_date), fit_label),
        "suggested_pitch": (
            f"Lead with the {deal.zone} {deal.asset_type.lower()} angle, anchor the mandate at "
            f"EUR {deal.ticket_eur_mn:,.1f}m and {deal.cap_rate_pct:.2f}% cap rate, and position it against "
            f"SCBSM's disclosed portfolio footprint."
        ),
        "same_zone_assets_count": int(len(same_zone)),
        "same_asset_assets_count": int(len(same_asset)),
        "same_zone_asset_assets_count": int(len(same_zone_asset)),
        "same_zone_assets_value_eur_mn": float(same_zone["fair_value_eur_mn"].sum()),
        "same_asset_assets_value_eur_mn": float(same_asset["fair_value_eur_mn"].sum()),
        "same_zone_asset_assets_value_eur_mn": float(same_zone_asset["fair_value_eur_mn"].sum()),
        "zone_reference_cap_rate_pct": zone_cap_rate,
    }
