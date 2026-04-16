from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .paths import SCBSM_PROFILE_PATH


def _today_iso() -> str:
    return pd.Timestamp.today().date().isoformat()


@dataclass(frozen=True)
class DealInput:
    mandate_name: str
    asset_type: str
    country: str
    zone: str
    city: str
    price_min_eur_mn: float
    price_max_eur_mn: float
    ticket_eur_mn: float
    cap_rate_pct: float | None
    size_sqm: float
    transaction_date: str
    noi_eur_mn: float | None = None
    lease_terms: str = ""
    building_grade: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScbsmProfile:
    investor_id: str
    name: str
    firm: str
    company: str
    investor_type: str
    display_name: str
    title: str
    sector_focus: list[str] = field(default_factory=list)
    geographic_focus: str = ""
    city_focus: str = ""
    country_focus: str = ""
    ticket_min_eur_mn: float = 0.0
    ticket_max_eur_mn: float = 0.0
    portfolio_cap_rate_pct: float = 0.0
    weighted_asset_yield_pct: float = 0.0
    portfolio_value_eur_mn: float = 0.0
    ltv_pct: float | None = None
    rental_value_eur_mn: float | None = None
    revenues_eur_mn: float | None = None
    net_debt_eur_mn: float | None = None
    equity_eur_mn: float | None = None
    last_updated: str = ""
    source_tag: str = ""
    preferred_channel: str = "email"
    owner: str = ""
    qualitative_focus: str = ""
    notes: str = ""
    fund_vintage_year: int | None = None
    fund_life_years: float | None = None
    lifecycle_override_note: str = ""
    marchand_de_bien: bool = False
    acquisition_date: str = ""
    resale_commitment_years: int = 5
    resale_override_note: str = ""
    asset_count: int = 0
    top_zone: str = ""
    top_asset_class: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: Any) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _slug_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _to_float(value: Any, fallback: float | None = None) -> float | None:
    if value is None or value == "":
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or fallback
    if isinstance(value, str) and value.strip():
        parts = [part.strip() for part in re.split(r"[;,/|]", value) if part.strip()]
        return parts or fallback
    return fallback


def _load_scbsm_profile_metadata() -> dict[str, Any]:
    if not SCBSM_PROFILE_PATH.exists():
        return {
            "investor_id": "scbsm",
            "name": "SCBSM",
            "firm": "SCBSM",
            "company": "SCBSM",
            "investor_type": "Listed Company",
            "display_name": "SCBSM",
            "title": "Listed French real estate company",
            "sector_focus": ["Office"],
            "geographic_focus": "Paris intramuros",
            "city_focus": "Paris",
            "country_focus": "France",
            "ticket_min_eur_mn": 15.0,
            "ticket_max_eur_mn": 50.0,
            "portfolio_cap_rate_pct": 4.74,
            "portfolio_value_eur_mn": 473.8,
            "ltv_pct": 38.84,
            "rental_value_eur_mn": 31.4,
            "revenues_eur_mn": 22.5,
            "net_debt_eur_mn": 184.1,
            "equity_eur_mn": 289.7,
            "last_updated": _today_iso(),
            "source_tag": "Auto - scbsm.fr",
            "preferred_channel": "email",
            "owner": "Jean",
            "qualitative_focus": "Prototype investor profile anchored on public SCBSM disclosures.",
            "notes": "",
            "marchand_de_bien": False,
            "resale_commitment_years": 5,
        }
    return json.loads(SCBSM_PROFILE_PATH.read_text(encoding="utf-8"))


def compute_fund_lifecycle_status(profile: ScbsmProfile) -> dict[str, Any]:
    investor_type = _slug_text(profile.investor_type)
    if "fund" not in investor_type:
        return {
            "status": "Not applicable",
            "is_likely_seller": False,
            "years_remaining": None,
            "reason": "SCBSM is configured as a listed company, not a closed-end fund.",
        }

    if profile.fund_vintage_year is None or profile.fund_life_years is None:
        return {
            "status": "Incomplete",
            "is_likely_seller": False,
            "years_remaining": None,
            "reason": "Fund lifecycle fields are incomplete, so no seller flag can be computed yet.",
        }

    years_remaining = float(profile.fund_life_years) - (date.today().year - int(profile.fund_vintage_year))
    if years_remaining <= 2:
        return {
            "status": "Likely seller",
            "is_likely_seller": True,
            "years_remaining": years_remaining,
            "reason": f"Estimated remaining fund life is {years_remaining:.1f} years, which falls inside the two-year seller flag threshold.",
        }

    return {
        "status": "Acquisition-capable",
        "is_likely_seller": False,
        "years_remaining": years_remaining,
        "reason": f"Estimated remaining fund life is {years_remaining:.1f} years.",
    }


def compute_resale_deadline_status(profile: ScbsmProfile) -> dict[str, Any]:
    if not bool(profile.marchand_de_bien):
        return {
            "status": "Not applicable",
            "deadline": None,
            "months_remaining": None,
            "reason": "SCBSM is not configured as a marchand-de-bien in this prototype.",
        }

    acquisition_date = _parse_date(profile.acquisition_date)
    if acquisition_date is None:
        return {
            "status": "Incomplete",
            "deadline": None,
            "months_remaining": None,
            "reason": "Acquisition date is missing, so the resale deadline cannot be computed.",
        }

    deadline = acquisition_date + pd.DateOffset(years=int(profile.resale_commitment_years))
    deadline = pd.Timestamp(deadline).date()
    months_remaining = (deadline.year - date.today().year) * 12 + (deadline.month - date.today().month)
    if months_remaining <= 18:
        return {
            "status": "Resale deadline approaching",
            "deadline": deadline.isoformat(),
            "months_remaining": months_remaining,
            "reason": f"The mandatory resale deadline falls on {deadline.isoformat()}, inside the 18-month alert window.",
        }

    return {
        "status": "No near-term resale trigger",
        "deadline": deadline.isoformat(),
        "months_remaining": months_remaining,
        "reason": f"The mandatory resale deadline falls on {deadline.isoformat()}.",
    }


def derive_scbsm_profile(assets: pd.DataFrame) -> ScbsmProfile:
    metadata = _load_scbsm_profile_metadata()
    portfolio = assets.copy()
    portfolio["fair_value_eur_mn"] = pd.to_numeric(portfolio["fair_value_eur_mn"], errors="coerce").fillna(0.0)
    portfolio["yield_mid_pct"] = pd.to_numeric(portfolio["yield_mid_pct"], errors="coerce")

    total_fair_value = float(portfolio["fair_value_eur_mn"].sum())
    weighted_asset_yield = (
        float((portfolio["fair_value_eur_mn"] * portfolio["yield_mid_pct"].fillna(0.0)).sum() / total_fair_value)
        if total_fair_value > 0
        else float(portfolio["yield_mid_pct"].dropna().mean())
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

    portfolio_value = _to_float(metadata.get("portfolio_value_eur_mn"), total_fair_value) or total_fair_value
    revenues = _to_float(metadata.get("revenues_eur_mn"))
    implied_cap_rate = (revenues / portfolio_value * 100.0) if revenues and portfolio_value else weighted_asset_yield
    portfolio_cap_rate = _to_float(metadata.get("portfolio_cap_rate_pct"), implied_cap_rate) or implied_cap_rate

    return ScbsmProfile(
        investor_id=str(metadata.get("investor_id", "scbsm")),
        name=str(metadata.get("name") or metadata.get("company") or "SCBSM"),
        firm=str(metadata.get("firm") or metadata.get("company") or "SCBSM"),
        company=str(metadata.get("company") or metadata.get("firm") or "SCBSM"),
        investor_type=str(metadata.get("investor_type", "Listed Company")),
        display_name=str(metadata.get("display_name") or metadata.get("company") or "SCBSM"),
        title=str(metadata.get("title", "SCBSM investor profile")),
        sector_focus=_string_list(metadata.get("sector_focus"), ["Office"]),
        geographic_focus=str(metadata.get("geographic_focus", "Paris intramuros")),
        city_focus=str(metadata.get("city_focus", "Paris")),
        country_focus=str(metadata.get("country_focus", "France")),
        ticket_min_eur_mn=float(_to_float(metadata.get("ticket_min_eur_mn"), 15.0) or 15.0),
        ticket_max_eur_mn=float(_to_float(metadata.get("ticket_max_eur_mn"), 50.0) or 50.0),
        portfolio_cap_rate_pct=float(portfolio_cap_rate),
        weighted_asset_yield_pct=float(weighted_asset_yield),
        portfolio_value_eur_mn=float(portfolio_value),
        ltv_pct=_to_float(metadata.get("ltv_pct")),
        rental_value_eur_mn=_to_float(metadata.get("rental_value_eur_mn")),
        revenues_eur_mn=revenues,
        net_debt_eur_mn=_to_float(metadata.get("net_debt_eur_mn")),
        equity_eur_mn=_to_float(metadata.get("equity_eur_mn")),
        last_updated=str(metadata.get("last_updated", _today_iso())),
        source_tag=str(metadata.get("source_tag", "Auto - scbsm.fr")),
        preferred_channel=str(metadata.get("preferred_channel", "email")),
        owner=str(metadata.get("owner", "Jean")),
        qualitative_focus=str(metadata.get("qualitative_focus", "")),
        notes=str(metadata.get("notes", "")),
        fund_vintage_year=int(metadata["fund_vintage_year"]) if metadata.get("fund_vintage_year") not in {None, ""} else None,
        fund_life_years=_to_float(metadata.get("fund_life_years")),
        lifecycle_override_note=str(metadata.get("lifecycle_override_note", "")),
        marchand_de_bien=bool(metadata.get("marchand_de_bien", False)),
        acquisition_date=str(metadata.get("acquisition_date", "")),
        resale_commitment_years=int(metadata.get("resale_commitment_years", 5) or 5),
        resale_override_note=str(metadata.get("resale_override_note", "")),
        asset_count=int(len(portfolio)),
        top_zone=str(zone_mix.index[0]) if not zone_mix.empty else "",
        top_asset_class=str(asset_mix.index[0]) if not asset_mix.empty else "",
    )


def _build_sector_criterion(deal: DealInput, profile: ScbsmProfile, same_asset_count: int) -> dict[str, Any]:
    focus = profile.sector_focus or ["Office"]
    focus_tokens = {_slug_text(item) for item in focus}
    asset_token = _slug_text(deal.asset_type)
    match = asset_token in focus_tokens
    if match:
        reason = (
            f"Sector match: `{deal.asset_type}` sits inside SCBSM's stated focus ({', '.join(focus)}). "
            f"The disclosed portfolio also contains {same_asset_count} comparable asset(s)."
        )
    else:
        reason = f"Sector mismatch: asset type = {deal.asset_type} while SCBSM focus is {', '.join(focus)}."
    return {
        "label": "Sector match",
        "status": "Yes" if match else "No",
        "match": match,
        "weight": 30.0,
        "earned": 30.0 if match else 0.0,
        "reason": reason,
    }


def _build_geography_criterion(deal: DealInput, profile: ScbsmProfile, same_city_count: int, same_zone_count: int) -> dict[str, Any]:
    city_token = _slug_text(deal.city)
    zone_token = _slug_text(deal.zone)
    focus_token = _slug_text(profile.geographic_focus)
    city_focus_token = _slug_text(profile.city_focus)
    country_match = _slug_text(deal.country) == _slug_text(profile.country_focus)
    paris_focus = "paris" in focus_token or "paris" in city_focus_token
    exact_match = country_match and (city_token == "paris" or zone_token == "paris")

    if paris_focus and exact_match:
        reason = (
            f"Geography match: mandate sits in Paris, which matches SCBSM's `{profile.geographic_focus}` focus. "
            f"The disclosed portfolio shows {same_city_count or same_zone_count} Paris asset(s)."
        )
        match = True
    elif country_match and same_zone_count > 0:
        reason = (
            f"Partial geography fit: mandate is within France and overlaps with SCBSM's disclosed footprint, "
            f"but the stated target geography remains `{profile.geographic_focus}`."
        )
        match = False
    else:
        location_label = deal.city or deal.zone or deal.country
        reason = f"Geography mismatch: location = {location_label} while SCBSM focus is `{profile.geographic_focus}`."
        match = False

    return {
        "label": "Geography match",
        "status": "Yes" if match else "No",
        "match": match,
        "weight": 25.0,
        "earned": 25.0 if match else 0.0,
        "reason": reason,
    }


def _build_ticket_criterion(deal: DealInput, profile: ScbsmProfile) -> dict[str, Any]:
    match = profile.ticket_min_eur_mn <= deal.ticket_eur_mn <= profile.ticket_max_eur_mn
    if match:
        reason = (
            f"Ticket range match: EUR {deal.ticket_eur_mn:,.1f}m sits within SCBSM's stated range of "
            f"EUR {profile.ticket_min_eur_mn:,.1f}m to EUR {profile.ticket_max_eur_mn:,.1f}m."
        )
    elif deal.ticket_eur_mn < profile.ticket_min_eur_mn:
        reason = (
            f"Ticket mismatch: EUR {deal.ticket_eur_mn:,.1f}m is below SCBSM's minimum ticket of "
            f"EUR {profile.ticket_min_eur_mn:,.1f}m."
        )
    else:
        reason = (
            f"Ticket mismatch: EUR {deal.ticket_eur_mn:,.1f}m is above SCBSM's maximum ticket of "
            f"EUR {profile.ticket_max_eur_mn:,.1f}m."
        )
    return {
        "label": "Ticket range match",
        "status": "Yes" if match else "No",
        "match": match,
        "weight": 20.0,
        "earned": 20.0 if match else 0.0,
        "reason": reason,
    }


def _build_cap_rate_criterion(deal: DealInput, profile: ScbsmProfile) -> dict[str, Any]:
    if deal.cap_rate_pct is None:
        return {
            "label": "Cap rate compatibility",
            "status": "Not assessed",
            "match": None,
            "weight": 0.0,
            "earned": 0.0,
            "reason": "No cap rate was entered, so yield compatibility and accretion were not assessed.",
            "yield_accretive": None,
            "yield_accretion_text": "No cap rate provided.",
        }

    gap = float(deal.cap_rate_pct) - float(profile.portfolio_cap_rate_pct)
    compatible = abs(gap) <= 1.0
    yield_accretive = gap >= 0
    if compatible:
        reason = (
            f"Cap rate is compatible: deal implied yield {deal.cap_rate_pct:.2f}% versus SCBSM portfolio "
            f"cap rate {profile.portfolio_cap_rate_pct:.2f}%."
        )
    else:
        reason = (
            f"Cap rate looks stretched: deal implied yield {deal.cap_rate_pct:.2f}% versus SCBSM portfolio "
            f"cap rate {profile.portfolio_cap_rate_pct:.2f}%."
        )

    if yield_accretive:
        yield_text = (
            f"Yield-accretive: the deal cap rate ({deal.cap_rate_pct:.2f}%) is at or above "
            f"SCBSM's current portfolio cap rate ({profile.portfolio_cap_rate_pct:.2f}%)."
        )
    else:
        yield_text = (
            f"Yield-dilutive: the deal cap rate ({deal.cap_rate_pct:.2f}%) is below "
            f"SCBSM's current portfolio cap rate ({profile.portfolio_cap_rate_pct:.2f}%)."
        )

    return {
        "label": "Cap rate compatibility",
        "status": "Yes" if compatible else "No",
        "match": compatible,
        "weight": 15.0,
        "earned": 15.0 if compatible else 0.0,
        "reason": reason,
        "yield_accretive": yield_accretive,
        "yield_accretion_text": yield_text,
    }


def _build_lifecycle_criterion(profile: ScbsmProfile) -> dict[str, Any]:
    lifecycle = compute_fund_lifecycle_status(profile)
    status = lifecycle["status"]
    if status == "Likely seller":
        earned = 0.0
        match = False
    else:
        earned = 10.0
        match = None if status == "Not applicable" else True
    return {
        "label": "Fund lifecycle status",
        "status": status,
        "match": match,
        "weight": 10.0,
        "earned": earned,
        "reason": lifecycle["reason"],
        "years_remaining": lifecycle["years_remaining"],
        "is_likely_seller": lifecycle["is_likely_seller"],
    }


def _fit_label(score: float) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Possible match"
    return "Do not target"


def _latest_scbsm_event(events: pd.DataFrame, investor_id: str) -> pd.Series | None:
    if events.empty or "investor_id" not in events.columns:
        return None
    frame = events.loc[events["investor_id"].eq(investor_id)].copy()
    if frame.empty:
        return None
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["created_at_utc"] = pd.to_datetime(frame["created_at_utc"], errors="coerce")
    frame = frame.sort_values(["event_date", "created_at_utc", "event_id"], ascending=[False, False, False])
    return frame.iloc[0]


def score_scbsm_for_deal(
    *,
    deal: DealInput,
    assets: pd.DataFrame,
    events: pd.DataFrame,
    profile: ScbsmProfile,
) -> dict[str, Any]:
    portfolio = assets.copy()
    portfolio["asset_class"] = portfolio["asset_class"].fillna("").astype(str)
    portfolio["zone"] = portfolio["zone"].fillna("").astype(str)
    portfolio["city"] = portfolio["city"].fillna("").astype(str)
    portfolio["fair_value_eur_mn"] = pd.to_numeric(portfolio["fair_value_eur_mn"], errors="coerce").fillna(0.0)
    portfolio["yield_mid_pct"] = pd.to_numeric(portfolio["yield_mid_pct"], errors="coerce")

    zone_token = _slug_text(deal.zone)
    city_token = _slug_text(deal.city)
    asset_token = _slug_text(deal.asset_type)
    same_zone = portfolio.loc[portfolio["zone"].map(_slug_text).eq(zone_token)].copy()
    same_city = portfolio.loc[portfolio["city"].map(_slug_text).eq(city_token)].copy()
    same_asset = portfolio.loc[portfolio["asset_class"].map(_slug_text).eq(asset_token)].copy()
    same_zone_asset = same_zone.loc[same_zone["asset_class"].map(_slug_text).eq(asset_token)].copy()

    sector = _build_sector_criterion(deal, profile, same_asset_count=int(len(same_asset)))
    geography = _build_geography_criterion(
        deal,
        profile,
        same_city_count=int(len(same_city)),
        same_zone_count=int(len(same_zone)),
    )
    ticket = _build_ticket_criterion(deal, profile)
    cap_rate = _build_cap_rate_criterion(deal, profile)
    lifecycle = _build_lifecycle_criterion(profile)
    criteria = {
        "sector": sector,
        "geography": geography,
        "ticket": ticket,
        "cap_rate": cap_rate,
        "fund_lifecycle": lifecycle,
    }

    weighted_score = sum(float(item["earned"]) for item in criteria.values())
    total_weight = sum(float(item["weight"]) for item in criteria.values())
    overall_score = round((weighted_score / total_weight) * 100.0, 1) if total_weight else 0.0
    fit_label = _fit_label(overall_score)

    warnings: list[str] = []
    if not sector["match"]:
        warnings.append(f"Asset type = {deal.asset_type} - SCBSM invests in {', '.join(profile.sector_focus)} only.")
    if not geography["match"]:
        location_label = deal.city or deal.zone or deal.country
        warnings.append(f"Location = {location_label} - SCBSM targets {profile.geographic_focus}.")
    if not ticket["match"]:
        if deal.ticket_eur_mn < profile.ticket_min_eur_mn:
            warnings.append(f"Ticket = EUR {deal.ticket_eur_mn:,.1f}m - below SCBSM minimum of EUR {profile.ticket_min_eur_mn:,.1f}m.")
        else:
            warnings.append(f"Ticket = EUR {deal.ticket_eur_mn:,.1f}m - above SCBSM maximum of EUR {profile.ticket_max_eur_mn:,.1f}m.")
    if lifecycle["status"] == "Likely seller":
        warnings.append("Fund lifecycle flag - investor is within two years of end-of-life and may be a seller.")

    summary_bits = [
        sector["status"],
        geography["status"],
        ticket["status"],
        cap_rate["status"],
        lifecycle["status"],
    ]
    match_summary = (
        f"Sector {summary_bits[0]} | Geography {summary_bits[1]} | Ticket {summary_bits[2]} | "
        f"Cap rate {summary_bits[3]} | Lifecycle {summary_bits[4]}"
    )
    explicit_reasons = "\n".join(f"- {criterion['label']}: {criterion['reason']}" for criterion in criteria.values())

    latest_event = _latest_scbsm_event(events, profile.investor_id)
    latest_touchpoint = ""
    latest_status = ""
    latest_event_date = None
    if latest_event is not None:
        latest_touchpoint = str(latest_event.get("touchpoint_type", ""))
        latest_status = str(latest_event.get("status_value", ""))
        latest_date = pd.to_datetime(latest_event.get("event_date"), errors="coerce")
        latest_event_date = latest_date.date().isoformat() if pd.notna(latest_date) else None

    off_target = bool(warnings)
    if off_target:
        recommended_action = "Do not include in outreach unless the analyst records an override rationale."
    elif fit_label == "Strong match":
        recommended_action = "SCBSM belongs in the first outreach wave for this mandate."
    else:
        recommended_action = "Keep SCBSM in the review list, but validate the weaker criteria before sending a teaser."

    lifecycle_status = compute_fund_lifecycle_status(profile)
    resale_status = compute_resale_deadline_status(profile)

    return {
        **profile.as_dict(),
        "criteria": criteria,
        "outreach_score": overall_score,
        "fit_label": fit_label,
        "match_summary": match_summary,
        "explicit_reasons": explicit_reasons,
        "warning_messages": warnings,
        "risk_flags": " | ".join(warnings),
        "recommended_action": recommended_action,
        "yield_accretive": cap_rate.get("yield_accretive"),
        "yield_accretion_text": cap_rate["yield_accretion_text"],
        "latest_touchpoint": latest_touchpoint,
        "latest_status": latest_status,
        "latest_event_date": latest_event_date,
        "same_zone_assets_count": int(len(same_zone)),
        "same_city_assets_count": int(len(same_city)),
        "same_asset_assets_count": int(len(same_asset)),
        "same_zone_asset_assets_count": int(len(same_zone_asset)),
        "same_zone_assets_value_eur_mn": float(same_zone["fair_value_eur_mn"].sum()),
        "same_asset_assets_value_eur_mn": float(same_asset["fair_value_eur_mn"].sum()),
        "same_zone_asset_assets_value_eur_mn": float(same_zone_asset["fair_value_eur_mn"].sum()),
        "zone_reference_cap_rate_pct": float(same_zone["yield_mid_pct"].dropna().mean()) if not same_zone.empty else profile.weighted_asset_yield_pct,
        "hard_mismatch_count": len(warnings),
        "off_target_warning": off_target,
        "lifecycle_status": lifecycle_status,
        "resale_status": resale_status,
    }
