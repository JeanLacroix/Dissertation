"""Orchestrate outreach data loading, scoring, and persistence workflows."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from .outreach_db import (
    append_outreach_event,
    append_profile_edit,
    initialize_outreach_db,
    load_assets,
    load_contacts,
    load_outreach_events,
    load_profile_edits,
    load_staged_mandates,
    mark_staged_mandate_loaded,
    refresh_seed_assets,
    stage_mandate,
)
from .outreach_scoring import DealInput, derive_scbsm_profile, score_scbsm_for_deal
from .paths import SCBSM_PROFILE_PATH


PROFILE_REQUIRED_FIELDS = [
    "name",
    "firm",
    "investor_type",
    "sector_focus",
    "geographic_focus",
    "ticket_min_eur_mn",
    "ticket_max_eur_mn",
    "portfolio_cap_rate_pct",
    "portfolio_value_eur_mn",
    "ltv_pct",
    "last_updated",
    "source_tag",
]


DEFAULT_DEAL = DealInput(
    mandate_name="New Paris office mandate",
    asset_type="Office",
    country="France",
    zone="Paris",
    city="Paris",
    price_min_eur_mn=35.0,
    price_max_eur_mn=45.0,
    ticket_eur_mn=40.0,
    cap_rate_pct=4.75,
    size_sqm=6000.0,
    transaction_date=str(pd.Timestamp.today().date()),
    noi_eur_mn=None,
    lease_terms="",
    building_grade="",
)


@dataclass(frozen=True)
class DashboardContext:
    """Bundle the data frames required to render the outreach dashboard."""
    assets: pd.DataFrame
    contacts: pd.DataFrame
    events: pd.DataFrame
    staged_mandates: pd.DataFrame
    profile_edits: pd.DataFrame
    scbsm_profile: dict[str, Any]
    current_deal: DealInput
    scbsm_evaluation: dict[str, Any]


def bootstrap_outreach_environment(force_reseed: bool = False) -> None:
    """Bootstrap outreach environment."""
    initialize_outreach_db(force_reseed=force_reseed)


def _clean_optional_float(value: Any, fallback: float | None = None) -> float | None:
    """Clean optional float."""
    if value in {None, ""}:
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


def load_profile_metadata() -> dict[str, Any]:
    """Load profile metadata."""
    if not SCBSM_PROFILE_PATH.exists():
        return {}
    return json.loads(SCBSM_PROFILE_PATH.read_text(encoding="utf-8"))


def build_deal_input(payload: dict[str, Any] | None = None) -> DealInput:
    """Build deal input."""
    payload = payload or {}
    defaults = DEFAULT_DEAL.as_dict()
    values = defaults | payload

    price_min = _clean_optional_float(values.get("price_min_eur_mn"), defaults["price_min_eur_mn"]) or defaults["price_min_eur_mn"]
    price_max = _clean_optional_float(values.get("price_max_eur_mn"), defaults["price_max_eur_mn"]) or defaults["price_max_eur_mn"]
    if price_min > price_max:
        price_min, price_max = price_max, price_min

    if payload and "ticket_eur_mn" not in payload:
        ticket = None
    else:
        ticket = _clean_optional_float(values.get("ticket_eur_mn"))
    if ticket is None:
        ticket = (float(price_min) + float(price_max)) / 2.0

    transaction_date = values.get("transaction_date") or defaults["transaction_date"]
    cap_rate = _clean_optional_float(values.get("cap_rate_pct"))
    noi = _clean_optional_float(values.get("noi_eur_mn"))

    return DealInput(
        mandate_name=str(values["mandate_name"]).strip() or defaults["mandate_name"],
        asset_type=str(values["asset_type"]).strip() or defaults["asset_type"],
        country=str(values["country"]).strip() or defaults["country"],
        zone=str(values["zone"]).strip() or defaults["zone"],
        city=str(values.get("city", "")).strip(),
        price_min_eur_mn=float(price_min),
        price_max_eur_mn=float(price_max),
        ticket_eur_mn=float(ticket),
        cap_rate_pct=cap_rate,
        size_sqm=float(_clean_optional_float(values.get("size_sqm"), defaults["size_sqm"]) or defaults["size_sqm"]),
        transaction_date=str(transaction_date),
        noi_eur_mn=noi,
        lease_terms=str(values.get("lease_terms", "") or ""),
        building_grade=str(values.get("building_grade", "") or ""),
    )


def load_dashboard_context(deal_input: dict[str, Any] | None = None) -> DashboardContext:
    """Load dashboard context."""
    bootstrap_outreach_environment()
    assets = load_assets()
    contacts = load_contacts()
    events = load_outreach_events()
    staged_mandates = load_staged_mandates()
    profile_edits = load_profile_edits()
    current_deal = build_deal_input(deal_input)
    scbsm_profile = derive_scbsm_profile(assets).as_dict()
    scbsm_evaluation = score_scbsm_for_deal(
        deal=current_deal,
        assets=assets,
        events=events,
        profile=derive_scbsm_profile(assets),
    )
    return DashboardContext(
        assets=assets,
        contacts=contacts,
        events=events,
        staged_mandates=staged_mandates,
        profile_edits=profile_edits,
        scbsm_profile=scbsm_profile,
        current_deal=current_deal,
        scbsm_evaluation=scbsm_evaluation,
    )


def get_scbsm_history(events: pd.DataFrame, investor_id: str = "scbsm") -> pd.DataFrame:
    """Return SCBSM history."""
    history = events.loc[events["investor_id"].eq(investor_id)].copy()
    if history.empty:
        return history

    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    history["created_at_utc"] = pd.to_datetime(history["created_at_utc"], errors="coerce")
    history = history.sort_values(["event_date", "created_at_utc", "event_id"], ascending=[False, False, False]).reset_index(drop=True)
    return history


def build_scbsm_fiche_markdown(
    *,
    deal: DealInput,
    profile: dict[str, Any],
    evaluation: dict[str, Any],
    history: pd.DataFrame,
) -> str:
    """Build SCBSM fiche markdown."""
    criteria_lines = []
    for key in ["sector", "geography", "ticket", "cap_rate", "fund_lifecycle"]:
        criterion = evaluation["criteria"][key]
        criteria_lines.append(f"- {criterion['label']}: {criterion['status']} - {criterion['reason']}")

    history_lines = []
    if history.empty:
        history_lines.append("- No SCBSM touchpoints logged yet.")
    else:
        for row in history.head(8).itertuples(index=False):
            history_lines.append(
                f"- {row.event_date:%Y-%m-%d}: {row.touchpoint_type} / {row.status_value} on `{row.mandate_name}`."
            )

    warnings = evaluation.get("warning_messages", [])
    warning_block = "\n".join(f"- {item}" for item in warnings) if warnings else "- No hard targeting warning."

    return (
        "# SCBSM mandate fit fiche\n\n"
        "## Mandate in scope\n"
        f"- Mandate: {deal.mandate_name}\n"
        f"- Asset type: {deal.asset_type}\n"
        f"- Geography: {deal.zone} / {deal.city or deal.country}\n"
        f"- Price range: EUR {deal.price_min_eur_mn:,.1f}m to EUR {deal.price_max_eur_mn:,.1f}m\n"
        f"- Ticket midpoint: EUR {deal.ticket_eur_mn:,.1f}m\n"
        f"- Cap rate estimate: {f'{deal.cap_rate_pct:.2f}%' if deal.cap_rate_pct is not None else 'Not provided'}\n\n"
        "## Investor profile\n"
        f"- Investor: {profile['name']}\n"
        f"- Firm: {profile['firm']}\n"
        f"- Investor type: {profile['investor_type']}\n"
        f"- Sector focus: {', '.join(profile['sector_focus'])}\n"
        f"- Geographic focus: {profile['geographic_focus']}\n"
        f"- Ticket range: EUR {profile['ticket_min_eur_mn']:,.1f}m to EUR {profile['ticket_max_eur_mn']:,.1f}m\n"
        f"- Portfolio cap rate: {profile['portfolio_cap_rate_pct']:.2f}%\n"
        f"- Portfolio value: EUR {profile['portfolio_value_eur_mn']:,.1f}m\n"
        f"- LTV: {profile['ltv_pct']:.2f}%\n"
        f"- Source: {profile['source_tag']} | Updated: {profile['last_updated']}\n\n"
        "## Criterion breakdown\n"
        + "\n".join(criteria_lines)
        + "\n\n## Warnings\n"
        + warning_block
        + "\n\n## Recommendation\n"
        + f"- Fit score: {evaluation['outreach_score']:.1f}\n"
        + f"- Fit label: {evaluation['fit_label']}\n"
        + f"- Yield accretion: {evaluation['yield_accretion_text']}\n"
        + f"- Recommended action: {evaluation['recommended_action']}\n\n"
        + "## Touchpoint history\n"
        + "\n".join(history_lines)
    )


def get_scbsm_fiche(deal_input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return SCBSM fiche."""
    context = load_dashboard_context(deal_input=deal_input)
    history = get_scbsm_history(context.events)
    return {
        "deal": context.current_deal.as_dict(),
        "profile": context.scbsm_profile,
        "evaluation": context.scbsm_evaluation,
        "history": history,
        "fiche_markdown": build_scbsm_fiche_markdown(
            deal=context.current_deal,
            profile=context.scbsm_profile,
            evaluation=context.scbsm_evaluation,
            history=history,
        ),
    }


def _profile_changed_fields(previous: dict[str, Any], updated: dict[str, Any]) -> list[str]:
    """Profile changed fields."""
    changed: list[str] = []
    keys = sorted(set(previous) | set(updated))
    for key in keys:
        if previous.get(key) != updated.get(key):
            changed.append(key)
    return changed


def save_profile_metadata(*, payload: dict[str, Any], edited_by: str, note: str = "") -> list[str]:
    """Save profile metadata."""
    previous = load_profile_metadata()
    changed_fields = _profile_changed_fields(previous, payload)
    if not changed_fields:
        return []
    SCBSM_PROFILE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    append_profile_edit(
        investor_id=str(payload.get("investor_id", "scbsm")),
        edited_at_utc=pd.Timestamp.now("UTC").isoformat(),
        edited_by=edited_by,
        changed_fields=changed_fields,
        note=note,
    )
    return changed_fields


def validate_profile_payload(payload: dict[str, Any]) -> list[str]:
    """Validate profile payload."""
    missing: list[str] = []
    for field in PROFILE_REQUIRED_FIELDS:
        value = payload.get(field)
        if isinstance(value, list):
            if not any(str(item).strip() for item in value):
                missing.append(field)
        elif value in {None, ""}:
            missing.append(field)
    return missing


def refresh_scbsm_profile_from_public_data(*, edited_by: str) -> dict[str, Any]:
    """Refresh SCBSM profile from public data."""
    previous = load_profile_metadata()
    refresh_seed_assets()
    refreshed = previous.copy()
    portfolio_value = refreshed.get("portfolio_value_eur_mn")
    revenues = refreshed.get("revenues_eur_mn")
    if portfolio_value in {None, ""} or revenues in {None, ""}:
        raise ValueError("Refresh failed because portfolio value or revenues are missing from the current profile snapshot.")

    cap_rate_raw = float(revenues) / float(portfolio_value) * 100.0
    refreshed["portfolio_cap_rate_pct"] = int(cap_rate_raw * 100.0) / 100.0
    refreshed["last_updated"] = str(pd.Timestamp.today().date())
    refreshed["source_tag"] = "Auto - scbsm.fr"
    save_profile_metadata(payload=refreshed, edited_by=edited_by, note="Refreshed from public SCBSM data snapshot.")
    return refreshed


def log_touchpoint(
    *,
    deal: DealInput,
    event_date: str,
    touchpoint_type: str,
    status_value: str,
    owner: str,
    notes: str,
) -> str:
    """Log touchpoint."""
    event_dt = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_dt):
        raise ValueError("Invalid event date.")
    backdated_flag = (pd.Timestamp.today().normalize() - event_dt.normalize()).days > 90
    return append_outreach_event(
        investor_id="scbsm",
        mandate_name=deal.mandate_name,
        deal_asset_type=deal.asset_type,
        deal_zone=deal.zone,
        deal_city=deal.city,
        price_min_eur_mn=deal.price_min_eur_mn,
        price_max_eur_mn=deal.price_max_eur_mn,
        deal_ticket_eur_mn=deal.ticket_eur_mn,
        deal_cap_rate_pct=deal.cap_rate_pct,
        event_date=event_date,
        touchpoint_type=touchpoint_type,
        status_value=status_value,
        owner=owner,
        notes=notes,
        created_at_utc=pd.Timestamp.now("UTC").isoformat(),
        backdated_flag=backdated_flag,
    )


def log_override_confirmation(*, deal: DealInput, owner: str, notes: str = "") -> str:
    """Log override confirmation."""
    return append_outreach_event(
        investor_id="scbsm",
        mandate_name=deal.mandate_name,
        deal_asset_type=deal.asset_type,
        deal_zone=deal.zone,
        deal_city=deal.city,
        price_min_eur_mn=deal.price_min_eur_mn,
        price_max_eur_mn=deal.price_max_eur_mn,
        deal_ticket_eur_mn=deal.ticket_eur_mn,
        deal_cap_rate_pct=deal.cap_rate_pct,
        event_date=date.today().isoformat(),
        touchpoint_type="criteria_override",
        status_value="confirmed",
        owner=owner,
        notes=notes,
        created_at_utc=pd.Timestamp.now("UTC").isoformat(),
        backdated_flag=False,
    )


def get_staged_mandate_payload(staged_mandate_id: str) -> dict[str, Any]:
    """Return staged mandate payload."""
    staged = load_staged_mandates()
    if staged.empty:
        raise KeyError(f"Unknown staged mandate id: {staged_mandate_id}")
    match = staged.loc[staged["staged_mandate_id"].eq(staged_mandate_id)]
    if match.empty:
        raise KeyError(f"Unknown staged mandate id: {staged_mandate_id}")
    row = match.iloc[0]
    return {
        "mandate_name": row["mandate_name"],
        "asset_type": row["asset_type"],
        "country": row["country"],
        "zone": row["zone"],
        "city": row["city"],
        "price_min_eur_mn": _clean_optional_float(row.get("price_min_eur_mn")),
        "price_max_eur_mn": _clean_optional_float(row.get("price_max_eur_mn")),
        "ticket_eur_mn": float(row["ticket_eur_mn"]),
        "cap_rate_pct": _clean_optional_float(row.get("cap_rate_pct")),
        "size_sqm": float(row["size_sqm"]),
        "transaction_date": str(row["transaction_date"]),
        "noi_eur_mn": _clean_optional_float(row.get("noi_eur_mn")),
        "lease_terms": row.get("lease_terms", ""),
        "building_grade": row.get("building_grade", ""),
        "source": row.get("source", ""),
        "notes": row.get("notes", ""),
        "staged_mandate_id": row["staged_mandate_id"],
    }


def load_staged_mandate_into_working_set(staged_mandate_id: str) -> dict[str, Any]:
    """Load staged mandate into working set."""
    payload = get_staged_mandate_payload(staged_mandate_id)
    mark_staged_mandate_loaded(staged_mandate_id)
    return payload


def create_mock_mandate(*, payload: dict[str, Any], lead_banker: str = "") -> str:
    """Create mock mandate."""
    deal = build_deal_input(payload)
    note = "Created from Streamlit UI."
    if lead_banker.strip():
        note = f"{note} Lead banker: {lead_banker.strip()}."

    staged_mandate_id = stage_mandate(
        mandate_name=deal.mandate_name,
        asset_type=deal.asset_type,
        country=deal.country,
        zone=deal.zone,
        city=deal.city,
        price_min_eur_mn=deal.price_min_eur_mn,
        price_max_eur_mn=deal.price_max_eur_mn,
        ticket_eur_mn=deal.ticket_eur_mn,
        cap_rate_pct=deal.cap_rate_pct,
        size_sqm=deal.size_sqm,
        transaction_date=deal.transaction_date,
        noi_eur_mn=deal.noi_eur_mn,
        lease_terms=deal.lease_terms,
        building_grade=deal.building_grade,
        source="streamlit_mock",
        notes=note,
    )
    mark_staged_mandate_loaded(staged_mandate_id)
    return staged_mandate_id
