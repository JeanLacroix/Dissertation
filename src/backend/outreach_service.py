from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .outreach_db import (
    append_outreach_event,
    initialize_outreach_db,
    load_assets,
    load_outreach_events,
    load_staged_mandates,
    mark_staged_mandate_loaded,
)
from .outreach_scoring import DealInput, derive_scbsm_profile, score_scbsm_for_deal


DEFAULT_DEAL = DealInput(
    mandate_name="New Paris office mandate",
    asset_type="Office",
    country="France",
    zone="Paris",
    city="Paris",
    ticket_eur_mn=35.0,
    cap_rate_pct=4.75,
    size_sqm=5000.0,
    transaction_date=str(pd.Timestamp.today().date()),
)


@dataclass(frozen=True)
class DashboardContext:
    assets: pd.DataFrame
    events: pd.DataFrame
    staged_mandates: pd.DataFrame
    scbsm_profile: dict[str, Any]
    current_deal: DealInput
    scbsm_evaluation: dict[str, Any]


def bootstrap_outreach_environment(force_reseed: bool = False) -> None:
    initialize_outreach_db(force_reseed=force_reseed)


def build_deal_input(payload: dict[str, Any] | None = None) -> DealInput:
    payload = payload or {}
    values = DEFAULT_DEAL.as_dict() | payload
    transaction_date = values.get("transaction_date") or DEFAULT_DEAL.transaction_date
    return DealInput(
        mandate_name=str(values["mandate_name"]).strip() or DEFAULT_DEAL.mandate_name,
        asset_type=str(values["asset_type"]).strip() or DEFAULT_DEAL.asset_type,
        country=str(values["country"]).strip() or DEFAULT_DEAL.country,
        zone=str(values["zone"]).strip() or DEFAULT_DEAL.zone,
        city=str(values["city"]).strip(),
        ticket_eur_mn=float(values["ticket_eur_mn"]),
        cap_rate_pct=float(values["cap_rate_pct"]),
        size_sqm=float(values["size_sqm"]),
        transaction_date=str(transaction_date),
    )


def load_dashboard_context(deal_input: dict[str, Any] | None = None) -> DashboardContext:
    bootstrap_outreach_environment()
    assets = load_assets()
    events = load_outreach_events()
    staged_mandates = load_staged_mandates()
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
        events=events,
        staged_mandates=staged_mandates,
        scbsm_profile=scbsm_profile,
        current_deal=current_deal,
        scbsm_evaluation=scbsm_evaluation,
    )


def get_scbsm_history(events: pd.DataFrame, investor_id: str = "scbsm") -> pd.DataFrame:
    history = events.loc[events["investor_id"].eq(investor_id)].copy()
    if history.empty:
        return history

    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    history["next_action_date"] = pd.to_datetime(history["next_action_date"], errors="coerce")
    history = history.sort_values("event_date", ascending=False).reset_index(drop=True)
    return history


def build_scbsm_fiche_markdown(
    *,
    deal: DealInput,
    profile: dict[str, Any],
    evaluation: dict[str, Any],
    history: pd.DataFrame,
) -> str:
    history_lines = []
    if history.empty:
        history_lines.append("- No SCBSM follow-up logged yet.")
    else:
        for row in history.head(5).itertuples(index=False):
            if pd.notna(row.next_action_date):
                history_lines.append(
                    f"- {row.event_date:%Y-%m-%d}: {row.channel} / {row.outcome} on `{row.mandate_name}`. "
                    f"Next action: {row.next_action_date:%Y-%m-%d}."
                )
            else:
                history_lines.append(
                    f"- {row.event_date:%Y-%m-%d}: {row.channel} / {row.outcome} on `{row.mandate_name}`."
                )

    risk_line = f"- Watch-outs: {evaluation['risk_flags']}\n" if evaluation.get("risk_flags") else ""

    return (
        f"# SCBSM mandate fit fiche\n\n"
        f"## Mandate in scope\n"
        f"- Mandate: {deal.mandate_name}\n"
        f"- Asset type: {deal.asset_type}\n"
        f"- Geography: {deal.zone} / {deal.city or deal.country}\n"
        f"- Ticket size: EUR {deal.ticket_eur_mn:,.1f}m\n"
        f"- Cap rate estimate: {deal.cap_rate_pct:.2f}%\n\n"
        f"## SCBSM profile\n"
        f"- Company: {profile['company']}\n"
        f"- Title: {profile['title']}\n"
        f"- Zone focus: {profile['zone_focus']}\n"
        f"- Asset focus: {profile['asset_focus']}\n"
        f"- Portfolio assets: {profile['asset_count']}\n"
        f"- Portfolio fair value: EUR {profile['total_fair_value_eur_mn']:,.1f}m\n"
        f"- Ticket range: EUR {profile['min_ticket_eur_mn']:,.1f}m to EUR {profile['max_ticket_eur_mn']:,.1f}m\n"
        f"- Weighted cap rate: {profile['weighted_cap_rate_pct']:.2f}%\n\n"
        f"## Evaluation\n"
        f"- Fit score: {evaluation['outreach_score']:.1f}\n"
        f"- Fit label: {evaluation['fit_label']}\n"
        f"- Match summary: {evaluation['match_summary']}\n"
        f"{risk_line}"
        f"- Suggested pitch: {evaluation['suggested_pitch']}\n"
        f"- Next best action: {evaluation['recommended_action']}\n\n"
        f"## Explicit reasons\n"
        f"{evaluation['explicit_reasons']}\n\n"
        f"## SCBSM interaction history\n"
        + "\n".join(history_lines)
    )


def get_scbsm_fiche(deal_input: dict[str, Any] | None = None) -> dict[str, Any]:
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


def log_follow_up(
    *,
    deal: DealInput,
    event_date: str,
    channel: str,
    outcome: str,
    next_action_date: str | None,
    owner: str,
    notes: str,
) -> str:
    return append_outreach_event(
        investor_id="scbsm",
        mandate_name=deal.mandate_name,
        deal_asset_type=deal.asset_type,
        deal_zone=deal.zone,
        deal_city=deal.city,
        deal_ticket_eur_mn=deal.ticket_eur_mn,
        deal_cap_rate_pct=deal.cap_rate_pct,
        event_date=event_date,
        channel=channel,
        outcome=outcome,
        next_action_date=next_action_date,
        owner=owner,
        notes=notes,
    )


def get_staged_mandate_payload(staged_mandate_id: str) -> dict[str, Any]:
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
        "ticket_eur_mn": float(row["ticket_eur_mn"]),
        "cap_rate_pct": float(row["cap_rate_pct"]),
        "size_sqm": float(row["size_sqm"]),
        "transaction_date": str(row["transaction_date"]),
        "source": row.get("source", ""),
        "notes": row.get("notes", ""),
        "staged_mandate_id": row["staged_mandate_id"],
    }


def load_staged_mandate_into_working_set(staged_mandate_id: str) -> dict[str, Any]:
    payload = get_staged_mandate_payload(staged_mandate_id)
    mark_staged_mandate_loaded(staged_mandate_id)
    return payload
