from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .outreach_db import (
    append_outreach_event,
    initialize_outreach_db,
    load_assets,
    load_contacts,
    load_outreach_events,
)
from .outreach_scoring import score_contacts_for_asset


@dataclass(frozen=True)
class DashboardContext:
    assets: pd.DataFrame
    contacts: pd.DataFrame
    events: pd.DataFrame
    selected_asset: pd.Series
    ranked_contacts: pd.DataFrame


def bootstrap_outreach_environment(force_reseed: bool = False) -> None:
    initialize_outreach_db(force_reseed=force_reseed)


def load_dashboard_context(asset_id: str | None = None) -> DashboardContext:
    bootstrap_outreach_environment()
    assets = load_assets()
    contacts = load_contacts()
    events = load_outreach_events()

    if assets.empty:
        raise RuntimeError("No assets are available for outreach scoring.")

    if asset_id is None or asset_id not in set(assets["asset_id"]):
        selected_asset = assets.sort_values("fair_value_eur", ascending=False).iloc[0]
    else:
        selected_asset = assets.loc[assets["asset_id"].eq(asset_id)].iloc[0]

    ranked = score_contacts_for_asset(
        asset_row=selected_asset,
        contacts=contacts,
        events=events,
    )
    return DashboardContext(
        assets=assets,
        contacts=contacts,
        events=events,
        selected_asset=selected_asset,
        ranked_contacts=ranked,
    )


def get_contact_history(events: pd.DataFrame, assets: pd.DataFrame, contact_id: str) -> pd.DataFrame:
    history = events.loc[events["contact_id"].eq(contact_id)].copy()
    if history.empty:
        return history

    history = history.merge(
        assets[["asset_id", "asset_name", "zone"]],
        how="left",
        on="asset_id",
    )
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    history["next_action_date"] = pd.to_datetime(history["next_action_date"], errors="coerce")
    history = history.sort_values("event_date", ascending=False).reset_index(drop=True)
    return history


def get_contact_fiche(asset_id: str, contact_id: str) -> dict[str, Any]:
    context = load_dashboard_context(asset_id=asset_id)
    ranked = context.ranked_contacts
    if contact_id not in set(ranked["contact_id"]):
        raise KeyError(f"Unknown contact id: {contact_id}")

    recommendation = ranked.loc[ranked["contact_id"].eq(contact_id)].iloc[0]
    contact = context.contacts.loc[context.contacts["contact_id"].eq(contact_id)].iloc[0]
    history = get_contact_history(context.events, context.assets, contact_id)

    return {
        "asset": context.selected_asset.to_dict(),
        "contact": contact.to_dict(),
        "recommendation": recommendation.to_dict(),
        "history": history,
        "fiche_markdown": build_contact_fiche_markdown(
            asset=context.selected_asset,
            recommendation=recommendation,
            history=history,
        ),
    }


def build_contact_fiche_markdown(
    *,
    asset: pd.Series,
    recommendation: pd.Series,
    history: pd.DataFrame,
) -> str:
    history_lines = []
    if history.empty:
        history_lines.append("- No follow-up logged yet.")
    else:
        for row in history.head(5).itertuples(index=False):
            asset_name = getattr(row, "asset_name", "") or "Unspecified asset"
            history_lines.append(
                f"- {row.event_date:%Y-%m-%d}: {row.channel} / {row.outcome} on {asset_name}. "
                f"Next action: {row.next_action_date:%Y-%m-%d}" if pd.notna(row.next_action_date) else
                f"- {row.event_date:%Y-%m-%d}: {row.channel} / {row.outcome} on {asset_name}."
            )

    return (
        f"# Fiche outreach - {recommendation['full_name']}\n\n"
        f"## Contact\n"
        f"- Company: {recommendation['company']}\n"
        f"- Title: {recommendation['title']}\n"
        f"- Zone focus: {recommendation['zone_focus']}\n"
        f"- Asset focus: {recommendation['asset_focus']}\n"
        f"- Preferred channel: {recommendation['preferred_channel']}\n\n"
        f"## Asset in scope\n"
        f"- Asset: {asset['asset_name']}\n"
        f"- Zone: {asset['zone']}\n"
        f"- Fair value: EUR {asset['fair_value_eur_mn']:,.1f}m\n"
        f"- Yield band: {asset['cap_rate_range_pct']}\n\n"
        f"## Recommendation\n"
        f"- Outreach score: {recommendation['outreach_score']:.1f}\n"
        f"- Fit label: {recommendation['fit_label']}\n"
        f"- Suggested pitch: {recommendation['suggested_pitch']}\n"
        f"- Next best action: {recommendation['recommended_action']}\n\n"
        f"## History\n"
        + "\n".join(history_lines)
    )


def log_follow_up(
    *,
    contact_id: str,
    asset_id: str,
    event_date: str,
    channel: str,
    outcome: str,
    next_action_date: str | None,
    owner: str,
    notes: str,
) -> str:
    return append_outreach_event(
        contact_id=contact_id,
        asset_id=asset_id,
        event_date=event_date,
        channel=channel,
        outcome=outcome,
        next_action_date=next_action_date,
        owner=owner,
        notes=notes,
    )
