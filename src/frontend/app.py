from __future__ import annotations

import io
from datetime import date

import pandas as pd
import streamlit as st

from src.backend.outreach_service import (
    bootstrap_outreach_environment,
    get_contact_fiche,
    get_contact_history,
    load_dashboard_context,
    log_follow_up,
)
from src.backend.paths import YIELD_EXTRACTION_NOTE_PATH

TOP_K_DEFAULT = 8


def _csv_download_bytes(frame: pd.DataFrame) -> bytes:
    output = io.StringIO()
    frame.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def _safe_choice(options: list[str], preferred: str | None) -> str:
    if preferred in options:
        return preferred
    return options[0]


def _render_asset_summary(asset: pd.Series) -> None:
    st.subheader("Selected asset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fair value", f"EUR {asset['fair_value_eur_mn']:,.1f}m")
    col2.metric("Yield midpoint", f"{asset['yield_mid_pct']:.2f}%")
    col3.metric("Yield band", asset["cap_rate_range_pct"])
    col4.metric("Zone", asset["zone"])

    with st.expander("Asset context", expanded=False):
        st.write(
            f"**{asset['asset_name']}** is tagged as `{asset['asset_class']}` in `{asset['investment_profile']}`. "
            f"Valuation date: `{asset['valuation_date']}`. Last expert visit: `{asset['last_visit_date']}`."
        )
        st.write(
            "The yield used by the ranking engine is the zone-level capitalisation band disclosed in the SCBSM URD, "
            "not an asset-specific expert yield."
        )


def _render_ranked_contacts(ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
    st.subheader("Algorithm recommendations")
    display = ranked.head(top_k).copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    display = display[
        [
            "rank",
            "contact_id",
            "full_name",
            "company",
            "title",
            "outreach_score",
            "fit_label",
            "zone_focus",
            "asset_focus",
            "days_since_last_touch",
            "recommended_action",
        ]
    ].rename(
        columns={
            "rank": "#",
            "contact_id": "Contact ID",
            "full_name": "Name",
            "company": "Company",
            "title": "Title",
            "outreach_score": "Score",
            "fit_label": "Fit",
            "zone_focus": "Zone focus",
            "asset_focus": "Asset focus",
            "days_since_last_touch": "Days since touch",
            "recommended_action": "Next best action",
        }
    )
    display["Score"] = display["Score"].round(1)
    st.dataframe(display, use_container_width=True, hide_index=True)
    return display


def _render_contact_fiche(asset_id: str, contact_id: str) -> dict[str, object]:
    fiche = get_contact_fiche(asset_id=asset_id, contact_id=contact_id)
    recommendation = pd.Series(fiche["recommendation"])
    contact = pd.Series(fiche["contact"])
    history = fiche["history"]

    st.subheader("Fiche outreach")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Outreach score", f"{recommendation['outreach_score']:.1f}")
    col2.metric("Yield fit", f"{recommendation['yield_fit_score']:.1f}")
    col3.metric("Ticket fit", f"{recommendation['ticket_fit_score']:.1f}")
    col4.metric("Geo fit", f"{recommendation['zone_match_score']:.1f}")

    st.write(
        f"**{contact['full_name']}** | {contact['title']} at **{contact['company']}**"
    )
    st.caption(
        f"{contact['city']} | zone focus `{contact['zone_focus']}` | asset focus `{contact['asset_focus']}` | "
        f"preferred channel `{contact['preferred_channel']}`"
    )
    st.write(recommendation["suggested_pitch"])

    with st.expander("Full contact profile", expanded=True):
        st.write(f"Email: `{contact['email']}`")
        st.write(
            "Ticket range: "
            f"EUR {contact['min_ticket_eur_mn']:,.0f}m to EUR {contact['max_ticket_eur_mn']:,.0f}m"
        )
        st.write(
            "Target yield range: "
            f"{contact['min_target_yield_pct']:.2f}% to {contact['max_target_yield_pct']:.2f}%"
        )
        st.write(f"Relationship stage: `{contact['relationship_stage']}`")
        st.write(f"Owner: `{contact['owner']}`")
        st.write(f"Notes: {contact['notes']}")

    with st.expander("Score breakdown", expanded=False):
        st.write(
            f"- Zone match: {recommendation['zone_match_score']:.1f}\n"
            f"- Asset focus: {recommendation['asset_focus_score']:.1f}\n"
            f"- Ticket fit: {recommendation['ticket_fit_score']:.1f}\n"
            f"- Yield fit: {recommendation['yield_fit_score']:.1f}\n"
            f"- Relationship stage bonus: {recommendation['relationship_stage_bonus']:.1f}\n"
            f"- Response bonus: {recommendation['response_bonus']:.1f}\n"
            f"- Strategic priority bonus: {recommendation['priority_bonus']:.1f}\n"
            f"- Outcome bonus: {recommendation['outcome_bonus']:.1f}\n"
            f"- Cooldown penalty: {recommendation['cooldown_penalty']:.1f}"
        )

    st.download_button(
        label="Download fiche outreach",
        data=str(fiche["fiche_markdown"]).encode("utf-8"),
        file_name=f"{contact_id}_fiche_outreach.md",
        mime="text/markdown",
    )

    st.markdown("**Follow-up history**")
    if isinstance(history, pd.DataFrame) and not history.empty:
        display = history[
            ["event_date", "channel", "outcome", "asset_name", "next_action_date", "notes"]
        ].rename(
            columns={
                "event_date": "Event date",
                "channel": "Channel",
                "outcome": "Outcome",
                "asset_name": "Asset",
                "next_action_date": "Next action",
                "notes": "Notes",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No follow-up has been logged for this contact yet.")

    return fiche


def _render_follow_up_form(context, default_contact_id: str) -> None:
    st.subheader("Contact follow-up")
    history = get_contact_history(context.events, context.assets, default_contact_id)
    if not history.empty:
        st.caption("Latest follow-ups for the selected contact are shown in the fiche above.")

    contact_options = {
        f"{row.full_name} ({row.company})": row.contact_id
        for row in context.contacts.itertuples(index=False)
    }
    contact_labels = list(contact_options.keys())
    default_label = next(
        (label for label, contact_id in contact_options.items() if contact_id == default_contact_id),
        contact_labels[0],
    )

    with st.form("follow_up_form"):
        contact_label = st.selectbox(
            "Contact",
            options=contact_labels,
            index=contact_labels.index(default_label),
        )
        channel = st.selectbox("Channel", options=["email", "phone", "meeting", "linkedin"])
        outcome = st.selectbox("Outcome", options=["positive", "neutral", "no_reply", "not_now"])
        event_date = st.date_input("Event date", value=date.today())
        use_next_action = st.checkbox("Set next action date", value=True)
        next_action_date = st.date_input("Next action date", value=date.today()) if use_next_action else None
        notes = st.text_area(
            "Notes",
            placeholder="What was sent, what the contact said, and what the next move should be.",
        )
        submitted = st.form_submit_button("Log follow-up")

    if submitted:
        contact_id = contact_options[contact_label]
        log_follow_up(
            contact_id=contact_id,
            asset_id=str(context.selected_asset["asset_id"]),
            event_date=event_date.isoformat(),
            channel=channel,
            outcome=outcome,
            next_action_date=next_action_date.isoformat() if next_action_date else None,
            owner="Jean",
            notes=notes.strip(),
        )
        st.success("Follow-up logged.")
        st.rerun()


def _render_methodology_note() -> None:
    if not YIELD_EXTRACTION_NOTE_PATH.exists():
        return
    with st.expander("Yield extraction methodology", expanded=False):
        st.markdown(YIELD_EXTRACTION_NOTE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Outreach Selection Console", layout="wide")
    bootstrap_outreach_environment()

    context = load_dashboard_context(asset_id=st.session_state.get("selected_asset_id"))
    asset_options = {
        f"{row.asset_name} | {row.zone} | EUR {row.fair_value_eur_mn:,.1f}m": row.asset_id
        for row in context.assets.itertuples(index=False)
    }
    asset_labels = list(asset_options.keys())
    current_asset_label = next(
        (label for label, asset_id in asset_options.items() if asset_id == context.selected_asset["asset_id"]),
        asset_labels[0],
    )

    with st.sidebar:
        st.title("Outreach console")
        asset_label = st.selectbox(
            "Asset in scope",
            options=asset_labels,
            index=asset_labels.index(current_asset_label),
        )
        top_k = st.slider("Top contacts to display", min_value=5, max_value=12, value=TOP_K_DEFAULT)
        fit_filter = st.multiselect("Fit label", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        st.caption("The ranking combines geography, asset focus, ticket size, target yield, relationship history, and cooldown.")

    selected_asset_id = asset_options[asset_label]
    if selected_asset_id != context.selected_asset["asset_id"]:
        st.session_state["selected_asset_id"] = selected_asset_id
        context = load_dashboard_context(asset_id=selected_asset_id)

    ranked = context.ranked_contacts.loc[context.ranked_contacts["fit_label"].isin(fit_filter)].copy()
    if ranked.empty:
        ranked = context.ranked_contacts.copy()

    default_contact_id = st.session_state.get("selected_contact_id")
    ranked_contact_ids = ranked["contact_id"].tolist()
    if default_contact_id not in ranked_contact_ids:
        default_contact_id = ranked_contact_ids[0]

    st.title("Outreach Selection and Follow-up")
    st.caption(
        "This Streamlit app ranks who to contact for a selected asset, keeps a lightweight local outreach database, "
        "and surfaces a full fiche for the contact you want to work."
    )

    _render_asset_summary(context.selected_asset)

    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        ranking_display = _render_ranked_contacts(ranked, top_k=top_k)
        contact_lookup = {
            f"{row.full_name} | {row.company} | score {row.outreach_score:.1f}": row.contact_id
            for row in ranked.head(top_k).itertuples(index=False)
        }
        selected_label = next(
            (label for label, contact_id in contact_lookup.items() if contact_id == default_contact_id),
            next(iter(contact_lookup)),
        )
        chosen_label = st.selectbox(
            "Contact to inspect",
            options=list(contact_lookup.keys()),
            index=list(contact_lookup.keys()).index(selected_label),
        )
        selected_contact_id = contact_lookup[chosen_label]
        st.session_state["selected_contact_id"] = selected_contact_id

        st.download_button(
            label="Download ranked contacts",
            data=_csv_download_bytes(ranking_display),
            file_name=f"{context.selected_asset['asset_id']}_ranked_contacts.csv",
            mime="text/csv",
        )

    with right:
        _render_contact_fiche(asset_id=str(context.selected_asset["asset_id"]), contact_id=selected_contact_id)

    tabs = st.tabs(["Follow-ups", "Database", "Method"])

    with tabs[0]:
        _render_follow_up_form(context, default_contact_id=selected_contact_id)

    with tabs[1]:
        st.subheader("Database snapshot")
        db_left, db_right = st.columns(2)
        with db_left:
            st.markdown("**Assets**")
            st.dataframe(
                context.assets[
                    ["asset_id", "asset_name", "zone", "asset_class", "fair_value_eur_mn", "cap_rate_range_pct"]
                ].rename(
                    columns={
                        "asset_id": "Asset ID",
                        "asset_name": "Asset",
                        "zone": "Zone",
                        "asset_class": "Class",
                        "fair_value_eur_mn": "Value (EUR mn)",
                        "cap_rate_range_pct": "Yield band",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with db_right:
            st.markdown("**Contacts**")
            st.dataframe(
                context.contacts[
                    ["contact_id", "full_name", "company", "zone_focus", "asset_focus", "relationship_stage"]
                ].rename(
                    columns={
                        "contact_id": "Contact ID",
                        "full_name": "Name",
                        "company": "Company",
                        "zone_focus": "Zone focus",
                        "asset_focus": "Asset focus",
                        "relationship_stage": "Stage",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        _render_methodology_note()


if __name__ == "__main__":
    main()
