from __future__ import annotations

import io
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.backend.outreach_service import (
    build_deal_input,
    bootstrap_outreach_environment,
    build_scbsm_fiche_markdown,
    get_scbsm_history,
    load_dashboard_context,
    load_staged_mandate_into_working_set,
    log_follow_up,
)
from src.backend.paths import PROJECT_ROOT, YIELD_EXTRACTION_NOTE_PATH

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
COMPS_SAMPLE_PATH = ARTIFACTS_DIR / "comps_sample.parquet"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEAL_FIELD_KEYS = {
    "mandate_name": "deal_mandate_name",
    "asset_type": "deal_asset_type",
    "country": "deal_country",
    "zone": "deal_zone",
    "city": "deal_city",
    "ticket_eur_mn": "deal_ticket_eur_mn",
    "cap_rate_pct": "deal_cap_rate_pct",
    "size_sqm": "deal_size_sqm",
    "transaction_date": "deal_transaction_date",
}


def _csv_download_bytes(frame: pd.DataFrame) -> bytes:
    output = io.StringIO()
    frame.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def _set_current_deal_state(payload: dict[str, Any]) -> None:
    baseline = build_deal_input(payload).as_dict()
    for field, state_key in DEAL_FIELD_KEYS.items():
        value = baseline[field]
        if field == "transaction_date":
            st.session_state[state_key] = pd.to_datetime(value).date()
        else:
            st.session_state[state_key] = value


def _ensure_current_deal_state() -> None:
    if "current_mandate_source" not in st.session_state:
        st.session_state["current_mandate_source"] = "manual"
    defaults = build_deal_input().as_dict()
    for field, state_key in DEAL_FIELD_KEYS.items():
        if state_key not in st.session_state:
            value = defaults[field]
            if field == "transaction_date":
                st.session_state[state_key] = pd.to_datetime(value).date()
            else:
                st.session_state[state_key] = value


def _read_current_deal_state() -> dict[str, Any]:
    return {
        "mandate_name": st.session_state[DEAL_FIELD_KEYS["mandate_name"]],
        "asset_type": st.session_state[DEAL_FIELD_KEYS["asset_type"]],
        "country": st.session_state[DEAL_FIELD_KEYS["country"]],
        "zone": st.session_state[DEAL_FIELD_KEYS["zone"]],
        "city": st.session_state[DEAL_FIELD_KEYS["city"]],
        "ticket_eur_mn": float(st.session_state[DEAL_FIELD_KEYS["ticket_eur_mn"]]),
        "cap_rate_pct": float(st.session_state[DEAL_FIELD_KEYS["cap_rate_pct"]]),
        "size_sqm": float(st.session_state[DEAL_FIELD_KEYS["size_sqm"]]),
        "transaction_date": st.session_state[DEAL_FIELD_KEYS["transaction_date"]].isoformat(),
    }


@st.cache_data(show_spinner=False)
def load_comparable_artifacts() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if not COMPS_SAMPLE_PATH.exists() or not METADATA_PATH.exists():
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in [COMPS_SAMPLE_PATH, METADATA_PATH] if not path.exists()]
        raise FileNotFoundError(f"Missing required artifact(s): {', '.join(missing)}. Run `python -m model.train` first.")

    comps = pd.read_parquet(COMPS_SAMPLE_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    benchmark_frame = pd.DataFrame(metadata["reference_benchmark"]["cells"])
    return comps, metadata, benchmark_frame


def _score_comparables(
    comps: pd.DataFrame,
    asset_type: str,
    country: str,
    query_year: int,
    query_size_sqm: float,
    weights: dict[str, float],
) -> pd.DataFrame:
    frame = comps.copy()
    query_log_size = float(np.log(max(query_size_sqm, 1.0)))
    comp_log_size = np.log(frame["size_bucket_mid_sqm"].clip(lower=1.0))
    frame["log_size_gap"] = np.abs(query_log_size - comp_log_size)

    size_component = weights["log_size_similarity_scale"] / (
        1.0 + weights["log_size_similarity_penalty_multiplier"] * frame["log_size_gap"]
    )
    frame["similarity_score"] = size_component
    frame["similarity_score"] += np.where(frame["asset_type"].eq(asset_type), weights["same_asset_type_bonus"], 0.0)
    frame["similarity_score"] += np.where(frame["country"].eq(country), weights["same_country_bonus"], 0.0)
    frame["similarity_score"] += np.where(frame["year_bucket_order"].eq(query_year), weights["same_year_bonus"], 0.0)

    return frame.sort_values(
        ["similarity_score", "log_size_gap", "year_bucket_order"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _display_comp_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.loc[
        :,
        [
            "similarity_score",
            "asset_type",
            "country",
            "year_bucket",
            "size_bucket",
            "price_bucket",
            "price_per_sqm_bucket",
        ],
    ].copy()
    display["similarity_score"] = display["similarity_score"].round(1)
    return display.rename(
        columns={
            "similarity_score": "Similarity score",
            "asset_type": "Asset type",
            "country": "Country",
            "year_bucket": "Year bucket",
            "size_bucket": "Size bucket",
            "price_bucket": "Price bucket",
            "price_per_sqm_bucket": "Price per sqm bucket",
        }
    )


def _deal_input_sidebar() -> dict[str, Any]:
    _ensure_current_deal_state()
    with st.sidebar:
        st.title("Mandate intake")
        if st.button("Reset to default mandate", use_container_width=True):
            _set_current_deal_state(build_deal_input().as_dict())
            st.session_state["current_mandate_source"] = "manual"
            st.rerun()

        st.caption(f"Current source: `{st.session_state['current_mandate_source']}`")
        st.text_input("Mandate name", key=DEAL_FIELD_KEYS["mandate_name"])
        st.selectbox(
            "Asset type",
            options=["Office", "Retail", "Mixed Commercial", "Industrial", "Hotel", "Residential", "Land"],
            key=DEAL_FIELD_KEYS["asset_type"],
        )
        st.text_input("Country", key=DEAL_FIELD_KEYS["country"])
        st.text_input("Zone / region", key=DEAL_FIELD_KEYS["zone"])
        st.text_input("City", key=DEAL_FIELD_KEYS["city"])
        st.number_input("Deal size (EUR mn)", min_value=0.5, step=1.0, key=DEAL_FIELD_KEYS["ticket_eur_mn"])
        st.number_input("Cap rate estimate (%)", min_value=1.0, max_value=15.0, step=0.25, key=DEAL_FIELD_KEYS["cap_rate_pct"])
        st.number_input("Approximate size (sqm)", min_value=100.0, step=100.0, key=DEAL_FIELD_KEYS["size_sqm"])
        st.date_input("Transaction date", key=DEAL_FIELD_KEYS["transaction_date"])
        st.caption(
            "This prototype scores one investor only: SCBSM. Fit is derived from the disclosed SCBSM portfolio, "
            "public yield context, and the minimal SCBSM interaction log."
        )
    return _read_current_deal_state()


def _render_platform_summary(context) -> None:
    st.title("Alantra x SCBSM Mandate Fit Prototype")
    st.caption(
        "This prototype evaluates one mandate against one investor: SCBSM. "
        "SCBSM assets remain SCBSM-owned portfolio rows, and inbound HTTP mandates feed a visible staging queue."
    )
    profile = context.scbsm_profile
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SCBSM assets", f"{profile['asset_count']}")
    col2.metric("Portfolio fair value", f"EUR {profile['total_fair_value_eur_mn']:,.1f}m")
    col3.metric("Weighted cap rate", f"{profile['weighted_cap_rate_pct']:.2f}%")
    col4.metric("Staged mandates", f"{len(context.staged_mandates)}")


def _render_deal_summary(deal) -> None:
    st.subheader("Mandate in scope")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Asset type", deal.asset_type)
    col2.metric("Geography", deal.zone)
    col3.metric("City", deal.city or "N/A")
    col4.metric("Ticket size", f"EUR {deal.ticket_eur_mn:,.1f}m")
    col5.metric("Cap rate", f"{deal.cap_rate_pct:.2f}%")
    st.caption(
        f"Mandate name: `{deal.mandate_name}` | Transaction date: `{deal.transaction_date}` | "
        f"Approximate size: `{deal.size_sqm:,.0f} sqm`"
    )


def _render_scbsm_evaluation(context) -> None:
    evaluation = context.scbsm_evaluation
    profile = context.scbsm_profile
    history = get_scbsm_history(context.events)

    st.subheader("SCBSM fit evaluation")
    top_left, top_mid, top_right, top_far = st.columns(4)
    top_left.metric("Fit score", f"{evaluation['outreach_score']:.1f}")
    top_mid.metric("Fit label", evaluation["fit_label"])
    top_right.metric("Zone reference cap rate", f"{evaluation['zone_reference_cap_rate_pct']:.2f}%")
    top_far.metric("Latest outcome", evaluation["latest_outcome"].replace("_", " ").title())

    sub_left, sub_mid, sub_right = st.columns(3)
    sub_left.metric("Same-zone assets", f"{evaluation['same_zone_assets_count']}")
    sub_mid.metric("Same-asset assets", f"{evaluation['same_asset_assets_count']}")
    sub_right.metric("Same zone + asset value", f"EUR {evaluation['same_zone_asset_assets_value_eur_mn']:,.1f}m")

    st.info(evaluation["match_summary"])
    if evaluation["risk_flags"]:
        st.warning(evaluation["risk_flags"])

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.markdown("**Why the mandate matches SCBSM**")
        st.markdown(evaluation["explicit_reasons"] or "- No explicit reason recorded.")

        fiche_markdown = build_scbsm_fiche_markdown(
            deal=context.current_deal,
            profile=profile,
            evaluation=evaluation,
            history=history,
        )
        st.download_button(
            label="Download SCBSM fiche",
            data=fiche_markdown.encode("utf-8"),
            file_name="scbsm_mandate_fit.md",
            mime="text/markdown",
        )

    with right:
        st.markdown("**SCBSM profile summary**")
        st.write(f"**{profile['company']}** | {profile['title']}")
        st.caption(profile["qualitative_focus"])
        st.write(f"Zone coverage: `{profile['zone_focus']}`")
        st.write(f"Asset coverage: `{profile['asset_focus']}`")
        st.write(
            f"Ticket range: EUR {profile['min_ticket_eur_mn']:,.1f}m to EUR {profile['max_ticket_eur_mn']:,.1f}m"
        )
        st.write(f"Weighted cap rate: {profile['weighted_cap_rate_pct']:.2f}%")
        st.write(f"Preferred channel: `{profile['preferred_channel']}`")
        st.write(f"Owner: `{profile['owner']}`")
        if profile["notes"]:
            st.write(profile["notes"])

    st.markdown("**SCBSM interaction log**")
    if not history.empty:
        display = history[
            [
                "event_date",
                "mandate_name",
                "deal_asset_type",
                "deal_zone",
                "deal_city",
                "deal_ticket_eur_mn",
                "deal_cap_rate_pct",
                "channel",
                "outcome",
                "next_action_date",
                "notes",
            ]
        ].rename(
            columns={
                "event_date": "Event date",
                "mandate_name": "Mandate",
                "deal_asset_type": "Asset type",
                "deal_zone": "Zone",
                "deal_city": "City",
                "deal_ticket_eur_mn": "Ticket (EUR mn)",
                "deal_cap_rate_pct": "Cap rate",
                "channel": "Channel",
                "outcome": "Outcome",
                "next_action_date": "Next action",
                "notes": "Notes",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No SCBSM interaction has been logged yet.")


def _render_follow_up_form(context) -> None:
    st.subheader("Log SCBSM follow-up")
    with st.form("follow_up_form"):
        channel = st.selectbox("Channel", options=["email", "phone", "meeting", "linkedin"])
        outcome = st.selectbox("Outcome", options=["positive", "neutral", "no_reply", "not_now"])
        event_date = st.date_input("Event date", value=date.today())
        use_next_action = st.checkbox("Set next action date", value=True)
        next_action_date = st.date_input("Next action date", value=date.today()) if use_next_action else None
        notes = st.text_area(
            "Notes",
            placeholder="What was sent, how SCBSM reacted, and what the next move should be.",
        )
        submitted = st.form_submit_button("Log follow-up")

    if submitted:
        log_follow_up(
            deal=context.current_deal,
            event_date=event_date.isoformat(),
            channel=channel,
            outcome=outcome,
            next_action_date=next_action_date.isoformat() if next_action_date else None,
            owner=context.scbsm_profile["owner"],
            notes=notes.strip(),
        )
        st.success("SCBSM follow-up logged.")
        st.rerun()


def _format_staged_mandate_label(row: pd.Series) -> str:
    received_at = pd.to_datetime(row["received_at"], errors="coerce")
    received_label = received_at.strftime("%Y-%m-%d %H:%M") if pd.notna(received_at) else "unknown time"
    return (
        f"{row['mandate_name']} | {row['asset_type']} | {row['zone']} | "
        f"EUR {float(row['ticket_eur_mn']):,.1f}m | {received_label}"
    )


def _render_inbound_mandates(context) -> None:
    st.subheader("Inbound mandates")
    st.caption("These rows are fed by the FastAPI sidecar through `POST /mandates/staging` and remain staged until loaded.")

    staged = context.staged_mandates.copy()
    if staged.empty:
        st.info("No staged mandates have been received yet.")
    else:
        display = staged[
            [
                "received_at",
                "staged_mandate_id",
                "mandate_name",
                "asset_type",
                "country",
                "zone",
                "city",
                "ticket_eur_mn",
                "cap_rate_pct",
                "status",
                "source",
            ]
        ].rename(
            columns={
                "received_at": "Received at",
                "staged_mandate_id": "Stage ID",
                "mandate_name": "Mandate",
                "asset_type": "Asset type",
                "country": "Country",
                "zone": "Zone",
                "city": "City",
                "ticket_eur_mn": "Ticket (EUR mn)",
                "cap_rate_pct": "Cap rate",
                "status": "Status",
                "source": "Source",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        stage_lookup = {
            _format_staged_mandate_label(row): row["staged_mandate_id"]
            for _, row in staged.iterrows()
        }
        selected_label = st.selectbox("Mandate to load", options=list(stage_lookup.keys()))
        staged_id = stage_lookup[selected_label]
        staged_row = staged.loc[staged["staged_mandate_id"].eq(staged_id)].iloc[0]
        st.caption(
            f"Source: `{staged_row['source'] or 'n/a'}` | Notes: `{staged_row['notes'] or 'n/a'}` | "
            f"Status: `{staged_row['status']}`"
        )

        if st.button("Load staged mandate into the working screen", use_container_width=True):
            payload = load_staged_mandate_into_working_set(staged_id)
            _set_current_deal_state(payload)
            st.session_state["current_mandate_source"] = f"staged:{staged_id}"
            st.rerun()

    with st.expander("HTTP intake example", expanded=False):
        st.code(
            """curl -X POST http://127.0.0.1:8000/mandates/staging ^
  -H "Content-Type: application/json" ^
  -d "{\"mandate_name\":\"Rue de Rivoli office\",\"asset_type\":\"Office\",\"country\":\"France\",\"zone\":\"Paris\",\"city\":\"Paris\",\"ticket_eur_mn\":42.0,\"cap_rate_pct\":4.9,\"size_sqm\":6200,\"transaction_date\":\"2026-04-15\",\"source\":\"manual test\",\"notes\":\"queued from local prototype\"}" """,
            language="powershell",
        )


def _render_portfolio_context(context) -> None:
    st.subheader("SCBSM portfolio context")
    profile = context.scbsm_profile
    assets = context.assets.copy()

    head_left, head_right = st.columns(2, gap="large")
    with head_left:
        st.markdown("**Profile**")
        st.write(f"Display name: `{profile['display_name']}`")
        st.write(f"Top zone: `{profile['top_zone']}`")
        st.write(f"Top asset class: `{profile['top_asset_class']}`")
        st.write(f"Portfolio fair value: EUR {profile['total_fair_value_eur_mn']:,.1f}m")
        st.write(f"Weighted cap rate: {profile['weighted_cap_rate_pct']:.2f}%")
        st.write(f"Cap rate range: {profile['min_cap_rate_pct']:.2f}% to {profile['max_cap_rate_pct']:.2f}%")

    with head_right:
        st.markdown("**Mix tables**")
        zone_mix = (
            assets.groupby("zone", as_index=False)["fair_value_eur_mn"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "Asset count", "sum": "Fair value (EUR mn)"})
        )
        asset_mix = (
            assets.groupby("asset_class", as_index=False)["fair_value_eur_mn"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "Asset count", "sum": "Fair value (EUR mn)"})
        )
        mix_left, mix_right = st.columns(2)
        mix_left.dataframe(zone_mix, use_container_width=True, hide_index=True)
        mix_right.dataframe(asset_mix, use_container_width=True, hide_index=True)

    st.markdown("**Disclosed SCBSM assets**")
    display = assets[
        [
            "asset_id",
            "asset_name",
            "asset_class",
            "city",
            "zone",
            "fair_value_eur_mn",
            "cap_rate_range_pct",
            "yield_mid_pct",
        ]
    ].rename(
        columns={
            "asset_id": "Asset ID",
            "asset_name": "Asset",
            "asset_class": "Asset class",
            "city": "City",
            "zone": "Zone",
            "fair_value_eur_mn": "Fair value (EUR mn)",
            "cap_rate_range_pct": "Public cap-rate band",
            "yield_mid_pct": "Yield midpoint",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_reference_benchmark(benchmark_frame: pd.DataFrame, asset_type: str, country: str, query_size_sqm: float) -> None:
    st.subheader("Reference benchmark")
    st.caption("Transparent benchmark, not a prediction.")

    cell = benchmark_frame.loc[
        benchmark_frame["asset_type"].eq(asset_type) & benchmark_frame["country"].eq(country)
    ].copy()
    if cell.empty:
        st.warning("No exact asset-type by country benchmark cell exists for this query.")
        return

    row = cell.iloc[0]
    left, right, third = st.columns(3)
    left.metric("Median price per sqm", f"EUR {row['median_price_per_sqm_eur']:,.0f}")
    right.metric("Cell sample size", f"{int(row['sample_size'])}")
    third.metric("Reference deal size", f"EUR {(row['median_price_per_sqm_eur'] * query_size_sqm / 1_000_000):,.1f}m")


def _render_comparable_module(prefill_asset_type: str, prefill_country: str, prefill_city: str, prefill_size: float) -> None:
    st.subheader("Comparable Retrieval / AVM module")
    st.caption(
        "This is the existing comparable-retrieval proof of concept. It returns the closest anonymised comparables "
        "and a transparent naive benchmark."
    )

    try:
        comps, metadata, benchmark_frame = load_comparable_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    weights = metadata["retrieval_scoring"]["weights"]
    asset_type_options = metadata["asset_type_levels"]
    country_options = metadata["country_group_levels"]
    default_asset = prefill_asset_type if prefill_asset_type in asset_type_options else asset_type_options[0]
    default_country = prefill_country if prefill_country in country_options else country_options[0]

    comp_left, comp_right = st.columns([0.8, 1.2], gap="large")
    with comp_left:
        asset_type = st.selectbox(
            "Comparable asset type",
            options=asset_type_options,
            index=asset_type_options.index(default_asset),
            key="comp_asset_type",
        )
        country = st.selectbox(
            "Comparable country",
            options=country_options,
            index=country_options.index(default_country),
            key="comp_country",
        )
        city = st.text_input("City note", value=prefill_city or "Paris", key="comp_city")
        size_sqm = st.number_input("Approximate size (sqm)", min_value=100.0, value=float(prefill_size), step=100.0, key="comp_size")
        transaction_date = st.date_input("Comparable transaction date", value=date.today(), key="comp_date")
        st.caption(f"City note `{city}` is stored for context but not passed to the current matching engine.")

    ranked = _score_comparables(
        comps=comps,
        asset_type=asset_type,
        country=country,
        query_year=transaction_date.year,
        query_size_sqm=float(size_sqm),
        weights=weights,
    )
    display_frame = _display_comp_table(ranked.head(10))

    with comp_right:
        st.dataframe(display_frame, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download retrieved comparables",
            data=_csv_download_bytes(display_frame),
            file_name="retrieved_comparables.csv",
            mime="text/csv",
        )
        _render_reference_benchmark(
            benchmark_frame=benchmark_frame,
            asset_type=asset_type,
            country=country,
            query_size_sqm=float(size_sqm),
        )


def _render_method_note() -> None:
    st.subheader("Method and data gap")
    st.write(
        "This version of the prototype is intentionally narrow: one mandate is evaluated against one investor, SCBSM. "
        "The investor profile is derived from SCBSM's disclosed portfolio rather than from a broader outreach registry."
    )
    st.write(
        "HTTP mandate intake was added as a staging layer so new opportunities can enter the system in structured form. "
        "That staging queue is visible in the app, and a mandate only becomes the active working case when the analyst loads it."
    )
    st.write(
        "The comparable-retrieval module remains separate. It provides valuation context, while the SCBSM module tests the "
        "single-investor outreach workflow."
    )
    if YIELD_EXTRACTION_NOTE_PATH.exists():
        with st.expander("Public yield extraction note", expanded=False):
            st.markdown(YIELD_EXTRACTION_NOTE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Alantra x SCBSM Mandate Fit Prototype", layout="wide")
    bootstrap_outreach_environment()

    deal_input = _deal_input_sidebar()
    context = load_dashboard_context(deal_input=deal_input)
    _render_platform_summary(context)
    _render_deal_summary(context.current_deal)

    tabs = st.tabs(
        [
            "SCBSM Fit",
            "Inbound Mandates",
            "SCBSM Portfolio",
            "Comparable Retrieval",
            "Method",
        ]
    )

    with tabs[0]:
        _render_scbsm_evaluation(context)
        st.divider()
        _render_follow_up_form(context)

    with tabs[1]:
        _render_inbound_mandates(context)

    with tabs[2]:
        _render_portfolio_context(context)

    with tabs[3]:
        _render_comparable_module(
            prefill_asset_type=context.current_deal.asset_type,
            prefill_country=context.current_deal.country,
            prefill_city=context.current_deal.city,
            prefill_size=context.current_deal.size_sqm,
        )

    with tabs[4]:
        _render_method_note()


if __name__ == "__main__":
    main()
