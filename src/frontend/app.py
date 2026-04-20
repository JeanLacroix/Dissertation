from __future__ import annotations

import html
import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.backend.comparables_service import (
    ComparableQuery,
    classify_comparable_scenario,
    load_prepared_comparables,
    retrieve_comparables,
)
from src.backend.outreach_scoring import derive_scbsm_profile, score_scbsm_for_deal
from src.backend.outreach_service import (
    build_deal_input,
    bootstrap_outreach_environment,
    create_mock_mandate,
    get_scbsm_history,
    load_dashboard_context,
    load_staged_mandate_into_working_set,
    load_profile_metadata,
    log_touchpoint,
    refresh_scbsm_profile_from_public_data,
    save_profile_metadata,
    validate_profile_payload,
)
from src.backend.paths import PROJECT_ROOT


ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
LEAD_BANKER_KEY = "deal_lead_banker"
FOLLOW_UP_DELAY_DAYS = 7
INVESTOR_VALIDATION_TOP_N = 6

SCREENS = [
    ("dashboard", "Dashboard"),
    ("new-mandate", "New Mandate"),
    ("investors", "Investors"),
    ("investor-validation", "Investor Validation"),
    ("comparables", "Comparables"),
    ("outreach-log", "Outreach Log"),
]

DEAL_FIELD_KEYS = {
    "mandate_name": "deal_mandate_name",
    "asset_type": "deal_asset_type",
    "country": "deal_country",
    "zone": "deal_zone",
    "city": "deal_city",
    "price_min_eur_mn": "deal_price_min_eur_mn",
    "price_max_eur_mn": "deal_price_max_eur_mn",
    "cap_rate_known": "deal_cap_rate_known",
    "cap_rate_pct": "deal_cap_rate_pct",
    "size_sqm": "deal_size_sqm",
    "transaction_date": "deal_transaction_date",
    "noi_known": "deal_noi_known",
    "noi_eur_mn": "deal_noi_eur_mn",
    "lease_terms": "deal_lease_terms",
    "building_grade": "deal_building_grade",
}

TOUCHPOINT_ACTION_OPTIONS = [
    ("Teaser sent", "teaser_sent", "sent"),
    ("Response - yes", "response", "yes"),
    ("Response - no", "response", "no"),
    ("Response - no reply", "response", "no reply"),
    ("NDA sent", "nda_sent", "sent"),
    ("NDA signed", "nda_signed", "signed"),
    ("NDA not signed", "nda_signed", "not signed"),
    ("Meeting held", "meeting_held", "held"),
    ("Outcome - shortlisted", "outcome", "shortlisted"),
    ("Outcome - declined", "outcome", "declined"),
    ("Outcome - won", "outcome", "won"),
    ("Outcome - lost", "outcome", "lost"),
]

TOUCHPOINT_ACTION_LOOKUP = {
    label: {
        "touchpoint_type": touchpoint_type,
        "status_value": status_value,
    }
    for label, touchpoint_type, status_value in TOUCHPOINT_ACTION_OPTIONS
}

TOUCHPOINT_LABELS = {
    "teaser_sent": "Teaser sent",
    "response": "Response",
    "nda_sent": "NDA sent",
    "nda_signed": "NDA signed",
    "meeting_held": "Meeting held",
    "outcome": "Outcome",
    "criteria_override": "Override logged",
}


def _csv_download_bytes(frame: pd.DataFrame) -> bytes:
    output = io.StringIO()
    frame.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def _workbook_download_bytes(
    *,
    evaluation: dict[str, Any],
    criteria_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    comparable_frame: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([evaluation]).to_excel(writer, index=False, sheet_name="Investor Fit")
        criteria_frame.to_excel(writer, index=False, sheet_name="Fit Criteria")
        history_frame.to_excel(writer, index=False, sheet_name="Touchpoints")
        comparable_frame.to_excel(writer, index=False, sheet_name="Comparable Results")
    output.seek(0)
    return output.getvalue()


def _load_method_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _valuation_warning_text(method_metadata: dict[str, Any]) -> str:
    default = "Point valuation is not reliable here. Use the comp set plus the cell median as a transparent benchmark."
    valuation = method_metadata.get("valuation_evaluation", {})
    rolling = valuation.get("rolling_origin", {}) if isinstance(valuation, dict) else {}
    if not isinstance(rolling, dict):
        return default

    model_mape = rolling.get("mean_mape_pct")
    naive_mape = valuation.get("rolling_naive_baseline_mean_mape_pct")
    if naive_mape is None:
        naive_mape = rolling.get("baseline_mean_mape_pct")
    if naive_mape is None:
        naive_mape = rolling.get("headline_fold_baseline_mape_pct")
    if model_mape is None or naive_mape is None:
        return default

    return f"Point valuation rejected. Hedonic rolling-origin MAPE {float(model_mape):.1f}% versus naive {float(naive_mape):.1f}%."


def _set_current_deal_state(payload: dict[str, Any]) -> None:
    baseline = build_deal_input(payload).as_dict()
    for field, state_key in DEAL_FIELD_KEYS.items():
        if field in {"cap_rate_known", "noi_known"}:
            continue
        value = baseline.get(field)
        if field == "transaction_date":
            st.session_state[state_key] = pd.to_datetime(value).date()
        elif field == "cap_rate_pct":
            st.session_state[state_key] = float(value) if value is not None else 4.75
            st.session_state[DEAL_FIELD_KEYS["cap_rate_known"]] = value is not None
        elif field == "noi_eur_mn":
            st.session_state[state_key] = float(value) if value is not None else 0.0
            st.session_state[DEAL_FIELD_KEYS["noi_known"]] = value is not None
        else:
            st.session_state[state_key] = value


def _ensure_current_deal_state() -> None:
    defaults = build_deal_input().as_dict()
    if LEAD_BANKER_KEY not in st.session_state:
        st.session_state[LEAD_BANKER_KEY] = "J. Dupont"
    for field, state_key in DEAL_FIELD_KEYS.items():
        if field == "cap_rate_known":
            if state_key not in st.session_state:
                st.session_state[state_key] = defaults["cap_rate_pct"] is not None
            continue
        if field == "noi_known":
            if state_key not in st.session_state:
                st.session_state[state_key] = defaults["noi_eur_mn"] is not None
            continue
        if state_key not in st.session_state:
            value = defaults.get(field)
            if field == "transaction_date":
                st.session_state[state_key] = pd.to_datetime(value).date()
            elif field == "cap_rate_pct":
                st.session_state[state_key] = float(value) if value is not None else 4.75
            elif field == "noi_eur_mn":
                st.session_state[state_key] = float(value) if value is not None else 0.0
            else:
                st.session_state[state_key] = value


def _read_current_deal_state() -> dict[str, Any]:
    cap_rate = (
        float(st.session_state[DEAL_FIELD_KEYS["cap_rate_pct"]])
        if st.session_state[DEAL_FIELD_KEYS["cap_rate_known"]]
        else None
    )
    noi = (
        float(st.session_state[DEAL_FIELD_KEYS["noi_eur_mn"]])
        if st.session_state[DEAL_FIELD_KEYS["noi_known"]]
        else None
    )
    return {
        "mandate_name": st.session_state[DEAL_FIELD_KEYS["mandate_name"]],
        "asset_type": st.session_state[DEAL_FIELD_KEYS["asset_type"]],
        "country": st.session_state[DEAL_FIELD_KEYS["country"]],
        "zone": st.session_state[DEAL_FIELD_KEYS["zone"]],
        "city": st.session_state[DEAL_FIELD_KEYS["city"]],
        "price_min_eur_mn": float(st.session_state[DEAL_FIELD_KEYS["price_min_eur_mn"]]),
        "price_max_eur_mn": float(st.session_state[DEAL_FIELD_KEYS["price_max_eur_mn"]]),
        "cap_rate_pct": cap_rate,
        "size_sqm": float(st.session_state[DEAL_FIELD_KEYS["size_sqm"]]),
        "transaction_date": st.session_state[DEAL_FIELD_KEYS["transaction_date"]].isoformat(),
        "noi_eur_mn": noi,
        "lease_terms": st.session_state[DEAL_FIELD_KEYS["lease_terms"]],
        "building_grade": st.session_state[DEAL_FIELD_KEYS["building_grade"]],
    }


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _badge(text: str, tone: str = "neutral") -> str:
    return f"<span class='adi-badge adi-badge--{tone}'>{_escape(text)}</span>"


def _criteria_status_badge(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "yes":
        return _badge("Yes", "success")
    if normalized == "not assessed":
        return _badge("Not assessed", "warning")
    return _badge("No", "danger")


def _nav_button(screen: str, label: str, *, primary: bool = False, small: bool = False, extra_query: str = "") -> str:
    classes = ["adi-link-button"]
    if primary:
        classes.append("adi-link-button--primary")
    if small:
        classes.append("adi-link-button--small")
    href = f"?screen={screen}"
    if extra_query:
        href = f"{href}&{extra_query}"
    return f"<a class='{' '.join(classes)}' href='{href}'>{_escape(label)}</a>"


def _section_intro(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='adi-section-intro'>"
            f"<h1>{_escape(title)}</h1>"
            f"<p>{_escape(body)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _field_label(label: str) -> None:
    st.markdown(f"<div class='adi-field-label'>{_escape(label)}</div>", unsafe_allow_html=True)


def _card_html(label: str, value: str, *, note: str = "", accent: bool = False) -> str:
    accent_class = " adi-card--accent" if accent else ""
    note_html = f"<div class='adi-card-note'>{_escape(note)}</div>" if note else ""
    return (
        f"<div class='adi-card{accent_class}'>"
        f"<div class='adi-card-label'>{_escape(label)}</div>"
        f"<div class='adi-card-value'>{_escape(value)}</div>"
        f"{note_html}"
        "</div>"
    )


def _banner_html(title: str, detail: str, *, tone: str) -> str:
    return (
        f"<div class='adi-banner adi-banner--{tone}'>"
        f"<div class='adi-banner-title'>{_escape(title)}</div>"
        f"<div class='adi-banner-detail'>{_escape(detail)}</div>"
        "</div>"
    )


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<div class='adi-note'>No rows available.</div>"
    table = frame.to_html(index=False, escape=False, classes="adi-table", border=0)
    return f"<div class='adi-table-wrap'>{table}</div>"


def _format_money_mn(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"EUR {float(value):,.1f}m"


def _format_pct(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}%"


def _title_case_status(value: str) -> str:
    return str(value or "").replace("_", " ").title()


def _touchpoint_display_name(value: str) -> str:
    return TOUCHPOINT_LABELS.get(str(value), _title_case_status(str(value)))


def _history_export_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Date", "Mandate", "Action", "Detail", "Follow-up", "Notes"])
    display = history.copy()
    display["Date"] = pd.to_datetime(display["event_date"], errors="coerce").dt.date.astype(str)
    display["Mandate"] = display["mandate_name"]
    display["Action"] = display["touchpoint_type"].map(_touchpoint_display_name)
    display["Detail"] = display["status_value"].map(_title_case_status)
    display["Follow-up"] = display.apply(lambda row: _follow_up_text(history, row), axis=1)
    display["Notes"] = display["notes"].fillna("")
    return display[["Date", "Mandate", "Action", "Detail", "Follow-up", "Notes"]]


def _deal_from_staged_row(row: pd.Series) -> Any:
    return build_deal_input(
        {
            "mandate_name": row["mandate_name"],
            "asset_type": row["asset_type"],
            "country": row["country"],
            "zone": row["zone"],
            "city": row.get("city", ""),
            "price_min_eur_mn": row.get("price_min_eur_mn"),
            "price_max_eur_mn": row.get("price_max_eur_mn"),
            "ticket_eur_mn": row.get("ticket_eur_mn"),
            "cap_rate_pct": row.get("cap_rate_pct"),
            "size_sqm": row.get("size_sqm"),
            "transaction_date": row.get("transaction_date"),
            "noi_eur_mn": row.get("noi_eur_mn"),
            "lease_terms": row.get("lease_terms", ""),
            "building_grade": row.get("building_grade", ""),
        }
    )


def _mandate_option_label(*, deal: Any, status: str, mandate_id: str = "") -> str:
    base = (
        f"{deal.mandate_name} | {deal.asset_type} | {deal.zone or deal.country} | "
        f"EUR {deal.ticket_eur_mn:,.1f}m"
    )
    if mandate_id:
        return f"{base} | {status} | {mandate_id}"
    return f"{base} | {status}"


def _available_mandate_options(context) -> tuple[dict[str, Any], str]:
    options: dict[str, Any] = {}
    current_label = _mandate_option_label(deal=context.current_deal, status="current")
    options[current_label] = context.current_deal

    if not context.staged_mandates.empty:
        for _, row in context.staged_mandates.iterrows():
            deal = _deal_from_staged_row(row)
            label = _mandate_option_label(
                deal=deal,
                status=str(row.get("status", "saved")),
                mandate_id=str(row.get("staged_mandate_id", "")),
            )
            options[label] = deal
    return options, current_label


def _slug_text(value: Any) -> str:
    return "".join(character if character.isalnum() else " " for character in str(value or "").lower()).strip()


def _selected_investor_id() -> str:
    investor = st.query_params.get("investor", "scbsm")
    if isinstance(investor, list):
        investor = investor[0] if investor else "scbsm"
    return str(investor or "scbsm").strip().lower() or "scbsm"


def _selected_contact_row(context) -> pd.Series | None:
    contacts = context.contacts.copy()
    if contacts.empty:
        return None
    contact_ids = contacts["contact_id"].astype(str).str.strip().str.lower()
    matches = contacts.loc[contact_ids.eq(_selected_investor_id())]
    if not matches.empty:
        return matches.iloc[0]
    return contacts.iloc[0]


def _contact_match_snapshot(row: pd.Series, context) -> dict[str, Any]:
    if str(row.get("contact_id", "")).strip().lower() == "scbsm":
        evaluation = context.scbsm_evaluation
        match_count = sum(
            1 for key in ["sector", "geography", "ticket", "cap_rate"] if evaluation["criteria"][key]["status"] == "Yes"
        )
        return {
            "match_count": match_count,
            "fit_label": evaluation["fit_label"],
            "sector_status": evaluation["criteria"]["sector"]["status"],
            "sector_reason": evaluation["criteria"]["sector"]["reason"],
            "geography_status": evaluation["criteria"]["geography"]["status"],
            "geography_reason": evaluation["criteria"]["geography"]["reason"],
            "ticket_status": evaluation["criteria"]["ticket"]["status"],
            "ticket_reason": evaluation["criteria"]["ticket"]["reason"],
            "yield_status": evaluation["criteria"]["cap_rate"]["status"],
            "yield_reason": evaluation["yield_accretion_text"],
        }

    asset_focus = str(row.get("asset_focus", "")).strip() or "Not specified"
    zone_focus = str(row.get("zone_focus", "")).strip()
    city_focus = str(row.get("city_focus", "")).strip()
    country_focus = str(row.get("country_focus", "")).strip() or "Not specified"
    sector_match = _slug_text(context.current_deal.asset_type) in _slug_text(asset_focus)
    sector_status = "Yes" if sector_match else "No"
    sector_reason = f"Mandate asset type = {context.current_deal.asset_type}. Contact focus = {asset_focus}."

    country_match = country_focus == "Not specified" or _slug_text(context.current_deal.country) == _slug_text(country_focus)
    local_focus_available = bool(zone_focus or city_focus)
    zone_match = _slug_text(context.current_deal.zone) in _slug_text(zone_focus) if zone_focus else False
    city_match = _slug_text(context.current_deal.city) in _slug_text(city_focus) if city_focus else False
    geography_match = country_match and (zone_match or city_match or not local_focus_available)
    geography_status = "Yes" if geography_match else "No"
    geography_focus = " / ".join([value for value in [zone_focus, city_focus, country_focus] if value]) or "Not specified"
    geography_reason = (
        f"Mandate location = {context.current_deal.zone or context.current_deal.city or context.current_deal.country}. "
        f"Contact geography focus = {geography_focus}."
    )

    min_ticket = pd.to_numeric(row.get("min_ticket_eur_mn"), errors="coerce")
    max_ticket = pd.to_numeric(row.get("max_ticket_eur_mn"), errors="coerce")
    if pd.isna(min_ticket) and pd.isna(max_ticket):
        ticket_status = "Not assessed"
        ticket_reason = "No ticket range is stored for this contact yet."
    else:
        lower_bound = 0.0 if pd.isna(min_ticket) else float(min_ticket)
        upper_bound = float("inf") if pd.isna(max_ticket) else float(max_ticket)
        ticket_match = lower_bound <= float(context.current_deal.ticket_eur_mn) <= upper_bound
        ticket_status = "Yes" if ticket_match else "No"
        ticket_reason = (
            f"Deal ticket = {_format_money_mn(context.current_deal.ticket_eur_mn)}. "
            f"Contact range = {_format_money_mn(min_ticket)} - {_format_money_mn(max_ticket)}."
        )

    min_yield = pd.to_numeric(row.get("min_target_yield_pct"), errors="coerce")
    max_yield = pd.to_numeric(row.get("max_target_yield_pct"), errors="coerce")
    if context.current_deal.cap_rate_pct is None:
        yield_status = "Not assessed"
        yield_reason = "No deal cap rate has been entered yet, so yield compatibility cannot be assessed."
    elif pd.isna(min_yield) and pd.isna(max_yield):
        yield_status = "Not assessed"
        yield_reason = "No target yield range is stored for this contact yet."
    else:
        lower_yield = 0.0 if pd.isna(min_yield) else float(min_yield)
        upper_yield = 100.0 if pd.isna(max_yield) else float(max_yield)
        yield_match = lower_yield <= float(context.current_deal.cap_rate_pct) <= upper_yield
        yield_status = "Yes" if yield_match else "No"
        yield_reason = (
            f"Deal cap rate = {_format_pct(context.current_deal.cap_rate_pct)}. "
            f"Contact target range = {_format_pct(min_yield)} - {_format_pct(max_yield)}."
        )
    statuses = [sector_status, geography_status, ticket_status, yield_status]
    assessed_count = sum(1 for status in statuses if status in {"Yes", "No"})
    match_count = sum(1 for status in statuses if status == "Yes")
    if match_count == assessed_count and match_count >= 3:
        fit_label = "Strong match"
    elif match_count >= 2:
        fit_label = "Possible match"
    else:
        fit_label = "Do not target"
    return {
        "match_count": match_count,
        "fit_label": fit_label,
        "sector_status": sector_status,
        "sector_reason": sector_reason,
        "geography_status": geography_status,
        "geography_reason": geography_reason,
        "ticket_status": ticket_status,
        "ticket_reason": ticket_reason,
        "yield_status": yield_status,
        "yield_reason": yield_reason,
    }


def _investor_contacts_frame(context, search_value: str) -> pd.DataFrame:
    contacts = context.contacts.copy()
    if contacts.empty:
        return pd.DataFrame(columns=["Contact", "Company", "Coverage", "Ticket range", "Stage", "Owner", "Action"])

    token = str(search_value or "").strip().lower()
    if token:
        mask = contacts.apply(
            lambda row: token
            in " ".join(
                [
                    str(row.get("full_name", "")),
                    str(row.get("company", "")),
                    str(row.get("asset_focus", "")),
                    str(row.get("zone_focus", "")),
                    str(row.get("city_focus", "")),
                ]
            ).lower(),
            axis=1,
        )
        contacts = contacts.loc[mask].copy()

    contacts["Coverage"] = contacts.apply(
        lambda row: f"{row.get('asset_focus', '')} | {row.get('zone_focus', '')}".strip(" |"),
        axis=1,
    )
    contacts["Ticket range"] = contacts.apply(
        lambda row: f"{_format_money_mn(row.get('min_ticket_eur_mn'))} - {_format_money_mn(row.get('max_ticket_eur_mn'))}",
        axis=1,
    )
    contacts["Stage"] = contacts["relationship_stage"].fillna("").map(lambda value: _badge(_title_case_status(value), "mint"))
    contacts["Action"] = contacts["contact_id"].map(
        lambda investor_id: f"<a href='?screen=investor-validation&investor={_escape(investor_id)}#investor-detail' class='adi-link-button adi-link-button--small'>View validation</a>"
    )
    display = contacts.rename(
        columns={
            "full_name": "Contact",
            "company": "Company",
            "owner": "Owner",
        }
    )
    return display[["Contact", "Company", "Coverage", "Ticket range", "Stage", "Owner", "Action"]]


def _investor_ranking_frame(context, top_n: int = INVESTOR_VALIDATION_TOP_N) -> pd.DataFrame:
    contacts = context.contacts.copy()
    if contacts.empty:
        return pd.DataFrame(
            columns=["Rank", "Investor", "Sector", "Geography", "Ticket", "Yield", "Criteria in common", "Fit", "View more infos"]
        )

    ranking_rows: list[dict[str, Any]] = []
    for _, row in contacts.iterrows():
        snapshot = _contact_match_snapshot(row, context)
        rank_tone = (
            "success"
            if snapshot["fit_label"] == "Strong match"
            else ("warning" if snapshot["fit_label"] == "Possible match" else "danger")
        )
        ranking_rows.append(
            {
                "contact_id": str(row.get("contact_id", "")).strip().lower(),
                "Investor": row.get("company", row.get("full_name", "Investor")),
                "Sector": _criteria_status_badge(snapshot["sector_status"]),
                "Geography": _criteria_status_badge(snapshot["geography_status"]),
                "Ticket": _criteria_status_badge(snapshot["ticket_status"]),
                "Yield": _criteria_status_badge(snapshot["yield_status"]),
                "match_count": int(snapshot["match_count"]),
                "Criteria in common": _badge(f"{snapshot['match_count']}/4", rank_tone),
                "Fit": _badge(snapshot["fit_label"], rank_tone),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(["match_count", "Investor"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1
    ranking["View more infos"] = ranking["contact_id"].map(
        lambda investor_id: f"<a href='?screen=investor-validation&investor={_escape(investor_id)}#investor-detail' class='adi-link-button adi-link-button--small'>View more infos</a>"
    )
    return ranking[["Rank", "Investor", "Sector", "Geography", "Ticket", "Yield", "Criteria in common", "Fit", "View more infos"]]


def _criteria_dataframe(evaluation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Criterion": item["label"],
                "Status": item["status"],
                "Reason": item["reason"],
                "Weight": item["weight"],
                "Earned": item["earned"],
            }
            for item in evaluation["criteria"].values()
        ]
    )


def _profile_edit_history_frame(profile_edits: pd.DataFrame) -> pd.DataFrame:
    if profile_edits.empty:
        return pd.DataFrame(columns=["Edited at", "Edited by", "Changed fields", "Note"])
    display = profile_edits.copy()
    display["edited_at_utc"] = pd.to_datetime(display["edited_at_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    display["changed_fields_json"] = display["changed_fields_json"].map(
        lambda value: ", ".join(json.loads(value)) if str(value).strip() else ""
    )
    return display.rename(
        columns={
            "edited_at_utc": "Edited at",
            "edited_by": "Edited by",
            "changed_fields_json": "Changed fields",
            "note": "Note",
        }
    )[["Edited at", "Edited by", "Changed fields", "Note"]]


def _small_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["Date", "Action"])
    display = history.copy().head(6)
    display["Date"] = pd.to_datetime(display["event_date"], errors="coerce").dt.date.astype(str)
    display["Action"] = display["touchpoint_type"].map(_touchpoint_display_name)
    return display[["Date", "Action"]]


def _follow_up_state(history: pd.DataFrame, row: pd.Series) -> tuple[str, str]:
    if str(row.get("touchpoint_type", "")) != "teaser_sent":
        return ("-", "neutral")

    event_dt = pd.to_datetime(row.get("event_date"), errors="coerce")
    if pd.isna(event_dt):
        return ("TBD", "neutral")

    progress_actions = {"response", "nda_sent", "nda_signed", "meeting_held", "outcome"}
    history_dates = pd.to_datetime(history["event_date"], errors="coerce")
    progressed = not history.loc[
        history["touchpoint_type"].isin(progress_actions)
        & (
            history_dates.gt(event_dt)
            | (history_dates.eq(event_dt) & history["touchpoint_type"].isin(progress_actions))
        )
    ].empty
    if progressed:
        return ("Not needed", "success")

    due_date = (event_dt.normalize() + pd.Timedelta(days=FOLLOW_UP_DELAY_DAYS)).date()
    tone = "danger" if due_date < date.today() else "warning"
    return (f"Due {due_date.isoformat()}", tone)


def _follow_up_text(history: pd.DataFrame, row: pd.Series) -> str:
    text, _ = _follow_up_state(history, row)
    return text


def _follow_up_badge(history: pd.DataFrame, row: pd.Series) -> str:
    text, tone = _follow_up_state(history, row)
    if text == "-":
        return "—"
    return _badge(text, tone)


def _history_detail_badge(touchpoint_type: str, status_value: str) -> str:
    if touchpoint_type == "teaser_sent" and status_value == "sent":
        return _badge("Sent", "mint")
    if status_value == "sent":
        return _badge("Sent", "mint")
    if status_value in {"yes", "signed", "held", "shortlisted", "won"}:
        return _badge(_title_case_status(status_value), "success")
    if status_value in {"no", "declined", "lost", "not signed"}:
        return _badge(_title_case_status(status_value), "danger")
    if status_value == "no reply":
        return _badge("No Reply", "warning")
    return _badge(_title_case_status(status_value), "neutral")


def _outreach_history_frame(history: pd.DataFrame, *, include_example_if_empty: bool = False) -> tuple[pd.DataFrame, bool]:
    if history.empty and include_example_if_empty:
        example_date = date.today().isoformat()
        example = pd.DataFrame(
            [
                {
                    "Date": example_date,
                    "Investor": "SCBSM",
                    "Action": "Teaser sent",
                    "Detail": _badge("Sent", "mint"),
                    "Follow-up": _badge(
                        f"Due {(pd.Timestamp(example_date) + pd.Timedelta(days=FOLLOW_UP_DELAY_DAYS)).date().isoformat()}",
                        "warning",
                    ),
                }
            ]
        )
        return example, True

    if history.empty:
        return pd.DataFrame(columns=["Date", "Investor", "Action", "Detail", "Follow-up"]), False

    display = history.copy()
    display["Date"] = pd.to_datetime(display["event_date"], errors="coerce").dt.date.astype(str)
    display["Investor"] = "SCBSM"
    display["Action"] = display["touchpoint_type"].map(_touchpoint_display_name)
    display["Detail"] = display.apply(
        lambda row: _history_detail_badge(str(row.get("touchpoint_type", "")), str(row.get("status_value", ""))),
        axis=1,
    )
    display["Follow-up"] = display.apply(lambda row: _follow_up_badge(history, row), axis=1)
    return display[["Date", "Investor", "Action", "Detail", "Follow-up"]], False


def _normalise_screen(value: Any) -> str:
    selected = str(value or "dashboard").strip().lower()
    valid = {slug for slug, _ in SCREENS}
    return selected if selected in valid else "dashboard"


def _current_screen() -> str:
    value = st.query_params.get("screen", "dashboard")
    if isinstance(value, list):
        value = value[0] if value else "dashboard"
    return _normalise_screen(value)


def _navigate(screen: str) -> None:
    st.query_params.clear()
    st.query_params["screen"] = screen
    st.rerun()


def _build_live_comparable_query(context) -> ComparableQuery:
    asset_type = "Mixed Use" if context.current_deal.asset_type == "Mixed Commercial" else context.current_deal.asset_type
    transaction_year = pd.to_datetime(context.current_deal.transaction_date, errors="coerce").year
    return ComparableQuery(
        asset_type=asset_type,
        country=context.current_deal.country or "France",
        city=context.current_deal.city or context.current_deal.zone,
        size_sqm=float(context.current_deal.size_sqm),
        transaction_year=int(transaction_year) if pd.notna(transaction_year) else None,
        cap_rate_pct=context.current_deal.cap_rate_pct,
    )


def _widened_reference_results(query: ComparableQuery, top_k: int = 10) -> pd.DataFrame:
    frame = load_prepared_comparables()
    pool = frame.loc[frame["primary_asset_type"].eq(query.asset_type)].copy()
    if pool.empty:
        return pd.DataFrame()

    pool["country_match"] = pool["country"].eq(query.country).astype(float)
    city_token = "".join(character if character.isalnum() else " " for character in str(query.city).lower()).strip()
    pool["city_match"] = (
        pool["asset_city"]
        .astype(str)
        .map(lambda value: "".join(character if character.isalnum() else " " for character in value.lower()).strip())
        .eq(city_token)
        .astype(float)
        if city_token
        else 0.0
    )
    if query.size_sqm is not None and query.size_sqm > 0:
        query_log_size = float(np.log(max(query.size_sqm, 1.0)))
        pool["size_score"] = 24.0 / (
            1.0 + 3.0 * np.abs(query_log_size - np.log(pool["TOTAL SIZE (SQ. M.)"].clip(lower=1.0)))
        )
    else:
        pool["size_score"] = 0.0

    if query.transaction_year is not None:
        pool["year_gap"] = (pool["transaction_year"].astype(float) - float(query.transaction_year)).abs()
        pool["year_score"] = 8.0 / (1.0 + pool["year_gap"].fillna(20.0))
    else:
        pool["year_gap"] = np.nan
        pool["year_score"] = 0.0

    pool["similarity_score"] = 54.0 + 18.0 * pool["country_match"] + 6.0 * pool["city_match"] + pool["size_score"] + pool["year_score"]
    pool = pool.sort_values(
        ["country_match", "city_match", "size_score", "year_gap", "DEAL DATE"],
        ascending=[False, False, False, True, False],
        na_position="last",
    )

    buckets = []
    preferred_countries = [query.country, "Germany", "United Kingdom"]
    for country in preferred_countries:
        take = 6 if country == query.country else 2
        buckets.append(pool.loc[pool["country"].eq(country)].head(take))
    combined = pd.concat(buckets, ignore_index=False).drop_duplicates(subset=["DEAL ID"]).copy()
    if len(combined) < top_k:
        filler = pool.loc[~pool["DEAL ID"].isin(combined["DEAL ID"])].head(top_k - len(combined))
        combined = pd.concat([combined, filler], ignore_index=False)
    combined = combined.head(top_k).reset_index(drop=True)
    combined["retrieval_scope"] = "Type x Europe (widened references)"
    return combined


def _match_badge(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return _badge("N/A", "neutral")
    value = min(100, max(0, int(round(float(score)))))
    if value >= 92:
        tone = "success"
    elif value >= 80:
        tone = "mint"
    else:
        tone = "warning"
    return _badge(f"{value}%", tone)


def _comparables_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Rank", "Property", "Country", "Size", "EUR / sqm", "Year", "Match %"])
    display = frame.copy().reset_index(drop=True)
    display["Rank"] = display.index + 1
    display["Property"] = display["DEAL NAME"].fillna(display["DEAL ID"]).astype(str)
    display["Country"] = display["country"]
    display["Size"] = pd.to_numeric(display["TOTAL SIZE (SQ. M.)"], errors="coerce").round(0).map(
        lambda value: f"{int(value):,} sqm" if pd.notna(value) else "N/A"
    )
    display["EUR / sqm"] = pd.to_numeric(display["price_per_sqm_winsorized_eur"], errors="coerce").round(0).map(
        lambda value: f"EUR {int(value):,}" if pd.notna(value) else "N/A"
    )
    display["Year"] = pd.to_numeric(display["transaction_year"], errors="coerce").fillna(0).astype(int).replace(0, "")
    display["Match %"] = display["similarity_score"].map(_match_badge)
    return display[["Rank", "Property", "Country", "Size", "EUR / sqm", "Year", "Match %"]]


def _profile_attribute_frame(profile: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Attribute": "Type", "Value": profile["investor_type"]},
            {"Attribute": "Sector", "Value": ", ".join(profile["sector_focus"])},
            {
                "Attribute": "Ticket size",
                "Value": (
                    f"{_format_money_mn(profile['ticket_min_eur_mn'])} - "
                    f"{_format_money_mn(profile['ticket_max_eur_mn'])}"
                ),
            },
            {
                "Attribute": "Portfolio cap rate",
                "Value": f"{_format_pct(profile['portfolio_cap_rate_pct'])} {_badge('Auto', 'mint')}",
            },
            {
                "Attribute": "Portfolio value",
                "Value": f"{_format_money_mn(profile['portfolio_value_eur_mn'])} {_badge('Auto', 'mint')}",
            },
            {
                "Attribute": "LTV",
                "Value": f"{_format_pct(profile['ltv_pct'])} {_badge('Auto', 'mint')}",
            },
        ]
    )


def _contact_profile_frame(contact: pd.Series) -> pd.DataFrame:
    geography = " / ".join(
        [value for value in [contact.get("zone_focus", ""), contact.get("city_focus", ""), contact.get("country_focus", "")] if value]
    )
    return pd.DataFrame(
        [
            {"Attribute": "Type", "Value": "Alantra contact"},
            {"Attribute": "Sector", "Value": contact.get("asset_focus", "Not specified")},
            {"Attribute": "Geography", "Value": geography or "Not specified"},
            {
                "Attribute": "Ticket size",
                "Value": (
                    f"{_format_money_mn(contact.get('min_ticket_eur_mn'))} - "
                    f"{_format_money_mn(contact.get('max_ticket_eur_mn'))}"
                ),
            },
            {
                "Attribute": "Target yield",
                "Value": (
                    f"{_format_pct(contact.get('min_target_yield_pct'))} - "
                    f"{_format_pct(contact.get('max_target_yield_pct'))}"
                ),
            },
            {"Attribute": "Stage", "Value": _title_case_status(str(contact.get("relationship_stage", ""))) or "Not specified"},
            {"Attribute": "Owner", "Value": contact.get("owner", "Not assigned") or "Not assigned"},
        ]
    )


def _criteria_match_frame(evaluation: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in ["sector", "geography", "ticket", "cap_rate"]:
        criterion = evaluation["criteria"][key]
        tone = "success" if criterion["status"] == "Yes" else ("warning" if criterion["status"] == "Not assessed" else "danger")
        rows.append(
            {
                "Criterion": criterion["label"].replace(" match", ""),
                "Assessment": _badge(criterion["status"], tone),
                "Explanation": criterion["reason"] if key != "cap_rate" else evaluation["yield_accretion_text"],
            }
        )
    return pd.DataFrame(rows)


def _generic_criteria_match_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Criterion": "Sector",
                "Assessment": _criteria_status_badge(snapshot["sector_status"]),
                "Explanation": snapshot["sector_reason"],
            },
            {
                "Criterion": "Geography",
                "Assessment": _criteria_status_badge(snapshot["geography_status"]),
                "Explanation": snapshot["geography_reason"],
            },
            {
                "Criterion": "Ticket",
                "Assessment": _criteria_status_badge(snapshot["ticket_status"]),
                "Explanation": snapshot["ticket_reason"],
            },
            {
                "Criterion": "Yield",
                "Assessment": _criteria_status_badge(snapshot["yield_status"]),
                "Explanation": snapshot["yield_reason"],
            },
        ]
    )


def _negative_example_evaluation(context) -> dict[str, Any]:
    what_if = build_deal_input(
        {
            "mandate_name": "Normandy Hotel What-if",
            "asset_type": "Hotel",
            "country": "France",
            "zone": "Normandy",
            "city": "Normandy",
            "price_min_eur_mn": 8.0,
            "price_max_eur_mn": 8.0,
            "ticket_eur_mn": 8.0,
            "size_sqm": 3000.0,
            "cap_rate_pct": None,
        }
    )
    return score_scbsm_for_deal(
        deal=what_if,
        assets=context.assets,
        events=context.events,
        profile=derive_scbsm_profile(context.assets),
    )


def _inject_theme_css() -> None:
    st.markdown(
        """
<style>
:root {
  --adi-bg: #eef1eb;
  --adi-surface: #ffffff;
  --adi-primary: #1f4738;
  --adi-primary-ink: #ffffff;
  --adi-mint: #edf6ee;
  --adi-border: #d6ddd4;
  --adi-text: #213027;
  --adi-muted: #617166;
  --adi-success-bg: #e7f4ea;
  --adi-success-ink: #2d6a3f;
  --adi-warning-bg: #f6ecd3;
  --adi-warning-ink: #8b6b1c;
  --adi-danger-bg: #f6e3e2;
  --adi-danger-ink: #8d4543;
}
html, body, [class*="css"] {
  font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
  background: linear-gradient(180deg, #edf1eb 0%, #f5f6f2 100%);
  color: var(--adi-text);
}
[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}
[data-testid="stHeader"] {
  background: rgba(245, 246, 242, 0.88);
}
.block-container {
  max-width: 980px;
  padding-top: 4.6rem;
  padding-bottom: 2.5rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--adi-surface);
  border: 1px solid var(--adi-border);
  border-radius: 16px;
  box-shadow: 0 8px 20px rgba(31, 71, 56, 0.04);
}
.adi-shell {
  background: var(--adi-primary);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-top: 0.25rem;
  margin-bottom: 1.1rem;
}
.adi-brand {
  color: rgba(255, 255, 255, 0.96);
  font-size: 1.04rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}
.adi-nav {
  display: flex;
  gap: 0.9rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.adi-nav a {
  color: rgba(255, 255, 255, 0.75);
  text-decoration: none;
  font-size: 0.93rem;
  font-weight: 500;
  border-bottom: 2px solid transparent;
  padding-bottom: 0.2rem;
}
.adi-nav a:hover,
.adi-nav a.adi-nav-active {
  color: rgba(255, 255, 255, 1);
  border-bottom-color: #d1e6d4;
}
.adi-section-intro {
  margin: 0 0 1rem 0;
}
.adi-section-intro h1 {
  margin: 0;
  color: var(--adi-text);
  font-size: 1.72rem;
  font-weight: 650;
  letter-spacing: -0.03em;
}
.adi-section-intro p {
  margin: 0.38rem 0 0 0;
  max-width: 740px;
  color: var(--adi-muted);
  font-size: 0.97rem;
  line-height: 1.55;
}
.adi-grid {
  display: grid;
  gap: 14px;
}
.adi-grid--three {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.adi-grid--two {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.adi-card {
  background: var(--adi-surface);
  border: 1px solid var(--adi-border);
  border-radius: 14px;
  padding: 1rem 1.05rem;
}
.adi-card--accent {
  background: var(--adi-mint);
}
.adi-card-label {
  color: var(--adi-muted);
  font-size: 0.73rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.55rem;
}
.adi-card-value {
  color: var(--adi-text);
  font-size: 1.65rem;
  font-weight: 650;
  line-height: 1.15;
}
.adi-card-note,
.adi-note {
  color: var(--adi-muted);
  font-size: 0.9rem;
  line-height: 1.5;
}
.adi-subtitle {
  color: var(--adi-text);
  font-size: 1.05rem;
  font-weight: 650;
  margin-bottom: 0.25rem;
}
.adi-field-label {
  margin: 0 0 0.28rem 0;
  color: var(--adi-muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
}
.adi-link-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  padding: 0.55rem 0.9rem;
  border-radius: 10px;
  border: 1px solid var(--adi-primary);
  color: var(--adi-primary);
  background: #ffffff;
  text-decoration: none;
  font-size: 0.92rem;
  font-weight: 600;
}
.adi-link-button--primary {
  background: var(--adi-primary);
  color: var(--adi-primary-ink);
}
.adi-link-button--small {
  min-height: 30px;
  padding: 0.38rem 0.7rem;
  font-size: 0.84rem;
  border-radius: 9px;
}
.adi-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.22rem 0.56rem;
  font-size: 0.77rem;
  font-weight: 650;
  white-space: nowrap;
}
.adi-badge--neutral {
  background: #eef2ef;
  color: #4c5a51;
}
.adi-badge--mint {
  background: var(--adi-mint);
  color: #35634b;
}
.adi-badge--success {
  background: var(--adi-success-bg);
  color: var(--adi-success-ink);
}
.adi-badge--warning {
  background: var(--adi-warning-bg);
  color: var(--adi-warning-ink);
}
.adi-badge--danger {
  background: var(--adi-danger-bg);
  color: var(--adi-danger-ink);
}
.adi-banner {
  border-radius: 12px;
  padding: 0.95rem 1rem;
  margin-bottom: 0.95rem;
  border: 1px solid var(--adi-border);
}
.adi-banner--success {
  background: var(--adi-success-bg);
  border-color: #cfe4d4;
}
.adi-banner--warning {
  background: var(--adi-warning-bg);
  border-color: #e8d7a4;
}
.adi-banner--danger {
  background: var(--adi-danger-bg);
  border-color: #e7c8c6;
}
.adi-banner-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--adi-text);
}
.adi-banner-detail {
  margin-top: 0.25rem;
  font-size: 0.9rem;
  color: var(--adi-muted);
  line-height: 1.5;
}
.adi-divider {
  border-top: 1px dashed var(--adi-border);
  margin: 0.65rem 0 0.25rem 0;
}
.adi-table-wrap {
  width: 100%;
  overflow-x: auto;
}
.adi-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.93rem;
}
.adi-table th {
  text-align: left;
  color: var(--adi-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0.76rem 0.72rem;
  border-bottom: 1px solid var(--adi-border);
  font-weight: 650;
}
.adi-table td {
  padding: 0.78rem 0.72rem;
  border-bottom: 1px solid #edf2ec;
  color: var(--adi-text);
  vertical-align: top;
}
.adi-table tbody tr:last-child td {
  border-bottom: none;
}
.adi-inline-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
  border-color: var(--adi-border) !important;
  border-radius: 10px !important;
  background: #ffffff !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
  border-color: #5f8b72 !important;
  box-shadow: 0 0 0 1px rgba(95, 139, 114, 0.2);
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
.stFormSubmitButton button {
  width: 100%;
  border-radius: 10px;
  border: 1px solid var(--adi-primary);
  background: var(--adi-primary);
  color: var(--adi-primary-ink);
  font-weight: 650;
  min-height: 40px;
}
@media (max-width: 900px) {
  .adi-grid--three,
  .adi-grid--two {
    grid-template-columns: 1fr;
  }
  .adi-shell {
    flex-direction: column;
    align-items: flex-start;
  }
  .adi-nav {
    justify-content: flex-start;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_shell() -> None:
    active = _current_screen()
    links = []
    for slug, label in SCREENS:
        active_class = "adi-nav-active" if slug == active else ""
        links.append(f"<a class='{active_class}' href='?screen={slug}'>{_escape(label)}</a>")
    st.markdown(
        (
            "<div class='adi-shell'>"
            "<div class='adi-brand'>Alantra · Deal Intelligence</div>"
            f"<div class='adi-nav'>{''.join(links)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_dashboard_screen(context) -> None:
    active_mandates = max(1, len(context.staged_mandates))
    _section_intro(
        "Dashboard",
        "A mandate-level starting point for investor screening, comparable retrieval, and outreach execution.",
    )

    st.markdown(
        (
            "<div class='adi-grid adi-grid--three'>"
            f"{_card_html('Active mandates', str(active_mandates), note='Live and saved mock-db mandates', accent=True)}"
            f"{_card_html('Investor registry', '1', note='Structured investors in scope')}"
            f"{_card_html('Comparables database', f'{len(load_prepared_comparables()):,}', note='European single-asset sample')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>Mandates</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='adi-note'>The system starts from a mandate, then flows into investor fit validation and comparable retrieval.</div>",
            unsafe_allow_html=True,
        )
        rows = [
            {
                "Name": context.current_deal.mandate_name,
                "Type": context.current_deal.asset_type,
                "Location": context.current_deal.zone or context.current_deal.country,
                "Status": _badge("Current", "success"),
                "Action": _nav_button("investor-validation", "Open", small=True),
            }
        ]
        if not context.staged_mandates.empty:
            for _, row in context.staged_mandates.head(5).iterrows():
                rows.append(
                    {
                        "Name": row["mandate_name"],
                        "Type": row["asset_type"],
                        "Location": row["zone"] or row["country"],
                        "Status": _badge(str(row.get("status", "saved")).title(), "mint"),
                        "Action": _badge("Saved in mock DB", "neutral"),
                    }
                )
        mandate_frame = pd.DataFrame(rows)
        st.markdown(_table_html(mandate_frame), unsafe_allow_html=True)
        if not context.staged_mandates.empty:
            option_lookup = {
                _mandate_option_label(
                    deal=_deal_from_staged_row(row),
                    status=str(row.get("status", "saved")),
                    mandate_id=str(row.get("staged_mandate_id", "")),
                ): str(row["staged_mandate_id"])
                for _, row in context.staged_mandates.iterrows()
            }
            selected_label = st.selectbox("Saved mock mandate", options=list(option_lookup.keys()))
            if st.button("Load saved mandate"):
                payload = load_staged_mandate_into_working_set(option_lookup[selected_label])
                _set_current_deal_state(payload)
                _navigate("investor-validation")
        st.markdown(_nav_button("new-mandate", "+ New Mandate", primary=True), unsafe_allow_html=True)


def _render_new_mandate_screen() -> None:
    _ensure_current_deal_state()
    _section_intro(
        "New Mandate",
        "Create a structured sell-side mandate. Submitting the form runs investor validation and comparable retrieval from the same intake.",
    )

    with st.container(border=True):
        with st.form("new_mandate_form"):
            row1 = st.columns(2)
            with row1[0]:
                _field_label("Mandate name")
                st.text_input("Mandate name", key=DEAL_FIELD_KEYS["mandate_name"], label_visibility="collapsed")
            with row1[1]:
                _field_label("Lead banker")
                st.text_input("Lead banker", key=LEAD_BANKER_KEY, label_visibility="collapsed")

            row2 = st.columns(2)
            with row2[0]:
                _field_label("Asset type")
                st.selectbox(
                    "Asset type",
                    options=["Office", "Retail", "Mixed Commercial", "Industrial", "Hotel", "Residential", "Land"],
                    key=DEAL_FIELD_KEYS["asset_type"],
                    label_visibility="collapsed",
                )
            with row2[1]:
                _field_label("Location")
                st.text_input("Location", key=DEAL_FIELD_KEYS["zone"], label_visibility="collapsed")

            row3 = st.columns(2)
            with row3[0]:
                _field_label("Size in sqm")
                st.number_input(
                    "Size in sqm",
                    min_value=100.0,
                    step=100.0,
                    key=DEAL_FIELD_KEYS["size_sqm"],
                    label_visibility="collapsed",
                )
            with row3[1]:
                _field_label("Price range min in EUR m")
                st.number_input(
                    "Price range min",
                    min_value=0.5,
                    step=1.0,
                    key=DEAL_FIELD_KEYS["price_min_eur_mn"],
                    label_visibility="collapsed",
                )

            row4 = st.columns(2)
            with row4[0]:
                _field_label("Price range max in EUR m")
                st.number_input(
                    "Price range max",
                    min_value=0.5,
                    step=1.0,
                    key=DEAL_FIELD_KEYS["price_max_eur_mn"],
                    label_visibility="collapsed",
                )
            with row4[1]:
                _field_label("Country")
                st.text_input("Country", key=DEAL_FIELD_KEYS["country"], label_visibility="collapsed")

            st.markdown("<div class='adi-divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='adi-subtitle'>Optional - income fields</div>", unsafe_allow_html=True)

            row5 = st.columns(2)
            with row5[0]:
                _field_label("Cap rate")
                st.checkbox("Cap rate available", key=DEAL_FIELD_KEYS["cap_rate_known"])
                if st.session_state[DEAL_FIELD_KEYS["cap_rate_known"]]:
                    st.number_input(
                        "Cap rate",
                        min_value=1.0,
                        max_value=15.0,
                        step=0.05,
                        key=DEAL_FIELD_KEYS["cap_rate_pct"],
                        label_visibility="collapsed",
                    )
            with row5[1]:
                _field_label("NOI")
                st.checkbox("NOI available", key=DEAL_FIELD_KEYS["noi_known"])
                if st.session_state[DEAL_FIELD_KEYS["noi_known"]]:
                    st.number_input(
                        "NOI",
                        min_value=0.1,
                        step=0.1,
                        key=DEAL_FIELD_KEYS["noi_eur_mn"],
                        label_visibility="collapsed",
                    )

            row6 = st.columns(2)
            with row6[0]:
                _field_label("Building grade")
                st.text_input("Building grade", key=DEAL_FIELD_KEYS["building_grade"], label_visibility="collapsed")
            with row6[1]:
                _field_label("Reference date")
                st.date_input("Reference date", key=DEAL_FIELD_KEYS["transaction_date"], label_visibility="collapsed")

            submitted = st.form_submit_button("Create → Run Investor Match + Comps")

        st.markdown(
            "<div class='adi-note'>Submitting saves the mandate into the local mock SQLite store, then opens investor validation with the resulting fit rationale.</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        payload = _read_current_deal_state()
        payload["city"] = str(payload.get("city", "")).strip() or str(payload["zone"]).strip()
        if payload["price_min_eur_mn"] > payload["price_max_eur_mn"]:
            st.error("Price range min cannot be above price range max.")
            return
        create_mock_mandate(payload=payload, lead_banker=str(st.session_state.get(LEAD_BANKER_KEY, "")))
        # The form widgets already wrote their values into session state.
        # Only persist the derived hidden city field here, otherwise Streamlit
        # raises when a widget-bound key is reassigned in the same run.
        st.session_state[DEAL_FIELD_KEYS["city"]] = payload["city"]
        _navigate("investor-validation")


def _render_registry_admin(context) -> None:
    profile = context.scbsm_profile
    raw_metadata = load_profile_metadata() or profile
    with st.expander("Admin - structured profile controls", expanded=False):
        st.markdown(
            "<div class='adi-note'>These controls keep the prototype aligned with the structured investor-profile requirement without dominating the main registry view.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Refresh from public data"):
            try:
                refresh_scbsm_profile_from_public_data(edited_by=profile["owner"])
            except Exception as error:  # noqa: BLE001
                st.error(f"Auto-update failed. Last known values retained. {error}")
            else:
                st.success("SCBSM profile updated from the committed public-data snapshot.")
                st.rerun()

        with st.form("registry_admin_form"):
            row1 = st.columns(2)
            with row1[0]:
                _field_label("Investor name")
                name = st.text_input("Investor name", value=str(raw_metadata.get("name", profile["name"])), label_visibility="collapsed")
            with row1[1]:
                _field_label("Firm")
                firm = st.text_input("Firm", value=str(raw_metadata.get("firm", profile["firm"])), label_visibility="collapsed")

            row2 = st.columns(2)
            with row2[0]:
                _field_label("Investor type")
                investor_type = st.text_input(
                    "Investor type",
                    value=str(raw_metadata.get("investor_type", profile["investor_type"])),
                    label_visibility="collapsed",
                )
            with row2[1]:
                _field_label("Sector focus")
                sector_focus = st.text_input(
                    "Sector focus",
                    value=", ".join(raw_metadata.get("sector_focus", profile["sector_focus"])),
                    label_visibility="collapsed",
                )

            row3 = st.columns(2)
            with row3[0]:
                _field_label("Geographic focus")
                geographic_focus = st.text_input(
                    "Geographic focus",
                    value=str(raw_metadata.get("geographic_focus", profile["geographic_focus"])),
                    label_visibility="collapsed",
                )
            with row3[1]:
                _field_label("City focus")
                city_focus = st.text_input(
                    "City focus",
                    value=str(raw_metadata.get("city_focus", profile["city_focus"])),
                    label_visibility="collapsed",
                )

            row4 = st.columns(2)
            with row4[0]:
                _field_label("Ticket min in EUR m")
                ticket_min = st.number_input(
                    "Ticket min",
                    min_value=0.5,
                    value=float(raw_metadata.get("ticket_min_eur_mn", profile["ticket_min_eur_mn"])),
                    step=1.0,
                    label_visibility="collapsed",
                )
            with row4[1]:
                _field_label("Ticket max in EUR m")
                ticket_max = st.number_input(
                    "Ticket max",
                    min_value=0.5,
                    value=float(raw_metadata.get("ticket_max_eur_mn", profile["ticket_max_eur_mn"])),
                    step=1.0,
                    label_visibility="collapsed",
                )

            row5 = st.columns(3)
            with row5[0]:
                _field_label("Portfolio cap rate")
                portfolio_cap_rate = st.number_input(
                    "Portfolio cap rate",
                    min_value=0.1,
                    value=float(raw_metadata.get("portfolio_cap_rate_pct", profile["portfolio_cap_rate_pct"])),
                    step=0.05,
                    label_visibility="collapsed",
                )
            with row5[1]:
                _field_label("Portfolio value in EUR m")
                portfolio_value = st.number_input(
                    "Portfolio value",
                    min_value=0.1,
                    value=float(raw_metadata.get("portfolio_value_eur_mn", profile["portfolio_value_eur_mn"])),
                    step=1.0,
                    label_visibility="collapsed",
                )
            with row5[2]:
                _field_label("LTV")
                ltv_pct = st.number_input(
                    "LTV",
                    min_value=0.1,
                    value=float(raw_metadata.get("ltv_pct", profile["ltv_pct"])),
                    step=0.05,
                    label_visibility="collapsed",
                )

            row6 = st.columns(2)
            with row6[0]:
                _field_label("Last updated")
                last_updated = st.text_input(
                    "Last updated",
                    value=str(raw_metadata.get("last_updated", profile["last_updated"])),
                    label_visibility="collapsed",
                )
            with row6[1]:
                _field_label("Source tag")
                source_tag = st.text_input(
                    "Source tag",
                    value=str(raw_metadata.get("source_tag", profile["source_tag"])),
                    label_visibility="collapsed",
                )

            saved = st.form_submit_button("Save structured profile")

        if saved:
            payload = raw_metadata | {
                "investor_id": "scbsm",
                "company": firm.strip(),
                "display_name": name.strip(),
                "title": raw_metadata.get("title", profile["title"]),
                "name": name.strip(),
                "firm": firm.strip(),
                "investor_type": investor_type.strip(),
                "sector_focus": [item.strip() for item in sector_focus.split(",") if item.strip()],
                "geographic_focus": geographic_focus.strip(),
                "city_focus": city_focus.strip(),
                "country_focus": str(raw_metadata.get("country_focus", profile["country_focus"])).strip() or "France",
                "ticket_min_eur_mn": float(ticket_min),
                "ticket_max_eur_mn": float(ticket_max),
                "portfolio_cap_rate_pct": float(portfolio_cap_rate),
                "portfolio_value_eur_mn": float(portfolio_value),
                "ltv_pct": float(ltv_pct),
                "last_updated": last_updated.strip(),
                "source_tag": source_tag.strip(),
                "preferred_channel": str(raw_metadata.get("preferred_channel", profile["preferred_channel"])),
                "owner": str(raw_metadata.get("owner", profile["owner"])),
            }
            missing = validate_profile_payload(payload)
            if payload["ticket_min_eur_mn"] > payload["ticket_max_eur_mn"]:
                st.error("Ticket min cannot exceed ticket max.")
            elif missing:
                st.error("Mandatory fields missing: " + ", ".join(missing))
            else:
                changed = save_profile_metadata(
                    payload=payload,
                    edited_by=str(payload["owner"]),
                    note="Manual profile update from registry admin controls.",
                )
                if changed:
                    st.success("Investor profile saved.")
                    st.rerun()
                else:
                    st.info("No profile fields changed.")

        history_frame = _profile_edit_history_frame(context.profile_edits)
        if history_frame.empty:
            st.markdown("<div class='adi-note'>No profile edits logged yet.</div>", unsafe_allow_html=True)
        else:
            st.markdown(_table_html(history_frame.head(5)), unsafe_allow_html=True)


def _render_investor_registry_screen(context) -> None:
    _section_intro(
        "Investor Contacts",
        "A structured Alantra-held list of potential investor contacts. Validation against the live mandate sits on the investor-validation screen.",
    )

    search_value = st.text_input("Search investor contacts", value="", placeholder="Search by company, focus, geography, or contact")
    contacts_frame = _investor_contacts_frame(context, search_value)
    populated_contacts = len(context.contacts.index)

    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>Alantra investor contacts</div>", unsafe_allow_html=True)
        st.markdown(_table_html(contacts_frame), unsafe_allow_html=True)
        st.markdown(
            (
                f"<div class='adi-note'>The registry currently contains {populated_contacts} populated "
                f"contact{'s' if populated_contacts != 1 else ''}. Right now that means SCBSM only; "
                "additional Alantra-held investor contacts will appear here as the registry grows.</div>"
            ),
            unsafe_allow_html=True,
        )

    _render_registry_admin(context)



def _render_investor_validation_screen(context) -> None:
    _section_intro(
        "Investor Validation",
        "The decision engine first ranks investors against the live mandate, then opens a detailed validation view for the selected investor.",
    )

    ranking_frame = _investor_ranking_frame(context, top_n=INVESTOR_VALIDATION_TOP_N)
    selected_contact = _selected_contact_row(context)
    selected_investor_id = str(selected_contact.get("contact_id", "scbsm")).strip().lower() if selected_contact is not None else "scbsm"
    selected_investor_name = (
        str(selected_contact.get("company") or selected_contact.get("full_name") or "SCBSM")
        if selected_contact is not None
        else "SCBSM"
    )
    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>Top investors for this mandate</div>", unsafe_allow_html=True)
        st.markdown(
            (
                f"<div class='adi-note'>Showing up to {INVESTOR_VALIDATION_TOP_N} potential investors from the Alantra registry, "
                "ranked by how many of the four core criteria they have in common with the live mandate: sector, geography, ticket, and yield.</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(_table_html(ranking_frame), unsafe_allow_html=True)

    history = get_scbsm_history(context.events)
    evaluation = context.scbsm_evaluation
    profile = context.scbsm_profile

    st.markdown("<a id='investor-detail'></a>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"<div class='adi-subtitle'>Investor drill-down: {_escape(selected_investor_name)}</div>",
            unsafe_allow_html=True,
        )
        if selected_investor_id != "scbsm":
            st.markdown(
                "<div class='adi-note'>This drill-down uses the structured Alantra contact record. The deeper auto-derived profile and touchpoint history remain fully wired for SCBSM in the prototype.</div>",
                unsafe_allow_html=True,
            )

    if selected_investor_id != "scbsm" and selected_contact is not None:
        selected_snapshot = _contact_match_snapshot(selected_contact, context)
        banner_tone = (
            "success"
            if selected_snapshot["fit_label"] == "Strong match"
            else ("warning" if selected_snapshot["fit_label"] == "Possible match" else "danger")
        )
        if selected_snapshot["yield_status"] == "Not assessed":
            banner_detail = (
                f"{selected_snapshot['match_count']}/4 criteria currently align. "
                f"{selected_snapshot['yield_reason']}"
            )
        elif selected_snapshot["fit_label"] == "Strong match":
            banner_detail = (
                f"{selected_snapshot['match_count']}/4 criteria align. "
                "This investor sits near the top of the current mandate shortlist."
            )
        elif selected_snapshot["fit_label"] == "Possible match":
            banner_detail = (
                f"{selected_snapshot['match_count']}/4 criteria align. "
                "This is a secondary target that still needs banker judgement."
            )
        else:
            banner_detail = (
                f"Only {selected_snapshot['match_count']}/4 criteria align. "
                "This investor should stay off the primary outreach list."
            )

        top_left, top_right = st.columns([1.55, 1.0])
        with top_left:
            with st.container(border=True):
                st.markdown(f"<div class='adi-subtitle'>{_escape(selected_investor_name)} contact profile</div>", unsafe_allow_html=True)
                st.markdown(
                    (
                        f"<div class='adi-note'>{_escape(str(selected_contact.get('title', '')).strip() or 'Alantra-held investor contact record.')}</div>"
                        "<div class='adi-inline-meta'>"
                        f"{_badge(_title_case_status(str(selected_contact.get('relationship_stage', 'new'))), 'mint')}"
                        f"{_badge(str(selected_contact.get('owner', 'Alantra')), 'neutral')}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(_table_html(_contact_profile_frame(selected_contact)), unsafe_allow_html=True)

        with top_right:
            with st.container(border=True):
                st.markdown("<div class='adi-subtitle'>Outreach history</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='adi-note'>Detailed touchpoint history is fully populated for SCBSM only in this prototype. Use the outreach log to test mandate-level interactions.</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(_nav_button("outreach-log", "+ Log interaction"), unsafe_allow_html=True)

        st.markdown("<div class='adi-divider'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f"<div class='adi-subtitle'>Match against: {_escape(context.current_deal.zone)} {_escape(context.current_deal.asset_type)} · EUR {context.current_deal.price_min_eur_mn:,.0f}-{context.current_deal.price_max_eur_mn:,.0f}m</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _banner_html(f"{selected_snapshot['fit_label']}.", banner_detail, tone=banner_tone),
                unsafe_allow_html=True,
            )
            st.markdown(_table_html(_generic_criteria_match_frame(selected_snapshot)), unsafe_allow_html=True)
            button_left, button_right = st.columns(2)
            with button_left:
                st.markdown(_nav_button("outreach-log", "Open outreach log"), unsafe_allow_html=True)
            with button_right:
                st.markdown(_nav_button("comparables", "View comps →"), unsafe_allow_html=True)
        return

    top_left, top_right = st.columns([1.55, 1.0])
    with top_left:
        with st.container(border=True):
            st.markdown("<div class='adi-subtitle'>SCBSM profile</div>", unsafe_allow_html=True)
            st.markdown(
                (
                    f"<div class='adi-note'>{_escape(profile['title'])}</div>"
                    "<div class='adi-inline-meta'>"
                    f"{_badge(profile['source_tag'], 'mint')}"
                    f"{_badge(profile['last_updated'], 'neutral')}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(_table_html(_profile_attribute_frame(profile)), unsafe_allow_html=True)

    with top_right:
        with st.container(border=True):
            st.markdown("<div class='adi-subtitle'>Outreach history</div>", unsafe_allow_html=True)
            small_history = _small_history_frame(history)
            if small_history.empty:
                st.markdown("<div class='adi-note'>No interactions logged yet for SCBSM.</div>", unsafe_allow_html=True)
            else:
                st.markdown(_table_html(small_history), unsafe_allow_html=True)
            st.markdown(_nav_button("outreach-log", "+ Log interaction"), unsafe_allow_html=True)

    st.markdown("<div class='adi-divider'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"<div class='adi-subtitle'>Match against: {_escape(context.current_deal.zone)} {_escape(context.current_deal.asset_type)} · EUR {context.current_deal.price_min_eur_mn:,.0f}-{context.current_deal.price_max_eur_mn:,.0f}m</div>",
            unsafe_allow_html=True,
        )
        core_hits = sum(
            1
            for key in ["sector", "geography", "ticket", "cap_rate"]
            if evaluation["criteria"][key]["status"] == "Yes"
        )
        st.markdown(
            _banner_html(
                "Strong match.",
                f"{core_hits}/4 core criteria met. The implied deal yield is accretive relative to the investor's current portfolio.",
                tone="success",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(_table_html(_criteria_match_frame(evaluation)), unsafe_allow_html=True)
        button_left, button_right = st.columns(2)
        with button_left:
            if st.button("Add to outreach"):
                log_touchpoint(
                    deal=context.current_deal,
                    event_date=date.today().isoformat(),
                    touchpoint_type="teaser_sent",
                    status_value="sent",
                    owner=profile["owner"],
                    notes="Added from investor validation screen.",
                )
                st.success("SCBSM added to the outreach log.")
                st.rerun()
        with button_right:
            st.markdown(_nav_button("comparables", "View comps →"), unsafe_allow_html=True)

    st.markdown("<div class='adi-divider'></div>", unsafe_allow_html=True)
    negative = _negative_example_evaluation(context)
    mismatch_rows = pd.DataFrame(
        [
            {
                "Criterion": negative["criteria"]["sector"]["label"].replace(" match", ""),
                "Assessment": _badge("Mismatch", "danger"),
                "Explanation": negative["criteria"]["sector"]["reason"],
            },
            {
                "Criterion": negative["criteria"]["geography"]["label"].replace(" match", ""),
                "Assessment": _badge("Mismatch", "danger"),
                "Explanation": negative["criteria"]["geography"]["reason"],
            },
            {
                "Criterion": "Ticket",
                "Assessment": _badge("Mismatch", "danger"),
                "Explanation": negative["criteria"]["ticket"]["reason"],
            },
        ]
    )

    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>What if: Normandy hotel at EUR 8m</div>", unsafe_allow_html=True)
        st.markdown(
            _banner_html(
                "Do not include this investor.",
                "Multiple hard mismatches would have triggered a targeting warning before any teaser was sent.",
                tone="danger",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(_table_html(mismatch_rows), unsafe_allow_html=True)
        st.markdown(
            "<div class='adi-note'>This mirrors a real off-target contact. The platform would have flagged the mismatch before outreach.</div>",
            unsafe_allow_html=True,
        )


def _render_comparables_screen(context) -> None:
    _section_intro(
        "Comparable Retrieval",
        "A transparent comp-retrieval screen that supports valuation judgment without claiming spurious point precision.",
    )

    query = _build_live_comparable_query(context)
    focused_result = retrieve_comparables(query)
    dataset_status = focused_result["dataset_status"]

    comp_mode = st.query_params.get("comps", "focused")
    if isinstance(comp_mode, list):
        comp_mode = comp_mode[0] if comp_mode else "focused"
    comp_mode = "wide" if str(comp_mode).strip().lower() == "wide" else "focused"

    if comp_mode == "wide":
        results_frame = _widened_reference_results(query)
        scope_note = "Widened European references from the available dataset for broader same-type market context."
    else:
        results_frame = focused_result["results"].copy()
        scope_note = "Focused retrieval inside the primary mandate scope using only currently available transactions."

    scenario = classify_comparable_scenario(
        has_size=query.size_sqm is not None,
        has_year=query.transaction_year is not None,
        cap_rate_pct=query.cap_rate_pct,
    )
    lower, upper = scenario["expected_mape_range_pct"]
    display_frame = _comparables_display_frame(results_frame)

    st.markdown(
        (
            "<div class='adi-grid adi-grid--two'>"
            f"{_card_html('Query', f'{query.asset_type} - {query.country} - {int(query.size_sqm):,} sqm - {query.transaction_year}', note=scope_note, accent=True)}"
            f"{_card_html('Scenario', scenario['label'], note=f'Expected MAPE range {lower:.0f}% - {upper:.0f}%')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    dataset_tone = "warning" if dataset_status["coverage_flag"] else "success"
    st.markdown(
        _banner_html(
            "Available dataset only",
            (
                f"{dataset_status['dataset_note']} Source file: {dataset_status['source_file']}. "
                f"Available rows: {dataset_status['available_rows']}. "
                f"Exact {query.asset_type} / {query.country} matches: {dataset_status['query_exact_country_rows']}. "
                f"{dataset_status['coverage_note']}"
            ),
            tone=dataset_tone,
        ),
        unsafe_allow_html=True,
    )

    benchmark = focused_result["benchmark"]
    median_text = (
        f"EUR {benchmark['median_price_per_sqm_eur']:,.0f} / sqm"
        if pd.notna(benchmark["median_price_per_sqm_eur"])
        else "N/A"
    )
    method_metadata = _load_method_metadata()
    warning_text = _valuation_warning_text(method_metadata)

    left_card, right_card = st.columns([1.0, 1.35])
    with left_card:
        st.markdown(
            _card_html(
                "Cell median benchmark",
                median_text,
                note=f"Sample size N = {benchmark['sample_size']}",
                accent=True,
            ),
            unsafe_allow_html=True,
        )
    with right_card:
        tone = "warning" if benchmark["thin_cell"] else "success"
        st.markdown(_banner_html("Benchmark guidance", warning_text, tone=tone), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>Top comparables</div>", unsafe_allow_html=True)
        st.markdown(_table_html(display_frame), unsafe_allow_html=True)
        action_left, action_right = st.columns(2)
        with action_left:
            st.download_button(
                "Export to Excel",
                data=_workbook_download_bytes(
                    evaluation={k: v for k, v in context.scbsm_evaluation.items() if k != "criteria"},
                    criteria_frame=_criteria_dataframe(context.scbsm_evaluation),
                    history_frame=_history_export_frame(get_scbsm_history(context.events)),
                    comparable_frame=display_frame,
                ),
                file_name="alantra_deal_intelligence_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with action_right:
            next_mode = "focused" if comp_mode == "wide" else "wide"
            label = "Use focused search" if comp_mode == "wide" else "Widen search"
            st.markdown(_nav_button("comparables", label, extra_query=f"comps={next_mode}"), unsafe_allow_html=True)

        if focused_result["widened"]:
            st.markdown(
                _banner_html(
                    "Automatic widening applied",
                    f"Exact {query.asset_type} / {query.country} matches were thin, so the retrieval widened to {focused_result['retrieval_scope']}.",
                    tone="warning",
                ),
                unsafe_allow_html=True,
            )


def _render_outreach_log_screen(context) -> None:
    _section_intro(
        "Outreach Log",
        "A lightweight interaction logger for repeated analyst use, backed by local SQLite history rather than an unstructured tracking file.",
    )

    history = get_scbsm_history(context.events)
    mandate_options, default_mandate_label = _available_mandate_options(context)
    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>Log new interaction</div>", unsafe_allow_html=True)
        st.markdown(
            _banner_html(
                "Follow-up reminder",
                (
                    f"In a later version, follow-up timing will be computed from investor behaviour and deal context. "
                    f"In this prototype, it is fixed at {FOLLOW_UP_DELAY_DAYS} calendar days after a teaser is sent."
                ),
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
        with st.form("outreach_log_form"):
            row1 = st.columns(2)
            with row1[0]:
                _field_label("Mandate")
                selected_mandate_label = st.selectbox(
                    "Mandate",
                    options=list(mandate_options.keys()),
                    index=list(mandate_options.keys()).index(default_mandate_label),
                    key="outreach_log_mandate",
                    label_visibility="collapsed",
                )
            with row1[1]:
                _field_label("Investor")
                investor_name = st.selectbox("Investor", options=["SCBSM"], label_visibility="collapsed")

            row2 = st.columns(2)
            with row2[0]:
                _field_label("Date")
                event_date = st.date_input("Date", value=date.today(), label_visibility="collapsed")
            with row2[1]:
                _field_label("Action")
                action_label = st.selectbox(
                    "Action",
                    options=[label for label, _, _ in TOUCHPOINT_ACTION_OPTIONS],
                    label_visibility="collapsed",
                )

            _field_label("Notes")
            notes = st.text_area("Notes", placeholder="Internal note for the deal team.", label_visibility="collapsed")
            submitted = st.form_submit_button("Save")

        if submitted:
            if investor_name != "SCBSM":
                st.error("This prototype only logs against SCBSM.")
            else:
                selected_deal = mandate_options[selected_mandate_label]
                action_payload = TOUCHPOINT_ACTION_LOOKUP[action_label]
                log_touchpoint(
                    deal=selected_deal,
                    event_date=event_date.isoformat(),
                    touchpoint_type=action_payload["touchpoint_type"],
                    status_value=action_payload["status_value"],
                    owner=context.scbsm_profile["owner"],
                    notes=notes.strip(),
                )
                st.success("Interaction saved to SQLite.")
                st.rerun()

    with st.container(border=True):
        st.markdown("<div class='adi-subtitle'>History</div>", unsafe_allow_html=True)
        selected_history_deal = mandate_options.get(st.session_state.get("outreach_log_mandate", default_mandate_label), context.current_deal)
        filtered_history = history.loc[history["mandate_name"].eq(selected_history_deal.mandate_name)].copy()
        history_frame, is_example = _outreach_history_frame(filtered_history, include_example_if_empty=True)
        if is_example:
            st.markdown(
                "<div class='adi-note'>No saved interactions exist yet. The row below illustrates the structure analysts will populate.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(_table_html(history_frame), unsafe_allow_html=True)
        st.markdown(
            "<div class='adi-note'>History is stored locally in SQLite and replaces the traditional contact tracking file.</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Alantra Deal Intelligence", layout="wide")
    bootstrap_outreach_environment()
    _inject_theme_css()
    _ensure_current_deal_state()
    _render_shell()

    context = load_dashboard_context(deal_input=_read_current_deal_state())
    active_screen = _current_screen()

    if active_screen == "dashboard":
        _render_dashboard_screen(context)
    elif active_screen == "new-mandate":
        _render_new_mandate_screen()
    elif active_screen == "investors":
        _render_investor_registry_screen(context)
    elif active_screen == "investor-validation":
        _render_investor_validation_screen(context)
    elif active_screen == "comparables":
        _render_comparables_screen(context)
    elif active_screen == "outreach-log":
        _render_outreach_log_screen(context)


if __name__ == "__main__":
    main()
