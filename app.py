from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
COMPS_SAMPLE_PATH = ARTIFACTS_DIR / "comps_sample.parquet"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEFAULT_QUERY_SIZE_SQM = 10_000.0
TOP_K = 10


@st.cache_data(show_spinner=False)
def load_app_artifacts() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
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


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
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


def _build_download_workbook(display_frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        display_frame.to_excel(writer, index=False, sheet_name="Retrieved Comps")
    output.seek(0)
    return output.getvalue()


def _render_reference_benchmark(
    benchmark_frame: pd.DataFrame,
    asset_type: str,
    country: str,
    metadata: dict[str, Any],
) -> None:
    st.subheader("Context panel")
    st.caption("Reference benchmark only. This is not a prediction.")

    cell = benchmark_frame.loc[
        benchmark_frame["asset_type"].eq(asset_type) & benchmark_frame["country"].eq(country)
    ].copy()

    if cell.empty:
        st.warning("No exact asset-type by country cell exists in the deployed sample for this query.")
        return

    row = cell.iloc[0]
    left, right = st.columns(2)
    left.metric("Median price per sqm", f"EUR {row['median_price_per_sqm_eur']:,.0f}")
    right.metric("Cell sample size", f"{int(row['sample_size'])}")
    st.caption(metadata["reference_benchmark"]["definition"])


def _render_methodology_note(metadata: dict[str, Any]) -> None:
    evaluation = metadata["valuation_evaluation"]
    rolling = evaluation["rolling_origin"]
    random_5_fold = evaluation["random_5_fold"]

    with st.expander("Methodology note"):
        st.write(metadata["methodology_note"])
        fold_text = ", ".join(
            f"{fold['test_year']}: {fold['model_mape_pct']:.1f}%"
            for fold in rolling["fold_mapes_pct"]
        )
        st.write(f"Rolling-origin fold MAPEs: {fold_text}.")
        st.write(
            "Final evaluated specification: "
            f"`{metadata['model_formula_evaluated']}`"
        )
        st.write(
            "Random 5-fold mean MAPE: "
            f"{random_5_fold['mean_mape_pct']:.1f}%."
        )
        st.write(
            "2026 headline fold MAPE versus naive benchmark: "
            f"{rolling['headline_fold_mape_pct']:.1f}% vs {rolling['headline_fold_baseline_mape_pct']:.1f}%."
        )


def main() -> None:
    st.set_page_config(page_title="Comparable Retrieval Tool", layout="wide")
    st.title("French Commercial Real Estate Comparable Retrieval Tool")
    st.caption(
        "This deployed artifact retrieves anonymised comparable transactions. "
        "It does not output a point valuation."
    )

    try:
        comps, metadata, benchmark_frame = load_app_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    weights = metadata["retrieval_scoring"]["weights"]
    asset_type_options = metadata["asset_type_levels"]
    country_options = metadata["country_group_levels"]

    with st.sidebar:
        st.header("Query")
        asset_type = st.selectbox("Primary asset type", options=asset_type_options, index=0)
        country = st.selectbox("Country", options=country_options, index=0)
        size_sqm = st.number_input("Total size (sqm)", min_value=1.0, value=DEFAULT_QUERY_SIZE_SQM, step=500.0)
        transaction_date = st.date_input("Transaction date", value=date.today())

    ranked = _score_comparables(
        comps=comps,
        asset_type=asset_type,
        country=country,
        query_year=transaction_date.year,
        query_size_sqm=float(size_sqm),
        weights=weights,
    )
    top_comps = ranked.head(TOP_K)
    display_frame = _display_table(top_comps)

    st.subheader("Retrieved comparables")
    st.dataframe(display_frame, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download the 10 retrieved comps",
        data=_build_download_workbook(display_frame),
        file_name="retrieved_comparables.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    _render_reference_benchmark(
        benchmark_frame=benchmark_frame,
        asset_type=asset_type,
        country=country,
        metadata=metadata,
    )
    _render_methodology_note(metadata)


if __name__ == "__main__":
    main()
