from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "scbsm"
LISTINGS_DIR = OUTPUT_ROOT / "listings"
DOCUMENTS_DIR = OUTPUT_ROOT / "documents"
UNPACKED_DIR = OUTPUT_ROOT / "unpacked"
TABLES_DIR = OUTPUT_ROOT / "tables"

LISTING_URL_TEMPLATE = "https://www.scbsm.fr/documents/rapports/?ID=SCBSM&annee={year}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
)
DEFAULT_YEARS = [2026, 2025, 2024, 2023, 2022, 2021]

DATE_PATTERN = re.compile(r"\b(\d{2}/\d{2}/\d{4})(?:\s+\d{2}:\d{2})?\b")
FILE_LINK_LABELS = {"pdf", "zip", "xhtml"}
ANNUAL_TITLE_TOKENS = (
    "document d'enregistrement universel",
    "document d’enregistrement universel",
    "rapport financier annuel",
    "urd",
)
TABLE_KEYWORDS = (
    "surface",
    "valeur",
    "expert",
    "immeuble",
    "bureaux",
    "bureau",
    "loyer",
    "vacance",
    "rendement",
    "actif",
    "qca",
    "paris",
)
OFFICE_KEYWORDS = (
    "bureaux",
    "bureau",
    "paris",
    "qca",
    "surface",
    "loyer",
    "valeur",
)
ASSET_TABLE_GROUPS = (
    ("surface",),
    ("valeur", "loyer", "rendement", "vacance", "vlm"),
    ("adresse", "immeuble", "actif"),
)


@dataclass(frozen=True)
class DocumentSelection:
    listing_year: int
    report_title: str
    section: str
    published_at: str
    link_kind: str
    url: str
    local_path: str
    source_page: str


def _normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return token[:100] or "document"


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _extract_previous_text(anchor: Tag) -> Iterable[str]:
    for element in anchor.previous_elements:
        if isinstance(element, NavigableString):
            text = _normalise_space(str(element))
            if text:
                yield text
        elif isinstance(element, Tag) and element.name in {"h1", "h2", "h3"}:
            break


def _find_previous_date(anchor: Tag) -> str:
    for text in _extract_previous_text(anchor):
        match = DATE_PATTERN.search(text)
        if match:
            return match.group(1)
    return ""


def _find_previous_heading(anchor: Tag) -> str:
    heading = anchor.find_previous(["h1", "h2", "h3"])
    if heading is None:
        return ""
    return _normalise_space(heading.get_text(" ", strip=True))


def _infer_link_kind(link_text: str, url: str) -> str:
    text = link_text.strip().lower()
    if text in FILE_LINK_LABELS:
        return text
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".pdf", ".zip"}:
        return suffix.lstrip(".")
    if suffix in {".html", ".xhtml"}:
        return "xhtml"
    return "title"


def scrape_listing(year: int) -> pd.DataFrame:
    source_page = LISTING_URL_TEMPLATE.format(year=year)
    html = _fetch_bytes(source_page)
    _write_bytes(LISTINGS_DIR / f"reports_{year}.html", html)

    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, str | int]] = []
    current_title = ""

    for anchor in soup.find_all("a", href=True):
        href = urljoin(source_page, anchor["href"])
        if "actusnews.com" not in href.lower():
            continue

        link_text = _normalise_space(anchor.get_text(" ", strip=True))
        if not link_text:
            continue

        link_kind = _infer_link_kind(link_text, href)
        if link_kind == "title":
            current_title = link_text

        report_title = current_title or link_text
        rows.append(
            {
                "listing_year": year,
                "source_page": source_page,
                "section": _find_previous_heading(anchor),
                "published_at": _find_previous_date(anchor),
                "report_title": report_title,
                "link_text": link_text,
                "link_kind": link_kind,
                "url": href,
            }
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["listing_year", "report_title", "link_kind", "url"])
    if frame.empty:
        return frame

    file_label_mask = frame["report_title"].str.lower().isin(FILE_LINK_LABELS)
    title_candidates = frame.loc[~frame["link_text"].str.lower().isin(FILE_LINK_LABELS), "link_text"]
    frame["resolved_title"] = frame["report_title"]

    for index, row in frame.loc[file_label_mask].iterrows():
        same_block = frame.loc[
            frame["section"].eq(row["section"])
            & frame["published_at"].eq(row["published_at"])
            & ~frame["link_text"].str.lower().isin(FILE_LINK_LABELS)
        ].copy()
        if same_block.empty:
            if not title_candidates.empty:
                frame.at[index, "resolved_title"] = str(title_candidates.iloc[0])
            continue

        same_block["distance"] = (same_block.index.to_series() - index).abs()
        nearest_title = same_block.sort_values("distance").iloc[0]["link_text"]
        frame.at[index, "resolved_title"] = str(nearest_title)

    frame["report_title"] = frame["resolved_title"]
    frame = frame.drop(columns=["resolved_title"])
    return frame.sort_values(["listing_year", "report_title", "link_kind"]).reset_index(drop=True)


def select_annual_documents(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()

    report_title_norm = inventory["report_title"].str.lower()
    section_norm = inventory["section"].fillna("").str.lower()
    annual_mask = report_title_norm.str.contains("|".join(map(re.escape, ANNUAL_TITLE_TOKENS))) | section_norm.str.contains(
        "rapport financier annuel"
    )
    link_mask = inventory["link_kind"].isin(["xhtml", "zip", "pdf"])

    candidates = inventory.loc[annual_mask & link_mask].copy()
    priority_map = {"xhtml": 0, "zip": 1, "pdf": 2}
    candidates["priority"] = candidates["link_kind"].map(priority_map).fillna(99)
    candidates = candidates.sort_values(["listing_year", "report_title", "priority", "published_at", "url"])
    selected = candidates.drop_duplicates(subset=["listing_year", "report_title"], keep="first").reset_index(drop=True)
    return selected


def _derive_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if name:
        return name
    return f"{int(time.time())}.bin"


def download_documents(selected: pd.DataFrame, sleep_seconds: float) -> list[DocumentSelection]:
    downloads: list[DocumentSelection] = []
    for row in selected.itertuples(index=False):
        filename = _derive_filename(row.url)
        local_dir = DOCUMENTS_DIR / str(row.listing_year)
        local_path = local_dir / filename
        if not local_path.exists():
            payload = _fetch_bytes(row.url)
            _write_bytes(local_path, payload)
            time.sleep(sleep_seconds)

        downloads.append(
            DocumentSelection(
                listing_year=int(row.listing_year),
                report_title=str(row.report_title),
                section=str(row.section),
                published_at=str(row.published_at),
                link_kind=str(row.link_kind),
                url=str(row.url),
                local_path=str(local_path.relative_to(PROJECT_ROOT)),
                source_page=str(row.source_page),
            )
        )
    return downloads


def _unpack_document(path: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix in {".html", ".xhtml"}:
        return [path]

    if suffix != ".zip":
        return []

    output_dir = UNPACKED_DIR / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    html_paths: list[Path] = []
    with ZipFile(path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            target = output_dir / Path(member).name
            target.write_bytes(archive.read(member))
            if target.suffix.lower() in {".html", ".xhtml", ".xml"}:
                html_paths.append(target)
    return html_paths


def _table_to_frame(table: Tag) -> pd.DataFrame:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_normalise_space(" ".join(cell.stripped_strings)) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    columns = [f"col_{idx}" for idx in range(1, width + 1)]
    return pd.DataFrame(padded, columns=columns)


def _matching_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]


def _has_asset_table_signal(text: str) -> bool:
    lowered = text.lower()
    has_surface = any(token in lowered for token in ASSET_TABLE_GROUPS[0])
    has_metric = any(token in lowered for token in ASSET_TABLE_GROUPS[1])
    has_asset_identifier = any(token in lowered for token in ASSET_TABLE_GROUPS[2])
    return (has_surface and has_metric) or (has_surface and has_asset_identifier)


def extract_candidate_tables(downloads: list[DocumentSelection]) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict[str, str | int | bool]] = []
    office_row_frames: list[pd.DataFrame] = []

    for document in downloads:
        local_path = PROJECT_ROOT / document.local_path
        parse_targets = _unpack_document(local_path)
        if not parse_targets and local_path.suffix.lower() in {".html", ".xhtml"}:
            parse_targets = [local_path]

        for parse_target in parse_targets:
            payload = parse_target.read_bytes()
            looks_like_xml = payload.lstrip().startswith(b"<?xml")
            parser = "xml" if looks_like_xml or parse_target.suffix.lower() in {".xml", ".xhtml"} else "lxml"
            soup = BeautifulSoup(payload, parser)
            for table_index, table in enumerate(soup.find_all("table"), start=1):
                frame = _table_to_frame(table)
                if frame.empty:
                    continue

                flat_text = _normalise_space(" ".join(frame.astype(str).fillna("").agg(" ".join, axis=1).tolist()))
                matched_keywords = _matching_keywords(flat_text, TABLE_KEYWORDS)
                office_signal = bool(_matching_keywords(flat_text, OFFICE_KEYWORDS))
                asset_table_signal = _has_asset_table_signal(flat_text)
                if not matched_keywords and not office_signal and not asset_table_signal:
                    continue

                table_dir = TABLES_DIR / parse_target.stem
                table_dir.mkdir(parents=True, exist_ok=True)
                table_path = table_dir / f"table_{table_index:03d}.csv"
                frame.to_csv(table_path, index=False)

                header_preview = " | ".join([value for value in frame.iloc[0].tolist() if value][:6])
                catalog_rows.append(
                    {
                        "listing_year": document.listing_year,
                        "report_title": document.report_title,
                        "published_at": document.published_at,
                        "link_kind": document.link_kind,
                        "source_url": document.url,
                        "document_path": document.local_path,
                        "parsed_file": str(parse_target.relative_to(PROJECT_ROOT)),
                        "table_index": table_index,
                        "rows": int(len(frame)),
                        "columns": int(frame.shape[1]),
                        "matched_keywords": ", ".join(matched_keywords),
                        "office_signal": office_signal,
                        "asset_table_signal": asset_table_signal,
                        "header_preview": header_preview,
                        "table_csv_path": str(table_path.relative_to(PROJECT_ROOT)),
                    }
                )

                if asset_table_signal:
                    office_rows = frame.copy()
                    office_rows.insert(0, "row_number", range(1, len(office_rows) + 1))
                    office_rows.insert(0, "table_index", table_index)
                    office_rows.insert(0, "parsed_file", str(parse_target.relative_to(PROJECT_ROOT)))
                    office_rows.insert(0, "report_title", document.report_title)
                    office_rows.insert(0, "listing_year", document.listing_year)
                    office_row_frames.append(office_rows)

    catalog = pd.DataFrame(catalog_rows)
    if not catalog.empty:
        catalog = catalog.sort_values(
            ["listing_year", "report_title", "table_index"],
            ignore_index=True,
        )
    office_rows = pd.concat(office_row_frames, ignore_index=True) if office_row_frames else pd.DataFrame()
    return catalog, office_rows


def run(years: list[int], sleep_seconds: float) -> dict[str, int]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    inventories = [scrape_listing(year) for year in years]
    inventory = pd.concat(inventories, ignore_index=True) if inventories else pd.DataFrame()
    inventory.to_csv(OUTPUT_ROOT / "report_inventory.csv", index=False)

    selected = select_annual_documents(inventory)
    selected.to_csv(OUTPUT_ROOT / "selected_annual_documents.csv", index=False)

    downloads = download_documents(selected, sleep_seconds=sleep_seconds)
    downloads_frame = pd.DataFrame(asdict(item) for item in downloads)
    downloads_frame.to_csv(OUTPUT_ROOT / "download_log.csv", index=False)

    table_catalog, office_rows = extract_candidate_tables(downloads)
    table_catalog.to_csv(OUTPUT_ROOT / "table_catalog.csv", index=False)
    if not table_catalog.empty and "office_signal" in table_catalog.columns:
        table_catalog.loc[table_catalog["office_signal"]].to_csv(OUTPUT_ROOT / "office_candidate_tables.csv", index=False)
    if not table_catalog.empty and "asset_table_signal" in table_catalog.columns:
        table_catalog.loc[table_catalog["asset_table_signal"]].to_csv(
            OUTPUT_ROOT / "office_asset_candidate_tables.csv",
            index=False,
        )
    if not office_rows.empty:
        office_rows.to_csv(OUTPUT_ROOT / "office_asset_candidate_rows.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "listing_years": years,
        "inventory_rows": int(len(inventory)),
        "selected_documents": int(len(selected)),
        "downloaded_documents": int(len(downloads)),
        "candidate_tables": int(len(table_catalog)),
        "office_asset_candidate_row_count": int(len(office_rows)),
        "output_root": str(OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
        "notes": [
            "Selection is limited to annual reports / URD-style filings on SCBSM's reports pages.",
            "Extraction targets HTML/XHTML content first; PDF-only filings are downloaded but not parsed in this proof of concept.",
            "Table relevance is keyword-based and should be reviewed manually before modelling.",
        ],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "inventory_rows": int(len(inventory)),
        "selected_documents": int(len(selected)),
        "downloaded_documents": int(len(downloads)),
        "candidate_tables": int(len(table_catalog)),
        "office_asset_candidate_rows": int(len(office_rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape SCBSM annual report links and extract candidate valuation tables for Office-France analysis."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Listing years to scrape from SCBSM reports pages.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between document downloads to keep the proof of concept polite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(years=args.years, sleep_seconds=args.sleep_seconds)
    print("SCBSM scraping proof of concept finished.")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
