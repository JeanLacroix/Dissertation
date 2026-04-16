from __future__ import annotations

from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from .outreach_db import load_staged_mandates, stage_mandate


class StagedMandateIn(BaseModel):
    mandate_name: str = Field(min_length=1)
    asset_type: str = Field(min_length=1)
    country: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    city: str = ""
    ticket_eur_mn: float = Field(gt=0)
    cap_rate_pct: float = Field(gt=0)
    size_sqm: float = Field(gt=0)
    transaction_date: date
    source: str | None = None
    notes: str | None = None


app = FastAPI(title="SCBSM Mandate Intake API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mandates/staging")
def get_staged_mandates() -> list[dict[str, object]]:
    frame = load_staged_mandates()
    if frame.empty:
        return []
    return frame.fillna("").to_dict(orient="records")


@app.post("/mandates/staging")
def post_staged_mandate(payload: StagedMandateIn) -> dict[str, object]:
    staged_mandate_id = stage_mandate(
        mandate_name=payload.mandate_name,
        asset_type=payload.asset_type,
        country=payload.country,
        zone=payload.zone,
        city=payload.city,
        ticket_eur_mn=payload.ticket_eur_mn,
        cap_rate_pct=payload.cap_rate_pct,
        size_sqm=payload.size_sqm,
        transaction_date=payload.transaction_date.isoformat(),
        source=payload.source,
        notes=payload.notes,
    )
    return {
        "status": "staged",
        "staged_mandate_id": staged_mandate_id,
    }


def main() -> None:
    uvicorn.run("src.backend.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
