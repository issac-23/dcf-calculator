"""Replay recorded Yahoo Finance responses so the E2E suite is offline.

The seam is `dcf.data.build_company_data(symbol, tk)`, which accepts anything
with `.info`, `.financials` and `.cashflow`. `RecordedTicker` is that, backed
by a JSON file in `fixtures/` captured from the real API by `record.py`.

Recording rather than hand-writing the fixtures is the point. A hand-written
dict encodes what we *believe* Yahoo returns; a recording encodes what it
actually returned, including the row labels that move around between tickers
("Interest Expense" vs "Interest Expense Non Operating") and the fields that
are simply absent for some companies. Those are the shapes that broke the
data layer in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def frame_to_json(df: pd.DataFrame | None) -> Dict[str, List[Any]]:
    if df is None or df.empty:
        return {"index": [], "columns": [], "data": []}
    return {
        "index": [str(i) for i in df.index],
        "columns": [
            c.isoformat() if hasattr(c, "isoformat") else str(c) for c in df.columns
        ],
        "data": [
            [None if pd.isna(v) else float(v) for v in row] for row in df.to_numpy()
        ],
    }


def frame_from_json(d: Dict[str, List[Any]]) -> pd.DataFrame:
    if not d.get("columns"):
        return pd.DataFrame()
    return pd.DataFrame(
        d["data"],
        index=d["index"],
        columns=[pd.Timestamp(c) for c in d["columns"]],
    )


class RecordedTicker:
    """Satisfies `dcf.data._TickerLike` from a recorded fixture."""

    def __init__(self, payload: Dict[str, Any]):
        self.symbol: str = payload["symbol"]
        self.recorded_at: str = payload["recorded_at"]
        self.info: Dict[str, Any] = payload["info"]
        self.financials: pd.DataFrame = frame_from_json(payload["financials"])
        self.cashflow: pd.DataFrame = frame_from_json(payload["cashflow"])


def load(symbol: str) -> RecordedTicker:
    path = FIXTURES / f"{symbol.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No recording for {symbol}. Run: python tests/e2e/record.py {symbol}"
        )
    return RecordedTicker(json.loads(path.read_text(encoding="utf-8")))


def available() -> List[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))
