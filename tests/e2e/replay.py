"""Replay recorded Yahoo Finance responses so the E2E suite is offline.

The seam is `dcf.market.MarketDataSource`. `RecordedSource` is its recorded
adapter: `ticker()` hands back a `RecordedTicker` (anything with `.info`,
`.financials` and `.cashflow`) backed by a JSON file in `fixtures/` captured
from the real API by `record.py`, and `treasury_history()` returns a pinned
10-year Treasury quote. The app picks it by setting DCF_MARKET_SOURCE to
`tests.e2e.replay:recorded_source`; see tests/conftest.py.

Recording rather than hand-writing the fixtures is the point. A hand-written
dict encodes what we *believe* Yahoo returns; a recording encodes what it
actually returned, including the row labels that move around between tickers
("Interest Expense" vs "Interest Expense Non Operating") and the fields that
are simply absent for some companies. Those are the shapes that broke the
data layer in the first place.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    """Satisfies `dcf.data.Fundamentals` from a recorded fixture."""

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


# Pinned, not fetched. The live 10-year Treasury moves daily and would make
# the snapshot churn for reasons that have nothing to do with our code. This
# is ^TNX's close, in percent, on the day the fixtures were recorded (5.18%).
# Re-recording it needs the live API, so it is not part of record.py.
PINNED_TREASURY_CLOSE = (5.18,)

# A journey that wants a different quote sets this to a comma-separated list
# of closes ("4.10,4.96", "nan,nan"), "empty" for no quotes, or "error" for a
# dead network. The quote then goes through the real parsing and fallback.
TREASURY_ENV_VAR = "E2E_TREASURY_CLOSE"


class RecordedSource:
    """The recorded `dcf.market.MarketDataSource`: no network, ever."""

    def __init__(self, treasury_close: Sequence[float] | Exception = PINNED_TREASURY_CLOSE):
        self._treasury_close = treasury_close

    def ticker(self, symbol: str) -> RecordedTicker:
        return load(symbol)

    def treasury_history(self) -> pd.DataFrame:
        if isinstance(self._treasury_close, Exception):
            raise self._treasury_close
        return pd.DataFrame({"Close": list(self._treasury_close)})


def recorded_source() -> RecordedSource:
    """Factory for DCF_MARKET_SOURCE: pinned Treasury unless the env var says otherwise."""
    raw = os.environ.get(TREASURY_ENV_VAR, "").strip()
    if not raw:
        return RecordedSource()
    if raw == "error":
        return RecordedSource(ConnectionError("Yahoo unreachable"))
    if raw == "empty":
        return RecordedSource([])
    return RecordedSource([float(x) for x in raw.split(",")])
