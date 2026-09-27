"""Capture live Yahoo Finance responses into `fixtures/` for the E2E suite.

This is the only code in the repo that is *meant* to touch the network, and
pytest never runs it. Run it by hand when a fixture should be refreshed:

    python tests/e2e/record.py            # re-record every ticker in ROSTER
    python tests/e2e/record.py JNJ AAPL   # just these

Refreshing will change the committed snapshot report, because prices and
filings move. That is the intended workflow: re-record, run the suite, read
the snapshot diff, and decide whether the new numbers are right before
committing them. A silent diff is the thing to be suspicious of — it means
the recording changed but the model's conclusions did not.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.e2e.replay import FIXTURES, frame_to_json  # noqa: E402

# Each ticker earns its place by covering a distinct path through the model.
ROSTER = {
    "JNJ": "low beta, mature, model lands near market",
    "AAPL": "high beta, flat historical growth, large negative gap",
    "TSLA": "reverse DCF finds no growth rate that reaches the price",
    "JPM": "financial sector, flagged as the wrong tool",
    "KRTX": "acquired and delisted, no financials at all",
}

# yfinance's `info` carries nested junk and occasional non-serializable values.
# Keep the flat scalars; that is everything dcf/data.py reads.
_SCALARS = (str, int, float, bool, type(None))


def record(symbol: str) -> Path:
    import yfinance as yf

    tk = yf.Ticker(symbol)

    try:
        raw_info = tk.info or {}
    except Exception:
        raw_info = {}
    info = {k: v for k, v in raw_info.items() if isinstance(v, _SCALARS)}

    def frame(attr):
        try:
            return frame_to_json(getattr(tk, attr))
        except Exception:
            return frame_to_json(None)

    payload = {
        "symbol": symbol.upper(),
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "note": ROSTER.get(symbol.upper(), ""),
        "info": info,
        "financials": frame("financials"),
        "cashflow": frame("cashflow"),
    }

    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{symbol.upper()}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


def main(argv):
    symbols = [s.upper() for s in argv[1:]] or list(ROSTER)
    for s in symbols:
        path = record(s)
        size = path.stat().st_size / 1024
        print(f"  {s:6} -> {path.name}  ({size:.0f} KB)")
    print(f"\nRecorded {len(symbols)}. Now run: pytest tests/e2e")


if __name__ == "__main__":
    main(sys.argv)
