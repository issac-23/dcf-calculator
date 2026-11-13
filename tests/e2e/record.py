"""Capture live Yahoo Finance responses into `fixtures/` for the E2E suite.

This is the only code in the repo that is *meant* to touch the network, and
pytest never runs it. Run it by hand when a fixture should be refreshed:

    python tests/e2e/record.py            # re-record every ticker in ROSTER
    python tests/e2e/record.py JNJ AAPL   # just these
    python tests/e2e/record.py --add balance_sheet,history AAPL
                                          # add only those keys to an existing
                                          # recording, leaving the rest as is

Refreshing will change the committed snapshot report, because prices and
filings move. That is the intended workflow: re-record, run the suite, read
the snapshot diff, and decide whether the new numbers are right before
committing them. A silent diff is the thing to be suspicious of — it means
the recording changed but the model's conclusions did not.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.e2e.replay import FIXTURES, frame_to_json, history_to_json  # noqa: E402

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


# Keys `--add` can fill in on an existing recording.
ADDABLE = ("balance_sheet", "history")


def _frame(tk, attr):
    try:
        return frame_to_json(getattr(tk, attr))
    except Exception:
        return frame_to_json(None)


def _history(tk, recorded_at):
    """Closes up to the recording date, so the price agrees with the rest of
    the recording; ten days of lead-in cover weekends and holidays. Without a
    date it is simply the last few days."""
    try:
        if recorded_at:
            end = datetime.strptime(recorded_at, "%Y-%m-%d") + timedelta(days=1)
            start = end - timedelta(days=10)
            return history_to_json(
                tk.history(
                    start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
                )
            )
        return history_to_json(tk.history(period="5d"))
    except Exception:
        return history_to_json(None)


def add_keys(symbol: str, keys) -> Path:
    """Add `keys` to an existing fixture and leave everything else as it was
    recorded, so the snapshot numbers do not move."""
    import yfinance as yf

    path = FIXTURES / f"{symbol.upper()}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    tk = yf.Ticker(symbol)
    if "balance_sheet" in keys:
        payload["balance_sheet"] = _frame(tk, "balance_sheet")
    if "history" in keys:
        payload["history"] = _history(tk, payload["recorded_at"])
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


def record(symbol: str) -> Path:
    import yfinance as yf

    tk = yf.Ticker(symbol)

    try:
        raw_info = tk.info or {}
    except Exception:
        raw_info = {}
    info = {k: v for k, v in raw_info.items() if isinstance(v, _SCALARS)}

    def frame(attr):
        return _frame(tk, attr)

    payload = {
        "symbol": symbol.upper(),
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "note": ROSTER.get(symbol.upper(), ""),
        "info": info,
        "financials": frame("financials"),
        "cashflow": frame("cashflow"),
        "balance_sheet": frame("balance_sheet"),
        "history": _history(tk, None),
    }

    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{symbol.upper()}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


def main(argv):
    if len(argv) > 1 and argv[1] == "--add":
        keys = argv[2].split(",") if len(argv) > 2 else []
        bad = [k for k in keys if k not in ADDABLE]
        if bad or not keys or len(argv) < 4:
            sys.exit("usage: record.py --add balance_sheet,history SYMBOL [SYMBOL...]")
        for s in (a.upper() for a in argv[3:]):
            path = add_keys(s, keys)
            print(f"  {s:6} -> {path.name}  (added {', '.join(keys)})")
        print("\nAdded. The rest of each recording is untouched.")
        return
    symbols = [s.upper() for s in argv[1:]] or list(ROSTER)
    for s in symbols:
        path = record(s)
        size = path.stat().st_size / 1024
        print(f"  {s:6} -> {path.name}  ({size:.0f} KB)")
    print(f"\nRecorded {len(symbols)}. Now run: pytest tests/e2e")


if __name__ == "__main__":
    main(sys.argv)
