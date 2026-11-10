"""The real Yahoo, once a day (.github/workflows/live.yml).

Every other test is offline, so none of them can notice Yahoo changing under
us. This one builds YahooSource explicitly rather than going through
load_source, because tests/conftest.py points DCF_MARKET_SOURCE at the
recorded source. Run it with `pytest -m live`.
"""

from __future__ import annotations

import re

import pytest

from dcf.data import build_company_data
from dcf.market import YahooSource

_URL_QUERY = re.compile(r"(https?://[^\s?'\"]+)\?\S*")
_SECRET = re.compile(r"(crumb|cookie|set-cookie)(=|:\s*)[^\s;,&'\"]+", re.I)


# This repo is public, so CI logs are too: keep query strings (they carry the
# crumb) and cookie values out of anything printed. Should become
# dcf.data.describe_failure once that function is on main.
def _scrub(text: str) -> str:
    return _SECRET.sub(r"\1\2<redacted>", _URL_QUERY.sub(r"\1", text))


def _describe(e: Exception) -> str:
    return _scrub(f"{type(e).__name__}: {e}")


def _probe(tk) -> str:
    """Say which of the three yfinance attributes failed or came back empty.

    dcf/data.py swallows these errors, so an exception from the build alone
    often cannot tell you which endpoint went dark. yfinance caches each one
    on the Ticker, so this reads the same responses the build then uses.
    """
    lines = []
    for attr in ("info", "financials", "cashflow"):
        try:
            value = getattr(tk, attr)
            size = len(value) if hasattr(value, "__len__") else "?"
            lines.append(f"  .{attr}: {type(value).__name__}, {size} entries")
        except Exception as e:
            lines.append(f"  .{attr}: raised {_describe(e)}")
    return "\n".join(lines)


@pytest.mark.live
def test_yahoo_serves_company_data():
    tk = YahooSource().ticker("AAPL")
    probe = _probe(tk)
    try:
        data = build_company_data("AAPL", tk)
    except Exception as e:
        pytest.fail(f"build_company_data(AAPL) raised {_describe(e)}\n{probe}")

    for field in ("shares_outstanding", "revenue_base", "current_price"):
        value = getattr(data, field)
        assert value and value > 0, f"{field} is {value!r}\n{probe}"


@pytest.mark.live
def test_yahoo_serves_treasury_history():
    try:
        history = YahooSource().treasury_history()
    except Exception as e:
        pytest.fail(f"treasury_history() raised {_describe(e)}")
    assert not history.empty, "treasury_history() returned an empty frame"
