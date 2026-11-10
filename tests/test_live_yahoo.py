"""The real Yahoo, once a day (.github/workflows/live.yml).

Every other test is offline, so none of them can notice Yahoo changing under
us. This one builds YahooSource explicitly rather than going through
load_source, because tests/conftest.py points DCF_MARKET_SOURCE at the
recorded source. Run it with `pytest -m live`.
"""

from __future__ import annotations

import pytest

from dcf.market import YahooSource, fetch_company_data


def _probe(symbol: str) -> str:
    """Say which of the three yfinance attributes failed or came back empty.

    dcf/data.py swallows these errors, so an exception from fetch_company_data
    alone often cannot tell you which endpoint went dark.
    """
    tk = YahooSource().ticker(symbol)
    lines = []
    for attr in ("info", "financials", "cashflow"):
        try:
            value = getattr(tk, attr)
            size = len(value) if hasattr(value, "__len__") else "?"
            lines.append(f"  .{attr}: {type(value).__name__}, {size} entries")
        except Exception as e:
            lines.append(f"  .{attr}: raised {type(e).__name__}: {e}")
    return "\n".join(lines)


@pytest.mark.live
def test_yahoo_serves_company_data():
    source = YahooSource()
    try:
        data = fetch_company_data(source, "AAPL")
    except Exception as e:
        pytest.fail(
            f"fetch_company_data(AAPL) raised {type(e).__name__}: {e}\n"
            f"{_probe('AAPL')}"
        )

    for field in ("shares_outstanding", "revenue_base", "current_price"):
        value = getattr(data, field)
        assert value and value > 0, f"{field} is {value!r}\n{_probe('AAPL')}"


@pytest.mark.live
def test_yahoo_serves_treasury_history():
    history = YahooSource().treasury_history()
    assert not history.empty, "treasury_history() returned an empty frame"
