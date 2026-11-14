"""The real Yahoo, once a day (.github/workflows/live.yml).

Every other test is offline, so none of them can notice Yahoo changing under
us. This one builds YahooSource explicitly rather than going through
load_source, because tests/conftest.py points DCF_MARKET_SOURCE at the
recorded source. Run it with `pytest -m live`.
"""

from __future__ import annotations

import pytest

from dcf.data import build_company_data, describe_failure
from dcf.market import YahooSource


def _probe(tk) -> str:
    """Say which of the yfinance attributes failed or came back empty.

    dcf/data.py swallows these errors, so an exception from the build alone
    often cannot tell you which endpoint went dark. yfinance caches each one
    on the Ticker, so this reads the same responses the build then uses.
    """
    lines = []
    for attr in ("info", "financials", "cashflow", "balance_sheet"):
        try:
            value = getattr(tk, attr)
            size = len(value) if hasattr(value, "__len__") else "?"
            lines.append(f"  .{attr}: {type(value).__name__}, {size} entries")
        except Exception as e:
            lines.append(f"  .{attr}: raised {describe_failure(e)}")
    return "\n".join(lines)


@pytest.mark.live
def test_yahoo_serves_company_data():
    tk = YahooSource().ticker("AAPL")
    probe = _probe(tk)
    try:
        data = build_company_data("AAPL", tk)
    except Exception as e:
        pytest.fail(f"build_company_data(AAPL) raised {describe_failure(e)}\n{probe}")

    # The app survives a dead `.info` by reading shares, debt and cash from
    # the balance sheet. That keeps users served, but this test exists to say
    # Yahoo is refusing `.info`, so a fallback is a failure here.
    assert data.quote_fallback is None, (
        f"Yahoo's company info did not answer ({data.quote_fallback}); the "
        f"app is running on its balance-sheet fallback\n{probe}"
    )

    for field in ("shares_outstanding", "revenue_base", "current_price"):
        value = getattr(data, field)
        assert value and value > 0, f"{field} is {value!r}\n{probe}"


@pytest.mark.live
def test_yahoo_serves_treasury_history():
    try:
        history = YahooSource().treasury_history()
    except Exception as e:
        pytest.fail(f"treasury_history() raised {describe_failure(e)}")
    assert not history.empty, "treasury_history() returned an empty frame"
