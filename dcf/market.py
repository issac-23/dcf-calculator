"""Where market data comes from: one interface, a live adapter, and a loader.

A `MarketDataSource` hands over the two raw things the model needs from the
outside world -- a ticker's fundamentals and the 10-year Treasury's price
history. Everything built on top of them (turning fundamentals into a
`CompanyData`, turning a quote into a risk-free rate, the fallback when the
quote is dead) is shared code in `fetch_company_data` and `fetch_risk_free_rate`
below, so every source gets the same parsing and the same failure handling.

There are two adapters. `YahooSource` is live yfinance. The recorded one lives
with the E2E suite (tests/e2e/replay.py) and replays captured responses.
The app chooses at startup through `load_source`: with DCF_MARKET_SOURCE unset
it is live; set to `package.module:factory` it calls that factory and uses what
it returns. That is the seam tests use instead of patching attributes.

Nothing here runs at import time, so importing it never touches the network.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Protocol

import pandas as pd

from dcf.data import (
    CompanyData,
    DataFetchError,
    Fundamentals,
    FALLBACK_RISK_FREE_RATE,
    build_company_data,
    describe_failure,
    risk_free_from_history,
)

SOURCE_ENV_VAR = "DCF_MARKET_SOURCE"

# A dead Treasury quote is not cached (the fallback must not outlive the
# outage), but refetching it on every rerun makes every slider drag wait on a
# failing request. So the failure is remembered briefly, per source. It lives
# here, not in app.py, because Streamlit re-executes app.py on every rerun and
# a dict there would start empty each time; a module imported once is shared by
# every session in the server process.
_RETRY_DEAD_QUOTE_AFTER_S = 60
_dead_quote_since: dict[str, float] = {}
_clock = time.monotonic


def quote_is_known_dead(source: str) -> bool:
    since = _dead_quote_since.get(source)
    return since is not None and _clock() - since < _RETRY_DEAD_QUOTE_AFTER_S


def remember_dead_quote(source: str) -> None:
    _dead_quote_since[source] = _clock()


def forget_dead_quote(source: str) -> None:
    _dead_quote_since.pop(source, None)

# The 10-year Treasury is the conventional risk-free proxy for a US equity
# DCF: long enough to match the horizon of a perpetuity, liquid enough to
# trust the quote.
_RISK_FREE_TICKER = "^TNX"


class MarketDataSource(Protocol):
    def ticker(self, symbol: str) -> Fundamentals:
        """Fundamentals for one symbol (info, financials, cashflow, and the
        balance sheet and price history used when `.info` does not answer)."""

    def treasury_history(self) -> pd.DataFrame:
        """Recent 10-year Treasury quotes: a frame with a "Close" column,
        in percent. May raise; the caller falls back."""


class YahooSource:
    """The live adapter."""

    def ticker(self, symbol: str) -> Fundamentals:
        import yfinance as yf  # imported lazily so tests can run without it

        return yf.Ticker(symbol)

    def treasury_history(self) -> pd.DataFrame:
        import yfinance as yf

        return yf.Ticker(_RISK_FREE_TICKER).history(period="5d")


def source_spec() -> str:
    """Which source is selected, as a string safe to use as a cache key."""
    return os.environ.get(SOURCE_ENV_VAR, "").strip()


def load_source() -> MarketDataSource:
    """The source this process uses: live unless DCF_MARKET_SOURCE says otherwise."""
    spec = source_spec()
    if not spec:
        return YahooSource()
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(
            f"{SOURCE_ENV_VAR} must look like 'package.module:factory', got {spec!r}"
        )
    return getattr(importlib.import_module(module_name), attr)()


def fetch_company_data(source: MarketDataSource, ticker: str) -> CompanyData:
    symbol = ticker.strip().upper()
    if not symbol:
        raise DataFetchError("Ticker symbol is empty")

    try:
        tk = source.ticker(symbol)
    except Exception as e:
        raise DataFetchError(
            f"Could not initialize ticker '{symbol}': {describe_failure(e)}"
        ) from e

    return build_company_data(symbol, tk)


def fetch_risk_free_rate(source: MarketDataSource) -> tuple[float, bool]:
    """Return (rate, is_live).

    Deliberately separate from fetch_company_data: this is one shared market
    rate rather than a property of the company, and it should be cached and
    fail on its own. A dead Treasury quote must not take a valuation with it.
    """
    try:
        return risk_free_from_history(source.treasury_history())
    except Exception:
        return FALLBACK_RISK_FREE_RATE, False
