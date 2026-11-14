"""Turn Yahoo Finance fundamentals into a `CompanyData`, with the historical
averages we use to seed the DCF inputs.

This module is pure: `build_company_data` turns anything shaped like a
yfinance Ticker into a `CompanyData`, and `risk_free_from_history` turns a
Treasury price history into a rate. Getting those objects from the network
(or a recording) is dcf/market.py's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd

# Sectors where a free-cash-flow DCF is the wrong tool.
_DCF_INAPPROPRIATE_SECTORS = {
    "Financial Services",
    "Financial",
    "Real Estate",
}


class DataFetchError(Exception):
    """Raised when we can't pull the bare-minimum fields for a ticker."""


# Only a URL's query string goes; a '?' elsewhere in the message is just prose.
_URL_QUERY = re.compile(r"(https?://[^\s?'\"]+)\?[^\s'\")>\]]*")
# A cookie header carries several pairs, so drop the rest of the line.
_COOKIE_HEADER = re.compile(r"\b(set-cookie|cookie)\s*:[^\r\n]*", re.IGNORECASE)
# key=value, key: value and JSON-style "key": "value", quoted or not.
_SECRET_PAIR = re.compile(
    r"[\"']?\b(crumb|cookie|session|token|csrf|a1|a3)\b[\"']?\s*[=:]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s;&,'\"})]+)",
    re.IGNORECASE,
)
_MAX_REASON_CHARS = 200


def describe_failure(exc: BaseException) -> str:
    """The exception as a user can safely see it: type and message, with URL
    query strings and cookie/crumb values removed, capped in length.

    yfinance puts the full request URL in some errors, and the crumb and
    cookie ride along in it.
    """
    message = _URL_QUERY.sub(r"\1", str(exc))
    message = _COOKIE_HEADER.sub(r"\1: [removed]", message)
    message = _SECRET_PAIR.sub(r"\1=[removed]", message)
    text = f"{type(exc).__name__}: {message}".strip().rstrip(":")
    if len(text) > _MAX_REASON_CHARS:
        text = text[: _MAX_REASON_CHARS - 3] + "..."
    return text


def _source_unreachable(symbol: str, what: str, reason: str) -> DataFetchError:
    return DataFetchError(
        f"Yahoo Finance didn't answer when fetching {what} for '{symbol}', "
        f"so this isn't a problem with the ticker. ({reason})"
    )


@dataclass(frozen=True)
class CompanyData:
    ticker: str
    name: str
    sector: str
    industry: str

    current_price: float
    revenue_base: float
    shares_outstanding: float
    net_debt: float

    # CAPM inputs. Optional because a derived WACC is a bonus, not a
    # precondition -- a missing beta should cost you the derivation, not the
    # whole valuation.
    beta: Optional[float]
    market_cap: Optional[float]
    total_debt: Optional[float]
    interest_expense: Optional[float]

    historical_revenue_growth: Optional[float]
    historical_operating_margin: Optional[float]
    historical_tax_rate: Optional[float]
    historical_capex_pct: Optional[float]
    historical_da_pct: Optional[float]
    historical_wc_pct: Optional[float]

    is_dcf_inappropriate: bool

    # Set when Yahoo's company-info request did not answer and the statements
    # stood in for it: shares, debt and cash from the balance sheet, the price
    # from the last close. Holds the sanitised reason. Name, sector and beta
    # are unavailable in that case.
    quote_fallback: Optional[str] = None

    # Set when `.info` answered but had no share count, and the balance sheet
    # supplied it. Only the share count differs; the price is still Yahoo's.
    shares_fallback: Optional[str] = None


class Fundamentals(Protocol):
    """The slice of yfinance.Ticker we depend on: what a market-data source
    hands back for one symbol (see dcf/market.py)."""

    info: dict
    financials: pd.DataFrame
    cashflow: pd.DataFrame
    # Only read when `.info` did not answer (see build_company_data).
    balance_sheet: pd.DataFrame

    def history(self, period: str) -> pd.DataFrame:
        """Recent daily prices: a frame with a "Close" column."""


def build_company_data(symbol: str, tk: Fundamentals) -> CompanyData:
    """Pure function over a Ticker-like object. Easy to test with a fake."""
    info, info_failure = _safe_info(tk)
    financials, financials_failure = _safe_frame(tk, "financials")
    cashflow, _ = _safe_frame(tk, "cashflow")

    # `.info` that raised, or that came back without a name, price or share
    # count, is the source not answering. That is only worth working around
    # when the statements did load: a delisted ticker has the same near-empty
    # `.info` and no statements, and has to keep failing as "not found".
    quote_fallback: Optional[str] = None
    if not financials.empty and not _has_quote_fields(info):
        quote_fallback = info_failure or _EMPTY_INFO_REASON
        info = {}

    # A fetch that raised is the data source failing to answer, which says
    # nothing about the ticker. Every "this ticker lacks X" error below is
    # only honest when the fetch for X actually succeeded.
    if not info and financials.empty:
        failure = info_failure or financials_failure
        if failure:
            raise _source_unreachable(symbol, "company data", failure)
        raise DataFetchError(
            f"No data found for '{symbol}'. Check the ticker and try again."
        )

    name = info.get("longName") or info.get("shortName") or symbol
    sector = info.get("sector", "") or ""
    industry = info.get("industry", "") or ""

    current_price = _coerce_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    revenue_base = _latest_row(financials, "Total Revenue")
    if revenue_base is None or revenue_base <= 0:
        if financials_failure:
            raise _source_unreachable(symbol, "financials", financials_failure)
        raise DataFetchError(
            f"Could not find revenue data for '{symbol}'. "
            "Yahoo may not have financials for this ticker."
        )

    # Where shares, price, debt and cash come from. `.info` is the first
    # choice; the statements stand in when it did not answer (everything), or
    # when it answered but had no share count (shares only).
    shares_fallback: Optional[str] = None
    if quote_fallback:
        balance_sheet, _ = _safe_frame(tk, "balance_sheet")
        shares_outstanding = _balance_sheet_shares(balance_sheet)
        if not shares_outstanding:
            raise _source_unreachable(symbol, "share data", quote_fallback)
        # A delisted ticker can still have statements, and valuing it at a
        # price of zero would be a confident answer built on nothing.
        current_price = _last_close(tk)
        if current_price is None:
            if info_failure:
                raise _source_unreachable(symbol, "price data", quote_fallback)
            raise DataFetchError(
                f"Yahoo returned no quote or price for '{symbol}': it may be "
                "delisted, or Yahoo may not have answered."
            )
        total_debt = _latest_row(balance_sheet, "Total Debt")
        total_cash = _latest_row(balance_sheet, "Cash And Cash Equivalents")
        if total_cash is None:
            total_cash = _latest_row(
                balance_sheet, "Cash Cash Equivalents And Short Term Investments"
            )
    else:
        shares_outstanding = _coerce_float(info.get("sharesOutstanding"))
        if not shares_outstanding or shares_outstanding <= 0:
            balance_sheet, _ = _safe_frame(tk, "balance_sheet")
            shares_outstanding = _balance_sheet_shares(balance_sheet)
            if not shares_outstanding:
                raise DataFetchError(
                    f"Could not find shares outstanding for '{symbol}'."
                )
            shares_fallback = _NO_SHARES_REASON
        # Debt Yahoo did not report stays None on the CompanyData -- it is
        # unknown, not zero -- but net_debt has to be a number, so it counts
        # as zero there and the valuation run reports a DEBT_UNREPORTED caveat.
        total_debt = _coerce_float(info.get("totalDebt"))
        total_cash = _coerce_float(info.get("totalCash"), default=0.0)

    net_debt = (total_debt or 0.0) - (total_cash or 0.0)

    hist = _historical_averages(financials, cashflow)

    return CompanyData(
        ticker=symbol,
        name=name,
        sector=sector,
        industry=industry,
        current_price=current_price or 0.0,
        revenue_base=revenue_base,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        beta=_coerce_float(info.get("beta")),
        market_cap=_coerce_float(info.get("marketCap")),
        total_debt=total_debt,
        interest_expense=_interest_expense(financials),
        historical_revenue_growth=hist["revenue_growth"],
        historical_operating_margin=hist["operating_margin"],
        historical_tax_rate=hist["tax_rate"],
        historical_capex_pct=hist["capex_pct"],
        historical_da_pct=hist["da_pct"],
        historical_wc_pct=hist["wc_pct"],
        is_dcf_inappropriate=sector in _DCF_INAPPROPRIATE_SECTORS,
        quote_fallback=quote_fallback,
        shares_fallback=shares_fallback,
    )


# Used only when the quote cannot be fetched, so a network hiccup costs the
# user a current rate rather than the whole derivation.
FALLBACK_RISK_FREE_RATE = 0.042


def risk_free_from_history(hist: pd.DataFrame) -> tuple[float, bool]:
    """Return (rate, is_live) from a ^TNX price history.

    ^TNX quotes in percent, so a close of 4.96 means 4.96%. Anything that is
    not a plausible 10-year yield falls back, and says so with is_live=False.
    """
    close = hist["Close"].dropna()
    if close.empty:
        return FALLBACK_RISK_FREE_RATE, False
    rate = float(close.iloc[-1]) / 100.0
    # A 10-year yield outside 0-15% means the feed is wrong, not that the
    # world changed.
    if not 0.0 < rate < 0.15:
        return FALLBACK_RISK_FREE_RATE, False
    return rate, True


# ---- helpers ---------------------------------------------------------------

_EMPTY_INFO_REASON = "the company-info response had no name, price or share count"

# What counts as `.info` having answered: any one of these is a real quote.
_BASIC_QUOTE_FIELDS = (
    "longName",
    "shortName",
    "currentPrice",
    "regularMarketPrice",
    "previousClose",
    "sharesOutstanding",
)


_NO_SHARES_REASON = "the company-info response had no share count"


def _balance_sheet_shares(balance_sheet: pd.DataFrame) -> Optional[float]:
    for label in ("Ordinary Shares Number", "Share Issued"):
        shares = _latest_row(balance_sheet, label)
        if shares and shares > 0:
            return shares
    return None


def _has_quote_fields(info: dict) -> bool:
    return any(info.get(k) for k in _BASIC_QUOTE_FIELDS)


def _interest_expense(financials: pd.DataFrame) -> Optional[float]:
    # "Interest Expense" is the headline line; the non-operating variant is
    # the fallback because some filings only break it out that way.
    interest_expense = _latest_row(financials, "Interest Expense")
    if interest_expense is None:
        interest_expense = _latest_row(financials, "Interest Expense Non Operating")
    if interest_expense is not None:
        # Reported as a positive cost on the income statement, but flip a
        # negative just in case, since yfinance is inconsistent about signs.
        interest_expense = abs(interest_expense)
    return interest_expense


def _last_close(tk: Fundamentals) -> Optional[float]:
    """The latest daily close, or None if the history is missing or unusable."""
    try:
        close = tk.history(period="5d")["Close"].dropna()
    except Exception:
        return None
    if close.empty:
        return None
    price = _coerce_float(close.iloc[-1])
    return price if price and price > 0 else None



def _safe_info(tk: Fundamentals) -> tuple[dict, Optional[str]]:
    """The info dict, plus a sanitised reason if fetching it raised."""
    try:
        info = tk.info or {}
        return (info if isinstance(info, dict) else {}), None
    except Exception as e:
        return {}, describe_failure(e)


def _safe_frame(tk: Fundamentals, attr: str) -> tuple[pd.DataFrame, Optional[str]]:
    """A statement frame, plus a sanitised reason if fetching it raised."""
    try:
        df = getattr(tk, attr)
        if isinstance(df, pd.DataFrame):
            return df, None
        return pd.DataFrame(), None
    except Exception as e:
        return pd.DataFrame(), describe_failure(e)


def _coerce_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        f = float(x)
        if pd.isna(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _latest_row(df: pd.DataFrame, label: str) -> Optional[float]:
    """Return the most recent annual value for a row label, or None."""
    if df.empty or label not in df.index:
        return None
    series = df.loc[label].dropna()
    if series.empty:
        return None
    # yfinance columns are sorted newest-first; defend against future changes.
    sorted_series = series.sort_index(ascending=False)
    return _coerce_float(sorted_series.iloc[0])


def _row_series(df: pd.DataFrame, *labels: str) -> Optional[pd.Series]:
    """Return the row matching the first available label, sorted oldest-first."""
    if df.empty:
        return None
    for label in labels:
        if label in df.index:
            s = df.loc[label].dropna()
            if not s.empty:
                return s.sort_index(ascending=True)
    return None


def _historical_averages(financials: pd.DataFrame, cashflow: pd.DataFrame) -> dict:
    """Compute mean ratios across available historical years."""
    revenue = _row_series(financials, "Total Revenue", "Operating Revenue")
    op_income = _row_series(financials, "Operating Income", "EBIT")
    tax_provision = _row_series(financials, "Tax Provision")
    pretax = _row_series(financials, "Pretax Income")
    capex = _row_series(cashflow, "Capital Expenditure")
    da = _row_series(
        cashflow,
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion",
        "Reconciled Depreciation",
    )
    delta_wc = _row_series(cashflow, "Change In Working Capital")

    revenue_growth = None
    if revenue is not None and len(revenue) >= 2:
        first, last = float(revenue.iloc[0]), float(revenue.iloc[-1])
        years = len(revenue) - 1
        if first > 0 and last > 0 and years > 0:
            revenue_growth = (last / first) ** (1 / years) - 1

    operating_margin = _safe_ratio_mean(op_income, revenue)
    tax_rate = _safe_ratio_mean(tax_provision, pretax)
    if tax_rate is not None:
        tax_rate = max(0.0, min(0.6, tax_rate))  # clip absurd values

    # Capex from yfinance is reported as a negative number; flip sign.
    capex_pct = _safe_ratio_mean(capex, revenue)
    if capex_pct is not None:
        capex_pct = abs(capex_pct)

    da_pct = _safe_ratio_mean(da, revenue)
    wc_pct = _safe_ratio_mean(delta_wc, revenue)

    return {
        "revenue_growth": revenue_growth,
        "operating_margin": operating_margin,
        "tax_rate": tax_rate,
        "capex_pct": capex_pct,
        "da_pct": da_pct,
        "wc_pct": wc_pct,
    }


def _safe_ratio_mean(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[float]:
    if num is None or den is None:
        return None
    aligned = pd.concat([num, den], axis=1, join="inner").dropna()
    if aligned.empty:
        return None
    n_col, d_col = aligned.columns[0], aligned.columns[1]
    valid = aligned[aligned[d_col] != 0]
    if valid.empty:
        return None
    ratios = valid[n_col] / valid[d_col]
    return float(ratios.mean())
