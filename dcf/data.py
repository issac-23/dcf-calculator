"""Fetch company fundamentals from Yahoo Finance and pre-compute the
historical averages we use to seed the DCF inputs.

This module isolates the yfinance dependency so the rest of the app stays
testable. The `fetch_company_data` function is the one entry point;
`_extract_*` helpers are pure and individually testable with fixture data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import pandas as pd

# Sectors where a free-cash-flow DCF is the wrong tool.
_DCF_INAPPROPRIATE_SECTORS = {
    "Financial Services",
    "Financial",
    "Real Estate",
}


class DataFetchError(Exception):
    """Raised when we can't pull the bare-minimum fields for a ticker."""


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

    historical_revenue_growth: Optional[float]
    historical_operating_margin: Optional[float]
    historical_tax_rate: Optional[float]
    historical_capex_pct: Optional[float]
    historical_da_pct: Optional[float]
    historical_wc_pct: Optional[float]

    is_dcf_inappropriate: bool
    warnings: List[str] = field(default_factory=list)


class _TickerLike(Protocol):
    """The slice of yfinance.Ticker we depend on. Lets tests inject fakes."""

    info: dict
    financials: pd.DataFrame
    cashflow: pd.DataFrame


def fetch_company_data(ticker: str) -> CompanyData:
    import yfinance as yf  # imported lazily so tests can run without it

    symbol = ticker.strip().upper()
    if not symbol:
        raise DataFetchError("Ticker symbol is empty")

    try:
        tk = yf.Ticker(symbol)
    except Exception as e:
        raise DataFetchError(f"Could not initialize ticker '{symbol}': {e}") from e

    return build_company_data(symbol, tk)


def build_company_data(symbol: str, tk: _TickerLike) -> CompanyData:
    """Pure function over a Ticker-like object. Easy to test with a fake."""
    info = _safe_info(tk)
    financials = _safe_frame(tk, "financials")
    cashflow = _safe_frame(tk, "cashflow")

    if not info and financials.empty:
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
        raise DataFetchError(
            f"Could not find revenue data for '{symbol}'. "
            "Yahoo may not have financials for this ticker."
        )

    shares_outstanding = _coerce_float(info.get("sharesOutstanding"))
    if not shares_outstanding or shares_outstanding <= 0:
        raise DataFetchError(
            f"Could not find shares outstanding for '{symbol}'."
        )

    total_debt = _coerce_float(info.get("totalDebt"), default=0.0) or 0.0
    total_cash = _coerce_float(info.get("totalCash"), default=0.0) or 0.0
    net_debt = total_debt - total_cash

    hist = _historical_averages(financials, cashflow)

    warnings: List[str] = []
    is_inappropriate = sector in _DCF_INAPPROPRIATE_SECTORS
    if is_inappropriate:
        warnings.append(
            f"DCF is unreliable for {sector.lower()} companies. "
            "Banks should be valued using equity DCF or P/B; "
            "REITs using FFO or NAV."
        )
    if hist["operating_margin"] is not None and hist["operating_margin"] < 0:
        warnings.append(
            "Historical operating margin is negative. "
            "The DCF will return a negative fair value unless you adjust margin assumptions."
        )

    return CompanyData(
        ticker=symbol,
        name=name,
        sector=sector,
        industry=industry,
        current_price=current_price or 0.0,
        revenue_base=revenue_base,
        shares_outstanding=shares_outstanding,
        net_debt=net_debt,
        historical_revenue_growth=hist["revenue_growth"],
        historical_operating_margin=hist["operating_margin"],
        historical_tax_rate=hist["tax_rate"],
        historical_capex_pct=hist["capex_pct"],
        historical_da_pct=hist["da_pct"],
        historical_wc_pct=hist["wc_pct"],
        is_dcf_inappropriate=is_inappropriate,
        warnings=warnings,
    )


# ---- helpers ---------------------------------------------------------------


def _safe_info(tk: _TickerLike) -> dict:
    try:
        info = tk.info or {}
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _safe_frame(tk: _TickerLike, attr: str) -> pd.DataFrame:
    try:
        df = getattr(tk, attr)
        if df is None:
            return pd.DataFrame()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
