"""Tests for the yfinance data layer.

We never hit the real Yahoo API in tests. A fake Ticker-like object provides
deterministic fixtures so failures point at our code, not at Yahoo flaking.
"""

import pandas as pd
import pytest

from dcf.data import (
    CompanyData,
    DataFetchError,
    build_company_data,
)


class FakeTicker:
    """Drop-in replacement for yfinance.Ticker exposing only what we use."""

    def __init__(self, info=None, financials=None, cashflow=None):
        self.info = info or {}
        self.financials = financials if financials is not None else pd.DataFrame()
        self.cashflow = cashflow if cashflow is not None else pd.DataFrame()


def _years(*ys):
    return [pd.Timestamp(f"{y}-12-31") for y in ys]


def _full_fixture():
    cols = _years(2024, 2023, 2022, 2021)  # newest-first, like real yfinance
    financials = pd.DataFrame(
        {
            cols[0]: [400_000, 100_000, 25_000, 110_000, 30_000],
            cols[1]: [380_000, 95_000, 23_000, 100_000, 28_000],
            cols[2]: [365_000, 90_000, 22_000, 95_000, 27_000],
            cols[3]: [350_000, 85_000, 21_000, 90_000, 26_000],
        },
        index=[
            "Total Revenue",
            "Operating Income",
            "Tax Provision",
            "Pretax Income",
            "Reconciled Depreciation",
        ],
    )
    cashflow = pd.DataFrame(
        {
            cols[0]: [-12_000, 30_000, -5_000],
            cols[1]: [-11_000, 28_000, -4_500],
            cols[2]: [-10_500, 27_000, -4_000],
            cols[3]: [-10_000, 26_000, -3_800],
        },
        index=[
            "Capital Expenditure",
            "Depreciation And Amortization",
            "Change In Working Capital",
        ],
    )
    info = {
        "longName": "Test Co.",
        "sector": "Technology",
        "industry": "Software",
        "currentPrice": 150.0,
        "sharesOutstanding": 1_000_000,
        "totalDebt": 50_000,
        "totalCash": 20_000,
    }
    return FakeTicker(info=info, financials=financials, cashflow=cashflow)


# ----- happy path ----------------------------------------------------------


def test_build_company_data_basic_fields():
    cd = build_company_data("TEST", _full_fixture())
    assert isinstance(cd, CompanyData)
    assert cd.ticker == "TEST"
    assert cd.name == "Test Co."
    assert cd.sector == "Technology"
    assert cd.current_price == 150.0
    assert cd.revenue_base == 400_000
    assert cd.shares_outstanding == 1_000_000
    # net_debt = totalDebt - totalCash
    assert cd.net_debt == 30_000


def test_historical_revenue_growth_is_cagr():
    cd = build_company_data("TEST", _full_fixture())
    # Revenue 350k -> 400k over 3 years => CAGR = (400/350)^(1/3) - 1
    expected = (400_000 / 350_000) ** (1 / 3) - 1
    assert cd.historical_revenue_growth == pytest.approx(expected, rel=1e-6)


def test_historical_operating_margin_is_mean():
    cd = build_company_data("TEST", _full_fixture())
    margins = [85 / 350, 90 / 365, 95 / 380, 100 / 400]
    assert cd.historical_operating_margin == pytest.approx(sum(margins) / 4, rel=1e-6)


def test_historical_tax_rate_clipped_into_range():
    cd = build_company_data("TEST", _full_fixture())
    assert cd.historical_tax_rate is not None
    assert 0.0 <= cd.historical_tax_rate <= 0.6


def test_historical_capex_pct_is_absolute_value():
    # yfinance reports capex as a negative number; we want a positive percentage.
    cd = build_company_data("TEST", _full_fixture())
    assert cd.historical_capex_pct is not None
    assert cd.historical_capex_pct > 0


def test_no_warnings_for_typical_tech_company():
    cd = build_company_data("TEST", _full_fixture())
    assert cd.warnings == []
    assert cd.is_dcf_inappropriate is False


# ----- sector flags -------------------------------------------------------


def test_financial_sector_flagged_as_inappropriate():
    fix = _full_fixture()
    fix.info["sector"] = "Financial Services"
    cd = build_company_data("TEST", fix)
    assert cd.is_dcf_inappropriate is True
    assert any("financial" in w.lower() for w in cd.warnings)


def test_real_estate_sector_flagged_as_inappropriate():
    fix = _full_fixture()
    fix.info["sector"] = "Real Estate"
    cd = build_company_data("TEST", fix)
    assert cd.is_dcf_inappropriate is True


def test_negative_operating_margin_emits_warning():
    fix = _full_fixture()
    cols = fix.financials.columns
    fix.financials.loc["Operating Income", cols] = [-50_000, -40_000, -30_000, -20_000]
    cd = build_company_data("TEST", fix)
    assert any("negative" in w.lower() for w in cd.warnings)


# ----- defensive handling --------------------------------------------------


def test_missing_revenue_raises_clear_error():
    fix = _full_fixture()
    fix.financials = fix.financials.drop("Total Revenue")
    with pytest.raises(DataFetchError, match="revenue"):
        build_company_data("TEST", fix)


def test_missing_shares_outstanding_raises():
    fix = _full_fixture()
    fix.info.pop("sharesOutstanding")
    with pytest.raises(DataFetchError, match="shares"):
        build_company_data("TEST", fix)


def test_completely_empty_data_raises():
    with pytest.raises(DataFetchError):
        build_company_data("BADTICK", FakeTicker())


def test_falls_back_to_previous_close_for_price():
    fix = _full_fixture()
    fix.info.pop("currentPrice")
    fix.info["previousClose"] = 145.0
    cd = build_company_data("TEST", fix)
    assert cd.current_price == 145.0


def test_zero_total_cash_and_debt_yields_zero_net_debt():
    fix = _full_fixture()
    fix.info["totalDebt"] = 0
    fix.info["totalCash"] = 0
    cd = build_company_data("TEST", fix)
    assert cd.net_debt == 0


def test_missing_optional_historicals_returns_none_not_raises():
    fix = _full_fixture()
    fix.cashflow = pd.DataFrame()  # wipe cash flow statement
    cd = build_company_data("TEST", fix)
    assert cd.historical_capex_pct is None
    assert cd.historical_da_pct is None
    # The non-cashflow historicals should still come through:
    assert cd.historical_operating_margin is not None
