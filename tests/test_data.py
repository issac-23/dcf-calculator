"""Tests for the yfinance data layer.

We never hit the real Yahoo API in tests. A fake Ticker-like object provides
deterministic fixtures so failures point at our code, not at Yahoo flaking.
"""

import pandas as pd
import pytest

from dcf.data import (
    FALLBACK_RISK_FREE_RATE,
    CompanyData,
    DataFetchError,
    build_company_data,
    fetch_risk_free_rate,
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
            cols[0]: [400_000, 100_000, 25_000, 110_000, 30_000, 2_400],
            cols[1]: [380_000, 95_000, 23_000, 100_000, 28_000, 2_300],
            cols[2]: [365_000, 90_000, 22_000, 95_000, 27_000, 2_200],
            cols[3]: [350_000, 85_000, 21_000, 90_000, 26_000, 2_100],
        },
        index=[
            "Total Revenue",
            "Operating Income",
            "Tax Provision",
            "Pretax Income",
            "Reconciled Depreciation",
            "Interest Expense",
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
        "beta": 1.2,
        "marketCap": 150_000_000,
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


def test_negative_operating_margin_emits_warning():
    fix = _full_fixture()
    cols = fix.financials.columns
    fix.financials.loc["Operating Income", cols] = [-50_000, -40_000, -30_000, -20_000]
    cd = build_company_data("TEST", fix)
    assert any("negative" in w.lower() for w in cd.warnings)


# ----- defensive handling --------------------------------------------------


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


def test_missing_optional_historicals_returns_none_not_raises():
    fix = _full_fixture()
    fix.cashflow = pd.DataFrame()  # wipe cash flow statement
    cd = build_company_data("TEST", fix)
    assert cd.historical_capex_pct is None
    assert cd.historical_da_pct is None
    # The non-cashflow historicals should still come through:
    assert cd.historical_operating_margin is not None


# ----- CAPM inputs ----------------------------------------------------------


def test_capm_fields_are_extracted():
    d = build_company_data("TEST", _full_fixture())
    assert d.beta == 1.2
    assert d.market_cap == 150_000_000
    assert d.total_debt == 50_000
    assert d.interest_expense == 2_400  # newest year, not an average


def test_capm_fields_are_none_when_absent_rather_than_zero():
    """A missing beta has to be distinguishable from a beta of zero: one means
    'no derivation', the other is a value the WACC module refuses."""
    tk = _full_fixture()
    tk.info = {k: v for k, v in tk.info.items() if k not in ("beta", "marketCap")}
    d = build_company_data("TEST", tk)
    assert d.beta is None
    assert d.market_cap is None
    # The valuation itself must survive losing the CAPM inputs.
    assert d.revenue_base > 0
    assert d.shares_outstanding > 0


def test_interest_expense_falls_back_to_the_non_operating_line():
    tk = _full_fixture()
    tk.financials = tk.financials.rename(
        index={"Interest Expense": "Interest Expense Non Operating"}
    )
    d = build_company_data("TEST", tk)
    assert d.interest_expense == 2_400


def test_interest_expense_is_none_when_not_reported():
    tk = _full_fixture()
    tk.financials = tk.financials.drop(index="Interest Expense")
    d = build_company_data("TEST", tk)
    assert d.interest_expense is None


def test_negative_interest_expense_is_normalised_to_a_cost():
    """yfinance is inconsistent about signs on this line."""
    tk = _full_fixture()
    tk.financials.loc["Interest Expense"] = -2_400
    d = build_company_data("TEST", tk)
    assert d.interest_expense == 2_400


# ----- the risk-free rate ---------------------------------------------------
#
# Written failure-modes-first, per AGENTS.md. `fetch_risk_free_rate` feeds
# straight into every WACC, so a wrong answer here is wrong everywhere, and
# until now nothing exercised it -- both test suites monkeypatched it away.
#
# The ways it can be wrong:
#   1. It returns a percent where a decimal is expected. ^TNX quotes 4.96 to
#      mean 4.96%, so a missing /100 turns a 5% rate into 496% and every
#      company becomes worthless. This is the expensive one.
#   2. Yahoo returns rows but every Close is NaN, and `.iloc[-1]` on the
#      dropna'd series raises instead of falling back.
#   3. The feed changes units or returns garbage, and we pass it through as a
#      real rate instead of recognising it as broken.
#   4. The network is down and the exception escapes, taking the whole
#      valuation with it rather than costing only a live rate.
#   5. The fallback path claims to be live, so the UI tells the user a stale
#      constant is today's Treasury yield.


def _close(*values):
    return lambda: pd.DataFrame({"Close": list(values)})


def test_a_normal_quote_is_converted_from_percent_to_decimal():
    rate, is_live = fetch_risk_free_rate(_close(4.10, 4.96))
    assert rate == pytest.approx(0.0496), "^TNX quotes percent; this needs /100"
    assert is_live


def test_an_all_nan_series_falls_back_instead_of_raising():
    rate, is_live = fetch_risk_free_rate(_close(float("nan"), float("nan")))
    assert (rate, is_live) == (FALLBACK_RISK_FREE_RATE, False)


def test_an_empty_series_falls_back():
    rate, is_live = fetch_risk_free_rate(_close())
    assert (rate, is_live) == (FALLBACK_RISK_FREE_RATE, False)


@pytest.mark.parametrize("quote", [0.0, -1.5, 15.0, 518.0])
def test_a_quote_outside_the_plausible_band_is_treated_as_a_broken_feed(quote):
    """15%+ means the units changed, not that the world did."""
    rate, is_live = fetch_risk_free_rate(_close(quote))
    assert (rate, is_live) == (FALLBACK_RISK_FREE_RATE, False)


def test_a_dead_network_costs_the_rate_and_not_the_valuation():
    def boom():
        raise RuntimeError("Yahoo unreachable")

    rate, is_live = fetch_risk_free_rate(boom)
    assert (rate, is_live) == (FALLBACK_RISK_FREE_RATE, False)
