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
    describe_failure,
)


class FakeTicker:
    """Drop-in replacement for yfinance.Ticker exposing only what we use."""

    def __init__(self, info=None, financials=None, cashflow=None):
        self.info = info or {}
        self.financials = financials if financials is not None else pd.DataFrame()
        self.cashflow = cashflow if cashflow is not None else pd.DataFrame()


class RefusingTicker(FakeTicker):
    """A source that raises on `.info`, the way Yahoo does for a blocked IP."""

    def __init__(self, error, **kwargs):
        super().__init__(**kwargs)
        self._error = error

    @property
    def info(self):
        raise self._error

    @info.setter
    def info(self, value):
        pass


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


# ----- defensive handling --------------------------------------------------


def test_missing_shares_outstanding_raises():
    fix = _full_fixture()
    fix.info.pop("sharesOutstanding")
    with pytest.raises(DataFetchError, match="shares"):
        build_company_data("TEST", fix)


# Source failures versus missing data:
#   1. .info raises but the statements load: the error blames the source,
#      not the ticker.
#   2. The exception text carries a request URL with crumb/cookie values: none
#      of that reaches the user.
#   3. The fetch succeeds and the field is simply absent: that is still the
#      ticker's problem and keeps its own message.


def test_a_refused_info_fetch_blames_the_source_not_the_ticker():
    fix = _full_fixture()
    refusing = RefusingTicker(
        ConnectionError("Too Many Requests"),
        financials=fix.financials,
        cashflow=fix.cashflow,
    )
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", refusing)
    message = str(excinfo.value)
    assert "Yahoo Finance didn't answer" in message
    assert "ConnectionError: Too Many Requests" in message
    assert "Could not find shares outstanding" not in message


def test_the_user_never_sees_the_crumb_or_cookie_in_a_source_failure():
    fix = _full_fixture()
    leaky = RuntimeError(
        "401 Client Error for url: "
        "https://query2.finance.yahoo.com/v10/quoteSummary/TEST?crumb=abc123&cookie=xyz"
    )
    refusing = RefusingTicker(leaky, financials=fix.financials, cashflow=fix.cashflow)
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", refusing)
    message = str(excinfo.value)
    assert "abc123" not in message and "xyz" not in message
    assert "RuntimeError" in message and "401 Client Error" in message


def test_json_quoted_secrets_and_whole_cookie_headers_are_removed():
    body = describe_failure(
        RuntimeError(
            'body {"crumb": "SECRET1"} Cookie: A3=d=SECRET2; B=SECRET3'
        )
    )
    assert "SECRET" not in body
    assert body.startswith("RuntimeError: body")


def test_a_question_mark_outside_a_url_is_kept():
    assert "why? because" in describe_failure(ValueError("why? because"))


def test_a_long_exception_message_is_capped():
    refusing = RefusingTicker(ValueError("x" * 1000))
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", refusing)
    assert len(describe_failure(ValueError("x" * 1000))) <= 200
    assert len(str(excinfo.value)) < 400


def test_absent_shares_with_a_working_fetch_still_blames_the_ticker():
    fix = _full_fixture()
    fix.info.pop("sharesOutstanding")
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", fix)
    message = str(excinfo.value)
    assert "Could not find shares outstanding for 'TEST'" in message
    assert "didn't answer" not in message


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


def test_total_debt_is_none_when_not_reported_and_net_debt_says_so():
    """Unreported debt is an absence, not a measurement of zero.

    Ways this goes wrong: total_debt comes back 0.0 and the WACC step treats
    the company as debt-free without anyone being told; or net_debt quietly
    drops the debt and the equity value is overstated with no sign of it.
    """
    tk = _full_fixture()
    tk.info = {k: v for k, v in tk.info.items() if k != "totalDebt"}
    d = build_company_data("TEST", tk)
    assert d.total_debt is None


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


# ----- the balance-sheet fallback ---------------------------------------------
# Failure modes (also in the PR body): the share row falling back from
# "Ordinary Shares Number" to "Share Issued"; debt missing reading as zero
# instead of unknown; cash taking the wrong row or none; a price history that
# raises or is empty taking the valuation down. The journeys run one recording,
# so these label lookups can only be reached here.


def _statements_only(balance_sheet, history=None):
    """A ticker whose `.info` is the near-empty dict yfinance can return."""
    fix = _full_fixture()
    fix.info = {"trailingPegRatio": None}
    fix.balance_sheet = balance_sheet
    if isinstance(history, Exception):
        def broken(period):
            raise history
        fix.history = broken
    else:
        usable = pd.DataFrame({"Close": [100.0]}) if history is None else history
        fix.history = lambda period: usable
    return fix


def _bs(**rows):
    col = pd.Timestamp("2024-12-31")
    return pd.DataFrame({col: list(rows.values())}, index=[k.replace("_", " ") for k in rows])


def test_shares_fall_back_to_share_issued_and_debt_and_cash_stay_honest():
    fix = _statements_only(
        _bs(Share_Issued=900.0, Cash_Cash_Equivalents_And_Short_Term_Investments=70.0)
    )
    data = build_company_data("TEST", fix)
    assert data.shares_outstanding == 900.0
    assert data.total_debt is None, "unreported debt is unknown, not zero"
    assert data.net_debt == -70.0


def test_cash_prefers_the_plain_cash_row_and_ordinary_shares_win():
    fix = _statements_only(
        _bs(
            Ordinary_Shares_Number=800.0,
            Share_Issued=900.0,
            Total_Debt=100.0,
            Cash_And_Cash_Equivalents=30.0,
            Cash_Cash_Equivalents_And_Short_Term_Investments=70.0,
        )
    )
    data = build_company_data("TEST", fix)
    assert data.shares_outstanding == 800.0
    assert data.net_debt == 70.0


@pytest.mark.parametrize(
    "history",
    [RuntimeError("boom"), pd.DataFrame({"Close": [float("nan")]}), pd.DataFrame()],
    ids=["raises", "all-nan", "empty"],
)
def test_a_dead_price_history_means_no_valuation_not_a_price_of_zero(history):
    fix = _statements_only(_bs(Ordinary_Shares_Number=800.0), history)
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", fix)
    message = str(excinfo.value)
    assert "no quote or price for 'TEST'" in message
    assert "delisted" in message and "may not have answered" in message


def test_a_dead_price_history_after_a_raised_info_blames_the_source():
    fix = _statements_only(_bs(Ordinary_Shares_Number=800.0), pd.DataFrame())
    refusing = RefusingTicker(RuntimeError("401 Client Error"))
    refusing.financials = fix.financials
    refusing.cashflow = fix.cashflow
    refusing.balance_sheet = fix.balance_sheet
    refusing.history = fix.history
    with pytest.raises(DataFetchError) as excinfo:
        build_company_data("TEST", refusing)
    message = str(excinfo.value)
    assert "Yahoo Finance didn't answer when fetching price data" in message
    assert "401 Client Error" in message


def test_partial_info_without_shares_takes_them_from_the_balance_sheet():
    fix = _full_fixture()
    fix.info.pop("sharesOutstanding")
    fix.balance_sheet = _bs(Ordinary_Shares_Number=800.0)
    data = build_company_data("TEST", fix)
    assert data.shares_outstanding == 800.0
    assert data.current_price == 150.0, "the price stays Yahoo's"
    assert data.name == "Test Co." and data.beta == 1.2
    assert data.shares_fallback and data.quote_fallback is None
