"""Tests for the CAPM WACC derivation.

The first test is the anchor: real AAPL inputs against a number worked out by
hand before the module existed, the same way test_engine.py pins the DCF to a
textbook problem. Everything after it covers a branch or a refusal.
"""

from __future__ import annotations

import pytest

from dcf.wacc import (
    DEFAULT_EQUITY_RISK_PREMIUM,
    WACCInputs,
    WACCUnavailable,
    compute_wacc,
)


def _aapl(**overrides) -> WACCInputs:
    """Real AAPL figures, September 2026."""
    defaults = dict(
        risk_free_rate=0.04959,      # ^TNX close
        beta=1.085,
        equity_risk_premium=0.050,
        market_cap=4_851_251_544_064,
        total_debt=84_343_996_416,
        interest_expense=3_933_000_000,
        tax_rate=0.156,
    )
    defaults.update(overrides)
    return WACCInputs(**defaults)


def test_matches_hand_computed_aapl():
    # Worked by hand: Re = 4.959% + 1.085 x 5% = 10.384%.
    # Rd = 3.933B / 84.344B = 4.663% pre-tax, 3.936% after 15.6% tax.
    # Weights 98.29% / 1.71% give WACC = 10.27%.
    b = compute_wacc(_aapl())
    assert b.cost_of_equity == pytest.approx(0.10384, abs=1e-5)
    assert b.cost_of_debt_pretax == pytest.approx(0.04663, abs=1e-5)
    assert b.cost_of_debt_after_tax == pytest.approx(0.03936, abs=1e-5)
    assert b.weight_equity == pytest.approx(0.98291, abs=1e-5)
    assert b.weight_debt == pytest.approx(0.01709, abs=1e-5)
    assert b.wacc == pytest.approx(0.1027, abs=1e-4)
    assert not b.debt_cost_unknown


def test_weights_sum_to_one():
    b = compute_wacc(_aapl())
    assert b.weight_equity + b.weight_debt == pytest.approx(1.0)


def test_wacc_sits_between_the_two_capital_costs():
    """A weighted average cannot fall outside the things it averages."""
    b = compute_wacc(_aapl())
    lo = min(b.cost_of_equity, b.cost_of_debt_after_tax)
    hi = max(b.cost_of_equity, b.cost_of_debt_after_tax)
    assert lo <= b.wacc <= hi


def test_higher_beta_raises_wacc():
    low = compute_wacc(_aapl(beta=0.8)).wacc
    high = compute_wacc(_aapl(beta=1.6)).wacc
    assert high > low


def test_more_debt_lowers_wacc_when_debt_is_cheaper():
    """The tax shield is the whole reason leverage moves WACC down here."""
    light = compute_wacc(_aapl(total_debt=10_000_000_000, interest_expense=466_000_000))
    heavy = compute_wacc(_aapl(total_debt=2_000_000_000_000, interest_expense=93_260_000_000))
    assert heavy.cost_of_debt_after_tax < heavy.cost_of_equity
    assert heavy.wacc < light.wacc


def test_zero_debt_gives_pure_cost_of_equity():
    b = compute_wacc(_aapl(total_debt=0, interest_expense=None))
    assert b.wacc == b.cost_of_equity
    assert b.weight_equity == 1.0
    assert b.weight_debt == 0.0
    # Nothing is missing here -- there is simply no debt to price.
    assert not b.debt_cost_unknown


def test_debt_without_interest_expense_is_flagged_not_guessed():
    b = compute_wacc(_aapl(interest_expense=None))
    assert b.debt_cost_unknown
    assert b.cost_of_debt_pretax is None
    assert b.wacc == b.cost_of_equity
    # The weights still reflect the real capital structure, so the UI can show
    # that debt exists even though its cost is unknown.
    assert b.weight_debt > 0


@pytest.mark.parametrize("beta", [0.0, -0.5, 4.5, 12.0])
def test_implausible_beta_is_refused(beta):
    """yfinance returns junk betas for thinly traded names. Refuse rather than
    emit a confident number built on one."""
    with pytest.raises(WACCUnavailable, match="beta"):
        compute_wacc(_aapl(beta=beta))


@pytest.mark.parametrize(
    "field,value",
    [
        ("risk_free_rate", -0.01),
        ("risk_free_rate", 0.25),
        ("equity_risk_premium", 0.20),
        ("market_cap", 0),
        ("total_debt", -1),
        ("tax_rate", 0.75),
    ],
)
def test_out_of_range_inputs_are_refused(field, value):
    with pytest.raises(WACCUnavailable):
        compute_wacc(_aapl(**{field: value}))


def test_default_erp_is_in_the_documented_band():
    """Guards the constant against a careless edit; 4-6% is the mature-market
    range the docstring cites."""
    assert 0.04 <= DEFAULT_EQUITY_RISK_PREMIUM <= 0.06


def test_result_lands_in_the_range_the_engine_accepts():
    """run_dcf rejects a WACC outside 1-50%, so a derived value has to clear
    that bar or the app would hand the engine something it refuses."""
    b = compute_wacc(_aapl())
    assert 0.01 <= b.wacc <= 0.5
