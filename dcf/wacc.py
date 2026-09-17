"""Weighted average cost of capital, derived rather than guessed.

Pure-Python, no I/O, same contract as engine.py: explicit inputs, deterministic
outputs. Fetching the market data these need is data.py's problem.

    Re   = risk_free + beta * equity_risk_premium        (CAPM)
    Rd   = interest_expense / total_debt                 (pre-tax, effective)
    V    = market_cap + total_debt
    WACC = (E/V) * Re + (D/V) * Rd * (1 - tax_rate)

Two honest caveats, both structural rather than fixable:

Cost of debt here is the average rate the company is currently paying on the
debt it already has, not the marginal rate it would pay to borrow now. Those
diverge when rates have moved since the debt was issued, which lately they
have. The marginal rate is the theoretically correct input and is not
available from public filings.

The equity risk premium is not observable at all -- it is an assumption about
the future, and every practitioner uses a different one. It stays an explicit
input with a documented default rather than being quietly baked in, because
its value is a judgment the user is entitled to disagree with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Damodaran's implied ERP for mature markets has sat roughly in the 4-6% band
# for years. 5% is a defensible midpoint, not a measurement.
DEFAULT_EQUITY_RISK_PREMIUM = 0.05

# yfinance occasionally reports a beta of 0, a negative value, or something
# absurd for thinly traded names. Outside this band we decline rather than
# produce a confident number from a bad input.
_BETA_MIN, _BETA_MAX = 0.1, 4.0


class WACCUnavailable(Exception):
    """Raised when the inputs cannot support a defensible WACC."""


@dataclass(frozen=True)
class WACCInputs:
    risk_free_rate: float
    beta: float
    equity_risk_premium: float

    market_cap: float
    total_debt: float

    # Annual interest expense. None when the filing does not break it out,
    # which is common for companies with little or no debt.
    interest_expense: Optional[float]
    tax_rate: float

    def validate(self) -> None:
        if not 0.0 <= self.risk_free_rate <= 0.20:
            raise WACCUnavailable(
                f"risk-free rate of {self.risk_free_rate:.2%} is outside 0-20%"
            )
        if not _BETA_MIN <= self.beta <= _BETA_MAX:
            raise WACCUnavailable(
                f"beta of {self.beta} is outside {_BETA_MIN}-{_BETA_MAX}; "
                "the source data is probably unreliable for this ticker"
            )
        if not 0.0 <= self.equity_risk_premium <= 0.15:
            raise WACCUnavailable(
                f"equity risk premium of {self.equity_risk_premium:.2%} is outside 0-15%"
            )
        if self.market_cap <= 0:
            raise WACCUnavailable("market cap must be positive")
        if self.total_debt < 0:
            raise WACCUnavailable("total debt cannot be negative")
        if not 0.0 <= self.tax_rate <= 0.6:
            raise WACCUnavailable("tax rate must be between 0% and 60%")


@dataclass(frozen=True)
class WACCBreakdown:
    """Every intermediate value, so the UI can show the derivation."""

    cost_of_equity: float
    cost_of_debt_pretax: Optional[float]
    cost_of_debt_after_tax: Optional[float]
    weight_equity: float
    weight_debt: float
    wacc: float

    # True when debt exists but its cost could not be established, so the
    # result is really just the cost of equity.
    debt_cost_unknown: bool = False


def compute_wacc(inputs: WACCInputs) -> WACCBreakdown:
    inputs.validate()

    cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium

    equity = inputs.market_cap
    debt = inputs.total_debt
    total = equity + debt

    # No debt: the capital structure is all equity and the debt term vanishes.
    # Reported as unknown=False because nothing is missing -- there is simply
    # no debt to price.
    if debt == 0:
        return WACCBreakdown(
            cost_of_equity=cost_of_equity,
            cost_of_debt_pretax=None,
            cost_of_debt_after_tax=None,
            weight_equity=1.0,
            weight_debt=0.0,
            wacc=cost_of_equity,
        )

    if inputs.interest_expense is None or inputs.interest_expense <= 0:
        # Debt exists but we cannot price it. Weighting it at the cost of
        # equity would overstate WACC; ignoring the debt entirely understates
        # it. We return the equity cost and flag it, so the caller can say so
        # out loud instead of presenting a number as fully derived.
        return WACCBreakdown(
            cost_of_equity=cost_of_equity,
            cost_of_debt_pretax=None,
            cost_of_debt_after_tax=None,
            weight_equity=equity / total,
            weight_debt=debt / total,
            wacc=cost_of_equity,
            debt_cost_unknown=True,
        )

    cost_of_debt_pretax = inputs.interest_expense / debt
    cost_of_debt_after_tax = cost_of_debt_pretax * (1 - inputs.tax_rate)

    weight_equity = equity / total
    weight_debt = debt / total
    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt_after_tax

    return WACCBreakdown(
        cost_of_equity=cost_of_equity,
        cost_of_debt_pretax=cost_of_debt_pretax,
        cost_of_debt_after_tax=cost_of_debt_after_tax,
        weight_equity=weight_equity,
        weight_debt=weight_debt,
        wacc=wacc,
    )
