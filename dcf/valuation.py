"""One valuation of one company: everything the app and the E2E report show.

`run_valuation` takes a company, the market's risk-free rate and whatever the
user has changed, and returns the whole answer: the inputs actually used, which
seeded defaults were clamped, the WACC breakdown (or why there is none), the
DCF result, upside and verdict, how much of the value is terminal value, the
growth rate the market implies against what the company delivered, and the
sensitivity grid.

Before this module the Streamlit script worked all of that out itself, and
tests/e2e/report.py worked out a second copy of the arithmetic. The verdict
thresholds and the optimistic/pessimistic reading of implied growth lived only
in app.py, where nothing could assert on them. Now both surfaces render the same
object, and the report is simply a valuation with no overrides.

Nothing here touches Streamlit or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Literal, Mapping, Optional

import numpy as np

from dcf.assumptions import seed_inputs
from dcf.data import CompanyData
from dcf.engine import DCFInputs, DCFResult, implied_revenue_growth, run_dcf, sensitivity_grid
from dcf.wacc import DEFAULT_EQUITY_RISK_PREMIUM, WACCBreakdown

# Fair value more than this far from the market price is called under- or
# overvalued; inside it the two are "fairly valued".
VERDICT_BAND = 0.10

# Implied growth more than this far (as a fraction, so 0.05 is 5 percentage
# points) from the company's own history reads as the market being optimistic
# or pessimistic; inside it, in line.
GROWTH_GAP_BAND = 0.05

Verdict = Literal["undervalued", "overvalued", "fair", "no_price"]
GrowthReading = Literal["optimistic", "pessimistic", "in_line"]

# What a person may override, keyed the way the sliders are: percent points,
# except projection_years, which is whole years. Anything absent keeps its
# seeded default, so an empty mapping is the first-load valuation.
Overrides = Mapping[str, float]


@dataclass(frozen=True)
class ImpliedGrowth:
    """What the reverse DCF found, and how it compares with history."""

    # None means no growth rate in the solver's range reproduces the price.
    rate: Optional[float]
    # rate minus the company's historical growth, as a fraction. None when
    # either side is missing.
    gap_vs_history: Optional[float]
    reading: Optional[GrowthReading]


@dataclass(frozen=True)
class Sensitivity:
    """Fair value per share over (WACC x terminal growth); NaN where undefined."""

    wacc: List[float]
    terminal_growth: List[float]
    fair_value: List[List[float]]


@dataclass(frozen=True)
class Valuation:
    company: CompanyData
    risk_free: float
    # Equity risk premium used for the WACC derivation, as a fraction.
    equity_risk_premium: float

    inputs: DCFInputs
    # Fields whose seeded default had to be clipped to fit the assumption's range.
    clamped: frozenset
    # Exactly one of these two is None.
    wacc: Optional[WACCBreakdown]
    wacc_reason: Optional[str]

    result: DCFResult
    # (fair value - price) / price. None without a usable market price.
    upside: Optional[float]
    verdict: Verdict
    # Share of enterprise value that is terminal value. None when EV is zero.
    tv_share: Optional[float]
    # None when there is no market price to solve against.
    implied: Optional[ImpliedGrowth]
    sensitivity: Sensitivity


def run_valuation(
    company: CompanyData,
    risk_free: float,
    overrides: Optional[Overrides] = None,
) -> Valuation:
    """Value `company`. Raises ValueError when the assumptions are invalid."""
    overrides = overrides or {}

    erp_pct = overrides.get("erp", DEFAULT_EQUITY_RISK_PREMIUM * 100)
    seed = seed_inputs(company, risk_free, erp_pct / 100)

    changes = {}
    for field, value in overrides.items():
        if field == "erp":
            continue
        changes[field] = int(value) if field == "projection_years" else value / 100
    inputs = replace(seed.inputs, **changes)

    result = run_dcf(inputs)
    price = company.current_price

    upside: Optional[float] = None
    implied: Optional[ImpliedGrowth] = None
    verdict: Verdict = "no_price"
    if price and price > 0:
        upside = (result.fair_value_per_share - price) / price
        if upside > VERDICT_BAND:
            verdict = "undervalued"
        elif upside < -VERDICT_BAND:
            verdict = "overvalued"
        else:
            verdict = "fair"
        implied = _implied_growth(inputs, company)

    tv_share = (
        result.pv_terminal_value / result.enterprise_value
        if result.enterprise_value
        else None
    )

    return Valuation(
        company=company,
        risk_free=risk_free,
        equity_risk_premium=erp_pct / 100,
        inputs=inputs,
        clamped=frozenset(seed.clamped),
        wacc=seed.wacc,
        wacc_reason=seed.wacc_reason,
        result=result,
        upside=upside,
        verdict=verdict,
        tv_share=tv_share,
        implied=implied,
        sensitivity=_sensitivity(inputs),
    )


def _implied_growth(inputs: DCFInputs, company: CompanyData) -> ImpliedGrowth:
    rate = implied_revenue_growth(inputs, company.current_price)
    hist = company.historical_revenue_growth
    if rate is None or hist is None:
        return ImpliedGrowth(rate, None, None)
    gap = rate - hist
    if gap > GROWTH_GAP_BAND:
        reading: GrowthReading = "optimistic"
    elif gap < -GROWTH_GAP_BAND:
        reading = "pessimistic"
    else:
        reading = "in_line"
    return ImpliedGrowth(rate, gap, reading)


def _sensitivity(inputs: DCFInputs) -> Sensitivity:
    wacc = np.round(
        np.arange(max(0.04, inputs.wacc - 0.025), inputs.wacc + 0.026, 0.005), 4
    ).tolist()
    growth = np.round(np.arange(0.0, 0.041, 0.005), 4).tolist()
    return Sensitivity(wacc, growth, sensitivity_grid(inputs, wacc, growth))
