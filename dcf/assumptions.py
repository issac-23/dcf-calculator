"""Turn a `CompanyData` into the assumptions the DCF starts from.

This used to live inside app.py as module-level helpers. It moved here so the
E2E suite exercises the same seeding the UI does. When these lived in the
Streamlit script the only way to test them was to re-implement them in the
test, which meant the test agreed with itself and not with the app.

Nothing here touches Streamlit or the network.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

from dcf.data import CompanyData
from dcf.engine import DCFInputs
from dcf.wacc import WACCBreakdown, WACCInputs, WACCUnavailable, compute_wacc

# The US federal statutory rate, used when a filing history gives no effective
# rate to work from.
FALLBACK_TAX_RATE = 0.21

# Each slider's (fallback, min, max) in percent points. The bounds are the
# real widget bounds in app.py: a historical average outside them is clamped,
# which is a documented limitation rather than an accident (see LIMITATIONS.md
# section 5 -- NVDA's 100% historical growth clamps to 30%).
SLIDER_BOUNDS = {
    "revenue_growth": (5.0, -10.0, 30.0),
    "operating_margin": (15.0, -20.0, 60.0),
    "tax_rate": (21.0, 0.0, 40.0),
    "capex_pct": (5.0, 0.0, 30.0),
    "da_pct": (5.0, 0.0, 30.0),
    "wc_pct": (2.0, -10.0, 30.0),
}

WACC_MIN_PCT, WACC_MAX_PCT = 4.0, 20.0
WACC_FALLBACK_PCT = 9.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def default_pct(historical: Optional[float], field: str) -> float:
    """Pick a slider default in percent points, clamped to the slider's range."""
    fallback, lo, hi = SLIDER_BOUNDS[field]
    if historical is None:
        return fallback
    return float(clamp(historical * 100, lo, hi))


def clamped_from_history(company: CompanyData) -> set[str]:
    """Which defaults the sliders had to clip to fit their range.

    Reported so the caller can say so out loud. A silently clamped default
    means the model is not valuing the company its own data describes.
    """
    historicals = {
        "revenue_growth": company.historical_revenue_growth,
        "operating_margin": company.historical_operating_margin,
        "tax_rate": company.historical_tax_rate,
        "capex_pct": company.historical_capex_pct,
        "da_pct": company.historical_da_pct,
        "wc_pct": company.historical_wc_pct,
    }
    clipped = set()
    for field, value in historicals.items():
        if value is None:
            continue
        _, lo, hi = SLIDER_BOUNDS[field]
        if not (lo <= value * 100 <= hi):
            clipped.add(field)
    return clipped


def derive_wacc(
    company: CompanyData, risk_free: float, erp: float
) -> tuple[Optional[WACCBreakdown], Optional[str]]:
    """Return (breakdown, reason_it_failed). Exactly one is None.

    Every missing-input case is reported as its own sentence rather than one
    generic failure, because "no beta for this ticker" and "market cap
    unavailable" call for different responses from the user.
    """
    missing = [
        name
        for name, value in (("beta", company.beta), ("market cap", company.market_cap))
        if value is None
    ]
    if missing:
        return None, f"Yahoo did not return {' or '.join(missing)} for this ticker."

    try:
        return (
            compute_wacc(
                WACCInputs(
                    risk_free_rate=risk_free,
                    beta=company.beta,
                    equity_risk_premium=erp,
                    market_cap=company.market_cap,
                    total_debt=company.total_debt or 0.0,
                    interest_expense=company.interest_expense,
                    tax_rate=company.historical_tax_rate
                    if company.historical_tax_rate is not None
                    else FALLBACK_TAX_RATE,
                )
            ),
            None,
        )
    except WACCUnavailable as e:
        return None, str(e)


def seeded_wacc_pct(breakdown: Optional[WACCBreakdown]) -> float:
    """The percent-point value the WACC slider opens on."""
    if breakdown is None:
        return WACC_FALLBACK_PCT
    return float(clamp(breakdown.wacc * 100, WACC_MIN_PCT, WACC_MAX_PCT))


class Seeded(NamedTuple):
    inputs: DCFInputs
    wacc: Optional[WACCBreakdown]
    wacc_reason: Optional[str]
    clamped: set[str]


def seed_inputs(
    company: CompanyData,
    risk_free: float,
    erp: float,
    terminal_growth: float = 0.025,
    projection_years: int = 5,
) -> Seeded:
    """Everything the app would show on first load, with no slider touched."""
    breakdown, reason = derive_wacc(company, risk_free, erp)
    inputs = DCFInputs(
        revenue_base=company.revenue_base,
        shares_outstanding=company.shares_outstanding,
        net_debt=company.net_debt,
        revenue_growth=default_pct(company.historical_revenue_growth, "revenue_growth") / 100,
        operating_margin=default_pct(company.historical_operating_margin, "operating_margin") / 100,
        tax_rate=default_pct(company.historical_tax_rate, "tax_rate") / 100,
        capex_pct=default_pct(company.historical_capex_pct, "capex_pct") / 100,
        da_pct=default_pct(company.historical_da_pct, "da_pct") / 100,
        wc_pct=default_pct(company.historical_wc_pct, "wc_pct") / 100,
        terminal_growth=terminal_growth,
        wacc=seeded_wacc_pct(breakdown) / 100,
        projection_years=projection_years,
    )
    return Seeded(inputs, breakdown, reason, clamped_from_history(company))
