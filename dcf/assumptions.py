"""Turn a `CompanyData` into the assumptions the DCF starts from.

This used to live inside app.py as module-level helpers. It moved here so the
E2E suite exercises the same seeding the UI does. When these lived in the
Streamlit script the only way to test them was to re-implement them in the
test, which meant the test agreed with itself and not with the app.

Nothing here touches Streamlit or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

from dcf.data import CompanyData
from dcf.engine import DCFInputs
from dcf.wacc import (
    DEFAULT_EQUITY_RISK_PREMIUM,
    WACCBreakdown,
    WACCInputs,
    WACCUnavailable,
    compute_wacc,
)

# The US federal statutory rate, used when a filing history gives no effective
# rate to work from.
FALLBACK_TAX_RATE = 0.21


@dataclass(frozen=True)
class Assumption:
    """Everything the model and the page need to know about one assumption.

    Values are in the unit a person sees: percent points for "%", whole years
    for "years". Adding an assumption is adding one entry to ASSUMPTIONS --
    the sliders, the seeding and the clamping all iterate it.
    """

    key: str
    label: str
    fallback: float
    lo: float
    hi: float
    step: float
    # "operating" assumptions sit above the discount-rate heading, "discount"
    # ones under it.
    group: Literal["operating", "discount"]
    unit: Literal["%", "years"] = "%"
    # The CompanyData field whose history seeds the default, if there is one.
    # Without it the default is just the fallback (or, for WACC, CAPM).
    history: Optional[str] = None
    help: Optional[str] = None
    # Help to show instead when the default could not be derived from data.
    help_unseeded: Optional[str] = None

    @property
    def format(self) -> Optional[str]:
        if self.unit == "years":
            return None
        return "%.1f%%" if self.step >= 0.5 else "%.2f%%"


# The bounds are the real widget bounds: a historical average outside them is
# clamped, which is a documented limitation rather than an accident (see
# LIMITATIONS.md section 5 -- NVDA's 100% historical growth clamps to 30%).
_SPECS = (
    Assumption(
        "revenue_growth", "Revenue growth (annual)", 5.0, -10.0, 30.0, 0.5,
        "operating", history="historical_revenue_growth",
    ),
    Assumption(
        "operating_margin", "Operating margin", 15.0, -20.0, 60.0, 0.5,
        "operating", history="historical_operating_margin",
    ),
    Assumption(
        "tax_rate", "Tax rate", 21.0, 0.0, 40.0, 0.5,
        "operating", history="historical_tax_rate",
    ),
    Assumption(
        "capex_pct", "Capex / revenue", 5.0, 0.0, 30.0, 0.5,
        "operating", history="historical_capex_pct",
    ),
    Assumption(
        "da_pct", "D&A / revenue", 5.0, 0.0, 30.0, 0.5,
        "operating", history="historical_da_pct",
    ),
    Assumption(
        "wc_pct", "Working capital / revenue", 2.0, -10.0, 30.0, 0.5,
        "operating", history="historical_wc_pct",
        help=(
            "Working capital balance as a percent of revenue. Year-over-year "
            "changes drive ΔWC in the FCF calc."
        ),
    ),
    Assumption(
        "erp", "Equity risk premium", DEFAULT_EQUITY_RISK_PREMIUM * 100, 3.0, 8.0, 0.25,
        "discount",
        help=(
            "The only CAPM input that cannot be looked up — it is an assumption "
            "about future returns, not a measurement. Damodaran's implied "
            "premium for mature markets has sat in the 4–6% band for years."
        ),
    ),
    # Seeded from CAPM when it can be derived, which is why it has no history.
    Assumption(
        "wacc", "WACC (discount rate)", 9.0, 4.0, 20.0, 0.25, "discount",
        help=(
            "Seeded from CAPM using this company's beta and capital structure. "
            "Drag to override."
        ),
        help_unseeded="Could not be derived for this ticker, so this is a generic default.",
    ),
    Assumption("terminal_growth", "Terminal growth rate", 2.5, 0.0, 5.0, 0.25, "discount"),
    Assumption(
        "projection_years", "Projection years", 5, 3, 10, 1, "discount", unit="years"
    ),
)

ASSUMPTIONS = {a.key: a for a in _SPECS}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def default_pct(historical: Optional[float], key: str) -> float:
    """Pick a slider default in percent points, clamped to the slider's range."""
    spec = ASSUMPTIONS[key]
    if historical is None:
        return spec.fallback
    return float(clamp(historical * 100, spec.lo, spec.hi))


def clamped_from_history(company: CompanyData) -> set[str]:
    """Which defaults the sliders had to clip to fit their range.

    Reported so the caller can say so out loud. A silently clamped default
    means the model is not valuing the company its own data describes.
    """
    clipped = set()
    for spec in ASSUMPTIONS.values():
        value = getattr(company, spec.history) if spec.history else None
        if value is not None and not (spec.lo <= value * 100 <= spec.hi):
            clipped.add(spec.key)
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
        for name, value in (
            ("beta", company.beta),
            ("market cap", company.market_cap),
            ("total debt", company.total_debt),
        )
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
                    total_debt=company.total_debt,
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
    spec = ASSUMPTIONS["wacc"]
    if breakdown is None:
        return spec.fallback
    return float(clamp(breakdown.wacc * 100, spec.lo, spec.hi))


class SeededAssumptions(NamedTuple):
    inputs: DCFInputs
    wacc: Optional[WACCBreakdown]
    wacc_reason: Optional[str]
    clamped: set[str]


def seed_inputs(
    company: CompanyData,
    risk_free: float,
    erp: float,
) -> SeededAssumptions:
    """Everything the app would show on first load, with no slider touched."""
    breakdown, reason = derive_wacc(company, risk_free, erp)
    fields = {
        spec.key: default_pct(getattr(company, spec.history), spec.key) / 100
        for spec in ASSUMPTIONS.values()
        if spec.history
    }
    inputs = DCFInputs(
        revenue_base=company.revenue_base,
        shares_outstanding=company.shares_outstanding,
        net_debt=company.net_debt,
        terminal_growth=ASSUMPTIONS["terminal_growth"].fallback / 100,
        wacc=seeded_wacc_pct(breakdown) / 100,
        projection_years=ASSUMPTIONS["projection_years"].fallback,
        **fields,
    )
    return SeededAssumptions(inputs, breakdown, reason, clamped_from_history(company))
