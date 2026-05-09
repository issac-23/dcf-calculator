"""Discounted cash flow valuation engine.

Pure-Python, no I/O. All inputs explicit, all outputs deterministic.
The model is a standard intermediate two-stage DCF:

    FCFF_t  = NOPAT_t + D&A_t - Capex_t - dWC_t
    NOPAT_t = Revenue_t * operating_margin * (1 - tax_rate)
    EV      = sum(FCFF_t / (1 + WACC)^t)  +  TV / (1 + WACC)^N
    TV      = FCFF_{N+1} / (WACC - g_terminal)        (Gordon Growth)
    Equity  = EV - net_debt
    Price   = Equity / shares_outstanding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DCFInputs:
    revenue_base: float
    shares_outstanding: float
    net_debt: float

    revenue_growth: float
    operating_margin: float
    tax_rate: float
    capex_pct: float
    da_pct: float
    wc_pct: float

    terminal_growth: float
    wacc: float

    projection_years: int = 5

    def validate(self) -> None:
        if self.projection_years < 1 or self.projection_years > 20:
            raise ValueError("projection_years must be between 1 and 20")
        if self.revenue_base <= 0:
            raise ValueError("revenue_base must be positive")
        if self.shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be positive")
        if not -0.5 <= self.revenue_growth <= 1.0:
            raise ValueError("revenue_growth must be between -50% and 100%")
        if not -1.0 <= self.operating_margin <= 1.0:
            raise ValueError("operating_margin must be between -100% and 100%")
        if not 0.0 <= self.tax_rate <= 0.6:
            raise ValueError("tax_rate must be between 0% and 60%")
        if not 0.0 <= self.capex_pct <= 1.0:
            raise ValueError("capex_pct must be between 0% and 100% of revenue")
        if not 0.0 <= self.da_pct <= 1.0:
            raise ValueError("da_pct must be between 0% and 100% of revenue")
        if not -0.5 <= self.wc_pct <= 0.5:
            raise ValueError("wc_pct must be between -50% and 50% of revenue")
        if not 0.01 <= self.wacc <= 0.5:
            raise ValueError("wacc must be between 1% and 50%")
        if not -0.05 <= self.terminal_growth <= 0.06:
            raise ValueError(
                "terminal_growth must be between -5% and 6% "
                "(long-run GDP growth is the realistic ceiling)"
            )
        if self.terminal_growth >= self.wacc:
            raise ValueError(
                "terminal_growth must be strictly less than wacc, "
                "otherwise terminal value is infinite"
            )


@dataclass(frozen=True)
class YearProjection:
    year: int
    revenue: float
    ebit: float
    nopat: float
    da: float
    capex: float
    delta_wc: float
    fcf: float
    discount_factor: float
    pv_fcf: float


@dataclass(frozen=True)
class DCFResult:
    inputs: DCFInputs
    projections: List[YearProjection] = field(default_factory=list)
    sum_pv_fcf: float = 0.0
    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    fair_value_per_share: float = 0.0


def run_dcf(inputs: DCFInputs) -> DCFResult:
    inputs.validate()

    projections: List[YearProjection] = []
    prev_revenue = inputs.revenue_base
    prev_wc = inputs.revenue_base * inputs.wc_pct

    for t in range(1, inputs.projection_years + 1):
        revenue = prev_revenue * (1 + inputs.revenue_growth)
        ebit = revenue * inputs.operating_margin
        nopat = ebit * (1 - inputs.tax_rate)
        da = revenue * inputs.da_pct
        capex = revenue * inputs.capex_pct
        wc = revenue * inputs.wc_pct
        delta_wc = wc - prev_wc
        fcf = nopat + da - capex - delta_wc

        discount_factor = 1.0 / ((1 + inputs.wacc) ** t)
        pv_fcf = fcf * discount_factor

        projections.append(
            YearProjection(
                year=t,
                revenue=revenue,
                ebit=ebit,
                nopat=nopat,
                da=da,
                capex=capex,
                delta_wc=delta_wc,
                fcf=fcf,
                discount_factor=discount_factor,
                pv_fcf=pv_fcf,
            )
        )
        prev_revenue = revenue
        prev_wc = wc

    sum_pv_fcf = sum(p.pv_fcf for p in projections)

    final_fcf = projections[-1].fcf
    terminal_fcf = final_fcf * (1 + inputs.terminal_growth)
    terminal_value = terminal_fcf / (inputs.wacc - inputs.terminal_growth)
    pv_terminal_value = terminal_value * projections[-1].discount_factor

    enterprise_value = sum_pv_fcf + pv_terminal_value
    equity_value = enterprise_value - inputs.net_debt
    fair_value_per_share = equity_value / inputs.shares_outstanding

    return DCFResult(
        inputs=inputs,
        projections=projections,
        sum_pv_fcf=sum_pv_fcf,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
    )


def sensitivity_grid(
    inputs: DCFInputs,
    wacc_range: List[float],
    terminal_growth_range: List[float],
) -> List[List[float]]:
    """Return a 2D grid of fair_value_per_share over (wacc x terminal_growth)."""
    grid: List[List[float]] = []
    for w in wacc_range:
        row: List[float] = []
        for g in terminal_growth_range:
            try:
                modified = DCFInputs(
                    revenue_base=inputs.revenue_base,
                    shares_outstanding=inputs.shares_outstanding,
                    net_debt=inputs.net_debt,
                    revenue_growth=inputs.revenue_growth,
                    operating_margin=inputs.operating_margin,
                    tax_rate=inputs.tax_rate,
                    capex_pct=inputs.capex_pct,
                    da_pct=inputs.da_pct,
                    wc_pct=inputs.wc_pct,
                    terminal_growth=g,
                    wacc=w,
                    projection_years=inputs.projection_years,
                )
                row.append(run_dcf(modified).fair_value_per_share)
            except ValueError:
                row.append(float("nan"))
        grid.append(row)
    return grid
