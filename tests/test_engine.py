"""Tests for the DCF engine.

The first test is the most important: a textbook DCF problem with a known
hand-computed answer. If this ever fails, the engine has drifted.
"""

import math

import pytest

from dcf import DCFInputs, run_dcf
from dcf.engine import implied_revenue_growth, sensitivity_grid


def _base_inputs(**overrides) -> DCFInputs:
    defaults = dict(
        revenue_base=1000.0,
        shares_outstanding=100.0,
        net_debt=0.0,
        revenue_growth=0.10,
        operating_margin=0.20,
        tax_rate=0.25,
        capex_pct=0.05,
        da_pct=0.05,
        wc_pct=0.0,
        terminal_growth=0.03,
        wacc=0.10,
        projection_years=5,
    )
    defaults.update(overrides)
    return DCFInputs(**defaults)


# ----- Hand-computed textbook case ------------------------------------------
#
#   Revenue base = 1000, growing 10% annually
#   Operating margin 20%, tax rate 25%, so NOPAT_t = Revenue_t * 0.15
#   D&A = Capex = 5% of revenue (cancel out), no WC change
#   => FCF_t = NOPAT_t = Revenue_t * 0.15
#   FCF_1 = 165, growing 10% per year
#   Each PV_FCF_t = 165 * 1.1^(t-1) / 1.1^t = 150 exactly
#   Sum of PV(FCF) for 5 years = 750
#   FCF_5 = 165 * 1.1^4 = 241.5765
#   TV = FCF_5 * 1.03 / 0.07 = 3554.6256...
#   PV(TV) = TV / 1.1^5 = 2207.146...
#   EV = 750 + 2207.146 = 2957.146
#   Equity per share = 29.5715


def test_textbook_case_matches_hand_computed_answer():
    result = run_dcf(_base_inputs())

    assert result.sum_pv_fcf == pytest.approx(750.0, abs=0.01)
    assert result.terminal_value == pytest.approx(3554.6256, abs=0.01)
    assert result.pv_terminal_value == pytest.approx(2207.146, abs=0.01)
    assert result.enterprise_value == pytest.approx(2957.146, abs=0.01)
    assert result.equity_value == pytest.approx(2957.146, abs=0.01)
    assert result.fair_value_per_share == pytest.approx(29.5715, abs=0.001)


def test_textbook_case_year_by_year_fcf():
    result = run_dcf(_base_inputs())
    expected_fcf = [165.0, 181.5, 199.65, 219.615, 241.5765]
    actual_fcf = [p.fcf for p in result.projections]
    for actual, expected in zip(actual_fcf, expected_fcf):
        assert actual == pytest.approx(expected, abs=0.001)


def test_textbook_case_pv_fcf_each_year_is_150():
    # Special property: when growth rate equals discount rate, every PV(FCF)
    # collapses to FCF_1 / (1 + r). This catches off-by-one discounting bugs.
    result = run_dcf(_base_inputs())
    for p in result.projections:
        assert p.pv_fcf == pytest.approx(150.0, abs=0.001)


# ----- Property tests -------------------------------------------------------


def test_more_net_debt_decreases_equity_value():
    no_debt = run_dcf(_base_inputs(net_debt=0.0)).fair_value_per_share
    with_debt = run_dcf(_base_inputs(net_debt=500.0)).fair_value_per_share
    assert no_debt > with_debt
    # Specifically: 500 of net debt across 100 shares = $5/share
    assert no_debt - with_debt == pytest.approx(5.0, abs=0.001)


def test_zero_shares_outstanding_raises():
    with pytest.raises(ValueError, match="shares_outstanding"):
        run_dcf(_base_inputs(shares_outstanding=0.0))


def test_negative_revenue_base_raises():
    with pytest.raises(ValueError, match="revenue_base"):
        run_dcf(_base_inputs(revenue_base=-100.0))


def test_unrealistic_terminal_growth_raises():
    with pytest.raises(ValueError, match="terminal_growth"):
        run_dcf(_base_inputs(terminal_growth=0.10, wacc=0.20))


def test_zero_wacc_raises():
    with pytest.raises(ValueError, match="wacc"):
        run_dcf(_base_inputs(wacc=0.0))


def test_invalid_projection_years_raises():
    with pytest.raises(ValueError, match="projection_years"):
        run_dcf(_base_inputs(projection_years=0))
    with pytest.raises(ValueError, match="projection_years"):
        run_dcf(_base_inputs(projection_years=50))


# ----- Sensitivity grid -----------------------------------------------------


def test_sensitivity_grid_shape_and_monotonicity():
    inputs = _base_inputs()
    waccs = [0.08, 0.10, 0.12]
    growths = [0.01, 0.025, 0.04]
    grid = sensitivity_grid(inputs, waccs, growths)

    assert len(grid) == 3
    assert all(len(row) == 3 for row in grid)

    # Within a row (fixed wacc), higher terminal growth => higher value
    for row in grid:
        assert row == sorted(row)

    # Within a column (fixed terminal growth), higher wacc => lower value
    for j in range(3):
        column = [grid[i][j] for i in range(3)]
        assert column == sorted(column, reverse=True)


def test_sensitivity_grid_handles_invalid_combinations():
    # Pair where g >= wacc should produce NaN, not crash.
    inputs = _base_inputs()
    grid = sensitivity_grid(inputs, [0.04], [0.05])
    assert math.isnan(grid[0][0])


# ----- Reverse DCF ----------------------------------------------------------


def test_implied_growth_round_trips_to_forward_dcf():
    # Forward: at growth=0.10 the base case fair value is 29.5715.
    # Reverse: given that price, we should recover ~0.10.
    inputs = _base_inputs()
    forward_price = run_dcf(inputs).fair_value_per_share
    implied = implied_revenue_growth(inputs, forward_price)
    assert implied is not None
    assert implied == pytest.approx(0.10, abs=0.001)


def test_implied_growth_round_trips_when_growth_destroys_value():
    """With a negative operating margin, fair value *falls* as growth rises.

    Bisection must read that direction off the bracket rather than assume
    fair value increases with growth, or it converges on the wrong endpoint.
    """
    inputs = _base_inputs(operating_margin=-0.10)

    # Confirm the premise: this configuration is monotonically decreasing.
    assert run_dcf(_base_inputs(operating_margin=-0.10, revenue_growth=0.5)) \
        .fair_value_per_share < run_dcf(
            _base_inputs(operating_margin=-0.10, revenue_growth=0.0)
        ).fair_value_per_share

    target = run_dcf(_base_inputs(operating_margin=-0.10, revenue_growth=0.30)) \
        .fair_value_per_share
    implied = implied_revenue_growth(inputs, target)
    assert implied is not None
    assert implied == pytest.approx(0.30, abs=0.001)


def test_implied_growth_returns_none_when_price_is_flat_in_growth():
    """No sensitivity to growth means no implied growth rate to report."""
    # NOPAT + D&A - capex nets to zero per dollar of revenue, and no working
    # capital drag, so fair value is identical at every growth rate.
    inputs = _base_inputs(
        operating_margin=0.0, tax_rate=0.0, da_pct=0.05, capex_pct=0.05, wc_pct=0.0
    )
    flat_price = run_dcf(inputs).fair_value_per_share
    assert implied_revenue_growth(inputs, flat_price) is None
