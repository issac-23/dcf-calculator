"""Tests for the DCF engine.

The first test is the most important: a textbook DCF problem with a known
hand-computed answer. If this ever fails, the engine has drifted.
"""

import math

import pytest

from dcf import DCFInputs, run_dcf
from dcf.engine import sensitivity_grid


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


def test_higher_wacc_decreases_fair_value():
    low = run_dcf(_base_inputs(wacc=0.08)).fair_value_per_share
    high = run_dcf(_base_inputs(wacc=0.12)).fair_value_per_share
    assert low > high


def test_higher_revenue_growth_increases_fair_value():
    low = run_dcf(_base_inputs(revenue_growth=0.05)).fair_value_per_share
    high = run_dcf(_base_inputs(revenue_growth=0.15)).fair_value_per_share
    assert high > low


def test_higher_terminal_growth_increases_fair_value():
    low = run_dcf(_base_inputs(terminal_growth=0.01)).fair_value_per_share
    high = run_dcf(_base_inputs(terminal_growth=0.04)).fair_value_per_share
    assert high > low


def test_higher_operating_margin_increases_fair_value():
    low = run_dcf(_base_inputs(operating_margin=0.10)).fair_value_per_share
    high = run_dcf(_base_inputs(operating_margin=0.30)).fair_value_per_share
    assert high > low


def test_higher_tax_rate_decreases_fair_value():
    low_tax = run_dcf(_base_inputs(tax_rate=0.15)).fair_value_per_share
    high_tax = run_dcf(_base_inputs(tax_rate=0.35)).fair_value_per_share
    assert low_tax > high_tax


def test_more_net_debt_decreases_equity_value():
    no_debt = run_dcf(_base_inputs(net_debt=0.0)).fair_value_per_share
    with_debt = run_dcf(_base_inputs(net_debt=500.0)).fair_value_per_share
    assert no_debt > with_debt
    # Specifically: 500 of net debt across 100 shares = $5/share
    assert no_debt - with_debt == pytest.approx(5.0, abs=0.001)


def test_more_shares_outstanding_decreases_per_share_value():
    fewer = run_dcf(_base_inputs(shares_outstanding=100.0)).fair_value_per_share
    more = run_dcf(_base_inputs(shares_outstanding=200.0)).fair_value_per_share
    assert fewer == pytest.approx(more * 2, abs=0.001)


def test_pv_factors_decrease_monotonically():
    result = run_dcf(_base_inputs())
    factors = [p.discount_factor for p in result.projections]
    assert factors == sorted(factors, reverse=True)


def test_revenue_compounds_correctly():
    result = run_dcf(_base_inputs(revenue_growth=0.10, revenue_base=1000.0))
    expected = [1000.0 * 1.1 ** t for t in range(1, 6)]
    actual = [p.revenue for p in result.projections]
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=0.001)


# ----- Edge / validation tests ---------------------------------------------


def test_terminal_growth_above_wacc_raises():
    with pytest.raises(ValueError, match="terminal_growth"):
        run_dcf(_base_inputs(terminal_growth=0.05, wacc=0.04))


def test_terminal_growth_equal_to_wacc_raises():
    with pytest.raises(ValueError, match="terminal_growth"):
        run_dcf(_base_inputs(terminal_growth=0.10, wacc=0.10))


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


def test_negative_growth_company_still_values():
    # Declining business — should still produce a positive (though lower) value.
    result = run_dcf(_base_inputs(revenue_growth=-0.05, terminal_growth=0.0))
    assert result.fair_value_per_share > 0
    assert math.isfinite(result.fair_value_per_share)


def test_unprofitable_company_can_have_negative_value():
    # Negative operating margin: this stress-tests the "Tesla 2018" case.
    result = run_dcf(_base_inputs(operating_margin=-0.10))
    assert math.isfinite(result.fair_value_per_share)
    # With negative NOPAT, FCF is negative => fair value should be negative
    assert result.fair_value_per_share < 0


def test_projection_length_respected():
    short = run_dcf(_base_inputs(projection_years=3))
    long = run_dcf(_base_inputs(projection_years=10))
    assert len(short.projections) == 3
    assert len(long.projections) == 10


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
