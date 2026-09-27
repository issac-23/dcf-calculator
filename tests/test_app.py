"""Render the Streamlit app headlessly and assert it raises nothing.

These are deliberately shallow. They do not check numbers -- test_engine.py
owns the math -- they check that every Streamlit call in app.py is one this
version of Streamlit still accepts.

That gap is not hypothetical. ``use_container_width`` was deprecated with a
stated removal, and Streamlit Cloud installs whatever is current when it
builds rather than what is pinned on a laptop. An app can pass every unit
test, run fine locally, and break on deploy the day the removal lands. The
engine tests cannot see that. These can.

The landing-state test is not enough on its own: app.py calls st.stop() when
no company is loaded, so it returns before ever reaching the projection
table or either chart. The full-render test injects a company into session
state to get past that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import dcf.data
from dcf.data import CompanyData

# AppTest resolves a relative path against the file that calls from_file(),
# not the working directory, so "app.py" would look for tests/app.py. Build an
# absolute path instead -- that holds regardless of where pytest is invoked.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def offline_risk_free(monkeypatch):
    """Pin the risk-free rate so these tests stay offline and deterministic.

    app.py fetches the 10-year Treasury for the CAPM derivation. That call
    swallows its own errors and falls back, so it never fails the suite -- it
    just quietly reaches for the network and waits. Left alone it took the run
    from 7s to 37s. Patching the source keeps the whole suite hermetic, which
    is a property test_data.py's docstring already claims.
    """
    monkeypatch.setattr(dcf.data, "fetch_risk_free_rate", lambda: (0.042, True))
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _fake_company(**overrides) -> CompanyData:
    defaults = dict(
        ticker="TEST",
        name="Test Corp",
        sector="Technology",
        industry="Software",
        current_price=42.0,
        revenue_base=1e10,
        shares_outstanding=1e9,
        net_debt=5e8,
        historical_revenue_growth=0.08,
        historical_operating_margin=0.22,
        historical_tax_rate=0.19,
        historical_capex_pct=0.04,
        historical_da_pct=0.05,
        historical_wc_pct=0.02,
        # Enough for the CAPM derivation to succeed, so the WACC panel is on
        # the rendered path rather than skipped.
        beta=1.1,
        market_cap=4.2e10,
        total_debt=5e8,
        interest_expense=2.5e7,
        is_dcf_inappropriate=False,
        warnings=[],
    )
    defaults.update(overrides)
    return CompanyData(**defaults)


def _run(company: CompanyData | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    if company is not None:
        at.session_state["company"] = company
    at.run()
    assert not at.exception, "; ".join(str(e.value) for e in at.exception)
    return at


def test_missing_price_skips_reverse_dcf_without_erroring():
    """Reverse DCF needs a market price to solve against; absent one it is
    skipped rather than fed a zero."""
    at = _run(_fake_company(current_price=0.0))
    markdown = " ".join(m.value for m in at.markdown)
    assert "Reverse DCF" not in markdown
    assert "Valuation" in markdown


# ----- cost of capital panel ------------------------------------------------


def test_missing_beta_explains_itself_instead_of_erroring():
    """A ticker with no beta should lose the derivation, not the valuation."""
    at = _run(_fake_company(beta=None))
    body = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    assert "beta" in body.lower()
    # The valuation still has to render.
    assert len(at.dataframe) == 1
    assert "Valuation" in " ".join(m.value for m in at.markdown)


def test_debt_without_interest_expense_warns_about_understating_wacc():
    at = _run(_fake_company(interest_expense=None))
    warnings = " ".join(w.value for w in at.warning)
    assert "understates" in warnings.lower() or "interest expense" in warnings.lower()
