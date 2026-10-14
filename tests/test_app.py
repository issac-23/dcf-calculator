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

from streamlit.testing.v1 import AppTest

from dcf.data import CompanyData, build_company_data

from tests.e2e import replay

# AppTest resolves a relative path against the file that calls from_file(),
# not the working directory, so "app.py" would look for tests/app.py. Build an
# absolute path instead -- that holds regardless of where pytest is invoked.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


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


def test_unreported_debt_is_not_silently_an_all_equity_wacc():
    """Yahoo omitting totalDebt must not turn into a confident cost of equity.

    Driven from the recorded J&J payload with that one field removed, so it
    goes through the real data layer. Ways it goes wrong: the WACC is derived
    as if debt were zero and shown as "Derived"; or the page raises; or the
    user is told nothing.
    """
    tk = replay.load("JNJ")
    tk.info = {k: v for k, v in tk.info.items() if k != "totalDebt"}
    at = _run(build_company_data("JNJ", tk))

    shown = " ".join(
        [i.value for i in at.info] + [c.value for c in at.caption] + [w.value for w in at.warning]
    )
    assert "total debt" in shown.lower()
    assert "Derived:" not in shown
    wacc = next(s for s in at.slider if s.label.startswith("WACC"))
    assert wacc.value == 9.0, "should fall back to the generic default, not cost of equity"


def test_terminal_growth_at_or_above_wacc_shows_an_error_not_a_traceback():
    """The sliders allow it: WACC bottoms out at 4% and terminal growth tops
    out at 5%. The engine refuses the pair and the page has to say so."""
    at = _run(_fake_company())
    next(s for s in at.slider if s.label.startswith("WACC")).set_value(4.0)
    next(s for s in at.slider if s.label == "Terminal growth rate").set_value(4.5)
    at.run()

    assert not at.exception, "; ".join(str(e.value) for e in at.exception)
    assert any("strictly less than wacc" in e.value for e in at.error)
    assert "Valuation" not in " ".join(m.value for m in at.markdown)
