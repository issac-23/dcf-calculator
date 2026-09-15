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

from dcf.data import CompanyData

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


def test_landing_state_renders():
    at = _run()
    assert any("Load financials" in b.label for b in at.button)
    assert any("Enter a ticker" in i.value for i in at.info)


def test_full_render_reaches_every_section():
    at = _run(_fake_company())

    # st.stop() sits between the landing state and everything below, so these
    # assertions are what prove the rest of the script actually executed.
    assert len(at.dataframe) == 1, "projection table did not render"
    assert len(at.metric) >= 12, f"expected the metric rows, got {len(at.metric)}"

    markdown = " ".join(m.value for m in at.markdown)
    for section in ("Valuation", "Free cash flow projection", "Reverse DCF",
                    "Sensitivity analysis"):
        assert section in markdown, f"{section!r} section missing"


def test_negative_margin_company_renders_its_warning():
    """The unprofitable path has its own branches; make sure they render too."""
    at = _run(_fake_company(
        historical_operating_margin=-0.10,
        warnings=["Historical operating margin is negative."],
    ))
    assert any("negative" in w.value.lower() for w in at.warning)


def test_missing_price_skips_reverse_dcf_without_erroring():
    """Reverse DCF needs a market price to solve against; absent one it is
    skipped rather than fed a zero."""
    at = _run(_fake_company(current_price=0.0))
    markdown = " ".join(m.value for m in at.markdown)
    assert "Reverse DCF" not in markdown
    assert "Valuation" in markdown
