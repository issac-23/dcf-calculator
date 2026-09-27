"""End-to-end journeys through the DCF calculator.

Each test below is a thing a person actually does with this app, run against
recorded Yahoo Finance responses from the real API. A journey starts at the
raw JSON payload and ends at either a rendered Streamlit page or a fair value
per share — no layer is stubbed in between.

There are deliberately few of them. A DCF has exactly one interesting
question ("what does this company come out worth, and do I believe it?"), and
the paths that question can take are: it works, it works but the answer is
absurd, the reverse solve has no answer, the company is the wrong shape for
the model, or the data isn't there. That is five journeys, and they are these
five. The sixth test pins the numbers.

The fixtures are real recordings, so these tests know real things — that
Apple's beta is near 1.1 and J&J's near 0.2, that Karuna stopped reporting
after the Bristol-Myers acquisition. Assertions are written against the
*behaviour* those facts produce rather than the facts themselves, so a
re-recording moves the snapshot without breaking the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import dcf.data
from dcf.assumptions import seed_inputs
from dcf.data import DataFetchError, build_company_data
from dcf.engine import run_dcf
from dcf.wacc import DEFAULT_EQUITY_RISK_PREMIUM

from tests.e2e import replay, report

APP = str(Path(__file__).resolve().parents[2] / "app.py")
SNAPSHOT = Path(__file__).resolve().parent / "__snapshots__" / "valuation_report.md"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Pin the Treasury fetch. Nothing in this suite touches the network."""
    monkeypatch.setattr(
        dcf.data, "fetch_risk_free_rate", lambda: (report.E2E_RISK_FREE, True)
    )
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def company(symbol):
    return build_company_data(symbol, replay.load(symbol))


def render_app(symbol) -> AppTest:
    """Drive the real Streamlit script with this company already loaded."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["company"] = company(symbol)
    at.run()
    assert not at.exception, "; ".join(str(e.value) for e in at.exception)
    return at


def body(at: AppTest) -> str:
    return " ".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
    )


# ----- the five journeys ----------------------------------------------------


def test_a_stable_company_values_near_its_market_price_and_the_page_renders():
    """J&J: the case the model is actually good at.

    Low beta, steady growth, positive cash flow. This journey is the one that
    has to hold — if a mature consumer-health company comes out 10x off, the
    model is broken, not merely limited.
    """
    row = report.evaluate("JNJ")

    assert row.error is None
    assert row.wacc_derived, "CAPM should derive cleanly from a full recording"
    assert 4 < row.wacc_pct < 9, f"a defensive name should discount cheaply, got {row.wacc_pct:.2f}%"
    assert abs(row.gap_pct) < 40, f"expected the model near market, got {row.gap_pct:+.1f}%"
    assert row.implied_growth is not None

    at = render_app("JNJ")
    text = body(at)
    for section in ("Valuation", "Free cash flow projection", "Reverse DCF",
                    "Sensitivity analysis", "Cost of capital"):
        assert section in text, f"{section!r} missing from the rendered page"
    assert len(at.dataframe) == 1, "projection table did not render"


def test_a_volatile_company_is_discounted_far_below_market_and_says_why():
    """Apple: the case the model is bad at, failing in the documented way.

    A high beta feeds a high discount rate, the model lands far under market,
    and the reverse DCF reports the growth the market must be assuming. The
    app is expected to show that gap rather than hide it — the number being
    wrong is fine, the number being unexplained is not.
    """
    row = report.evaluate("AAPL")

    assert row.error is None
    assert row.wacc_derived
    assert row.wacc_pct > 9, "a beta near 1 should not discount like a utility"
    assert row.gap_pct < -40, f"expected a large discount to market, got {row.gap_pct:+.1f}%"

    # The whole point: the market's implied growth is far above delivered growth.
    assert row.implied_growth > (row.company.historical_revenue_growth + 0.10), (
        "the reverse DCF should show the market assuming much more growth than "
        "this company has delivered"
    )

    at = render_app("AAPL")
    assert "Reverse DCF" in body(at)


def test_a_price_the_model_cannot_reach_reports_nothing_rather_than_guessing():
    """Tesla: the bisection has no root in its search range.

    This is the regression test for the reverse-DCF bug. The solver used to
    assume price rises monotonically with growth, which is false when margins
    are thin enough that growth destroys value; it would return a confident
    -50% — the bracket floor — instead of admitting defeat. It must return
    None here, and the page must still render.
    """
    row = report.evaluate("TSLA")

    assert row.error is None
    assert row.implied_growth is None, (
        f"expected no solution, got {row.implied_growth!r} — if the solver has "
        "started finding a root here, check it is a real one and re-record"
    )

    at = render_app("TSLA")
    assert "Valuation" in body(at)
    assert len(at.dataframe) == 1


def test_a_bank_is_flagged_as_the_wrong_shape_for_this_model():
    """JPMorgan: FCFF is not a meaningful number for a bank.

    The app does not refuse to value it — refusing would be worse, because
    the user would not learn why. It values it and says the tool is wrong.
    """
    c = company("JPM")

    assert c.is_dcf_inappropriate
    assert c.warnings, "an inappropriate sector must produce a visible warning"

    at = render_app("JPM")
    warnings = " ".join(w.value for w in at.warning).lower()
    assert "financial" in warnings or "dcf" in warnings
    assert "Valuation" in body(at), "it should still render a valuation"


def test_a_delisted_company_fails_with_a_message_a_human_can_act_on():
    """Karuna: acquired by Bristol-Myers in March 2024, no longer reporting.

    Two failures share this path — a company that never had revenue, and one
    that stopped having it. Neither can be valued by a model whose first line
    compounds a revenue base. What matters is that the failure names the
    missing thing instead of surfacing a KeyError.
    """
    with pytest.raises(DataFetchError) as excinfo:
        company("KRTX")

    message = str(excinfo.value)
    assert "KRTX" in message
    assert "revenue" in message.lower()

    # And the app surfaces it as an error rather than dying.
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Enter a ticker" in i.value for i in at.info)


# ----- the artifact ---------------------------------------------------------


def test_the_valuation_report_matches_the_committed_snapshot():
    """Regenerate the report and diff it against the committed copy.

    This is the repeatable artifact. The five journeys above assert on
    behaviour and tolerate drift; this one pins every number, so a change in
    any layer shows up as a readable diff in `__snapshots__`.

    Refresh with: E2E_UPDATE_SNAPSHOT=1 pytest tests/e2e
    """
    generated = report.render(report.evaluate_all())

    if os.environ.get("E2E_UPDATE_SNAPSHOT"):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(generated, encoding="utf-8")
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")

    assert SNAPSHOT.exists(), (
        f"no snapshot at {SNAPSHOT}. Create it with "
        "E2E_UPDATE_SNAPSHOT=1 pytest tests/e2e"
    )

    committed = SNAPSHOT.read_text(encoding="utf-8")
    if committed != generated:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                committed.splitlines(),
                generated.splitlines(),
                fromfile="committed",
                tofile="generated",
                lineterm="",
            )
        )
        pytest.fail(
            "The valuation report changed.\n\n"
            "If this is intended, rerun with E2E_UPDATE_SNAPSHOT=1 and commit "
            f"the new file.\n\n{diff}"
        )
