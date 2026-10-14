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
from dcf.data import DataFetchError, build_company_data

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


def build_company(symbol):
    return build_company_data(symbol, replay.load(symbol))


def exceptions(at: AppTest) -> str:
    return "; ".join(str(e.value) for e in at.exception)


def load_via_the_button(monkeypatch, symbol) -> AppTest:
    """Type a ticker and click Load financials, the way a person does.

    Only the network call itself is replaced — the button, the spinner, the
    session-state write and both error branches are the real ones. Injecting
    session_state directly would skip all of that, which is how the load
    handler went untested for the life of the app.
    """
    def fake_fetch(ticker: str):
        return build_company(ticker.strip().upper())

    monkeypatch.setattr(dcf.data, "fetch_company_data", fake_fetch)
    st.cache_data.clear()

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input[0].set_value(symbol)
    at.button[0].click().run()
    assert not at.exception, exceptions(at)
    return at


def assert_page_shows_the_reports_fair_value(at: AppTest, symbol: str) -> None:
    """The page's opening valuation is the one the snapshot pins.

    The snapshot is built by seed_inputs -> run_dcf; the page builds its
    inputs from slider values. If the two ever seed differently the snapshot
    stops describing what a person sees, and nothing else would notice.
    """
    expected = f"${report.evaluate(symbol).model_price:,.2f}"
    shown = next(m.value for m in at.metric if m.label == "Fair value / share")
    assert shown == expected, (
        f"the page opened on {shown} for {symbol} but the valuation report "
        f"says {expected}"
    )


def page_text(at: AppTest) -> str:
    return " ".join(
        [m.value for m in at.markdown]
        + [cap.value for cap in at.caption]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
    )


# ----- the five journeys ----------------------------------------------------


def test_a_stable_company_values_near_its_market_price_and_the_page_renders(monkeypatch):
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

    at = load_via_the_button(monkeypatch, "JNJ")
    assert_page_shows_the_reports_fair_value(at, "JNJ")
    text = page_text(at)
    for section in ("Valuation", "Free cash flow projection", "Reverse DCF",
                    "Sensitivity analysis", "Cost of capital"):
        assert section in text, f"{section!r} missing from the rendered page"
    assert len(at.dataframe) == 1, "projection table did not render"


def test_typing_a_ticker_and_clicking_load_gets_you_a_valuation(monkeypatch):
    """The same journey as above, with the ticker typed the way people type it.

    Every journey loads through the button; this one keeps the handler's own
    behaviour in view -- whitespace and case are normalised, nothing errors,
    and the loaded company lands in session state.
    """
    at = load_via_the_button(monkeypatch, " jnj ")

    assert not at.error, "; ".join(e.value for e in at.error)
    assert at.session_state["company"].ticker == "JNJ"
    assert "Valuation" in page_text(at)
    assert len(at.dataframe) == 1


def test_a_volatile_company_is_discounted_far_below_market_and_says_why(monkeypatch):
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

    at = load_via_the_button(monkeypatch, "AAPL")
    assert_page_shows_the_reports_fair_value(at, "AAPL")
    assert "Reverse DCF" in page_text(at)


def test_a_price_the_model_cannot_reach_reports_nothing_rather_than_guessing(monkeypatch):
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

    at = load_via_the_button(monkeypatch, "TSLA")
    assert_page_shows_the_reports_fair_value(at, "TSLA")
    assert "Valuation" in page_text(at)
    assert len(at.dataframe) == 1


def test_a_bank_is_flagged_as_the_wrong_shape_for_this_model(monkeypatch):
    """JPMorgan: FCFF is not a meaningful number for a bank.

    The app does not refuse to value it — refusing would be worse, because
    the user would not learn why. It values it and says the tool is wrong.
    """
    bank = build_company("JPM")

    assert bank.is_dcf_inappropriate
    assert bank.warnings, "an inappropriate sector must produce a visible warning"

    at = load_via_the_button(monkeypatch, "JPM")
    assert_page_shows_the_reports_fair_value(at, "JPM")
    warnings = " ".join(w.value for w in at.warning).lower()
    assert "financial" in warnings or "dcf" in warnings
    assert "Valuation" in page_text(at), "it should still render a valuation"

    # JPM's recorded working-capital history sits far outside the slider's
    # range, so the default is clamped. The sidebar has to say which input.
    notices = " ".join(i.value for i in at.sidebar.info)
    assert "Working capital / revenue" in notices, (
        "a clamped default must be named in the sidebar, not applied silently"
    )
    assert "Operating margin" not in notices, "only clamped inputs should be named"


def test_a_delisted_company_fails_with_a_message_a_human_can_act_on(monkeypatch):
    """Karuna: acquired by Bristol-Myers in March 2024, no longer reporting.

    Two failures share this path — a company that never had revenue, and one
    that stopped having it. Neither can be valued by a model whose first line
    compounds a revenue base. What matters is that the failure names the
    missing thing instead of surfacing a KeyError.
    """
    with pytest.raises(DataFetchError) as excinfo:
        build_company("KRTX")

    message = str(excinfo.value)
    assert "KRTX" in message
    assert "revenue" in message.lower()

    # And a user who types KRTX and clicks Load sees that sentence as an
    # error, with the app still standing and still asking for a ticker.
    at = load_via_the_button(monkeypatch, "KRTX")

    errors = " ".join(e.value for e in at.error)
    assert "KRTX" in errors and "revenue" in errors.lower()
    assert at.session_state["company"] is None
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
