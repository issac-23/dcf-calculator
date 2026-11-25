"""End-to-end journeys through the DCF calculator.

Each test below is a thing a person actually does with this app, run against
recorded Yahoo Finance responses from the real API. A journey starts at the
raw JSON payload and ends at either a rendered Streamlit page or a fair value
per share — no layer is stubbed in between.

There are deliberately few of them. A DCF has exactly one interesting
question ("what does this company come out worth, and do I believe it?"), and
the paths that question can take are: it works (once with a tidy ticker, once
typed the way people type it), it works but the answer is absurd, the reverse
solve has no answer, the company is the wrong shape for the model, the data
isn't there or can't be valued (an ADR whose statements are in another
currency than its price), or Yahoo's company info won't answer, which the app
works around when the statements loaded, or the Treasury quote behind
every WACC is dead. That is eight journeys, and they are these eight. The
ninth and last test pins the numbers.

The fixtures are real recordings, so these tests know real things — that
Apple's beta is near 1.1 and J&J's near 0.2, that Karuna stopped reporting
after the Bristol-Myers acquisition. Assertions are written against the
*behaviour* those facts produce rather than the facts themselves, so a
re-recording moves the snapshot without breaking the suite.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dcf.caveats import CaveatKind
from dcf.assumptions import ASSUMPTIONS, seed_inputs, seeded_wacc_pct
from dcf import market
from dcf.data import FALLBACK_RISK_FREE_RATE, DataFetchError, build_company_data
from dcf.valuation import run_valuation

from tests.e2e import replay, report

APP = str(Path(__file__).resolve().parents[2] / "app.py")
SNAPSHOT = Path(__file__).resolve().parent / "__snapshots__" / "valuation_report.md"


def build_company(symbol):
    return build_company_data(symbol, replay.load(symbol))


def exceptions(at: AppTest) -> str:
    return "; ".join(str(e.value) for e in at.exception)


def load_via_the_button(symbol) -> AppTest:
    """Type a ticker and click Load financials, the way a person does.

    Only the market data source is replaced, and it is replaced the way the app
    expects: tests/conftest.py sets DCF_MARKET_SOURCE to the recorded source.
    The button, the spinner, the session-state write and both error branches
    are the real ones. Injecting session_state directly would skip all of
    that, which is how the load handler went untested for the life of the app.
    """
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
    expected = f"${report.evaluate(symbol).valuation.result.fair_value_per_share:,.2f}"
    shown = next(m.value for m in at.metric if m.label == "Fair value / share")
    assert shown == expected, (
        f"the page opened on {shown} for {symbol} but the valuation report "
        f"says {expected}"
    )


_VERDICT_WORDING = {
    "undervalued": "undervalued",
    "overvalued": "overvalued",
    "fair": "Fairly valued",
}


def assert_page_shows_the_verdict(at: AppTest, verdict: str) -> None:
    """The fair-value tile's delta says what the valuation run concluded."""
    tile = next(m for m in at.metric if m.label == "Fair value / share")
    assert _VERDICT_WORDING[verdict] in tile.delta, (
        f"expected a {verdict} verdict on the page, got {tile.delta!r}"
    )


def page_text(at: AppTest) -> str:
    return " ".join(
        [m.value for m in at.markdown]
        + [cap.value for cap in at.caption]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
    )


# ----- the seven journeys ----------------------------------------------------


def test_a_stable_company_values_near_its_market_price_and_the_page_renders():
    """J&J: the case the model is actually good at.

    Low beta, steady growth, positive cash flow. This journey is the one that
    has to hold — if a mature consumer-health company comes out 10x off, the
    model is broken, not merely limited.
    """
    row = report.evaluate("JNJ")
    v = row.valuation

    assert row.error is None
    assert v.wacc is not None, "CAPM should derive cleanly from a full recording"
    wacc_pct = v.inputs.wacc * 100
    assert 4 < wacc_pct < 9, f"a defensive name should discount cheaply, got {wacc_pct:.2f}%"
    assert abs(v.upside) < 0.40, f"expected the model near market, got {v.upside:+.1%}"
    assert v.verdict == "undervalued", "J&J lands about 14% above its price"
    assert v.implied.rate is not None
    assert v.implied.reading == "in_line", "J&J's implied growth is close to its history"

    at = load_via_the_button("JNJ")
    assert_page_shows_the_reports_fair_value(at, "JNJ")
    assert_page_shows_the_verdict(at, "undervalued")
    text = page_text(at)
    for section in ("Valuation", "Free cash flow projection", "Reverse DCF",
                    "Sensitivity analysis", "Cost of capital"):
        assert section in text, f"{section!r} missing from the rendered page"
    assert len(at.dataframe) == 1, "projection table did not render"

    # The person then plays with the discount rate. The WACC slider opens on
    # CAPM, which depends on the equity risk premium, so it follows ERP until
    # they touch it, and keeps their number once they have.
    company = at.session_state["company"]

    def seeded_wacc(erp_pct: float) -> float:
        return seeded_wacc_pct(seed_inputs(company, report.E2E_RISK_FREE, erp_pct / 100).wacc)

    def slider(label: str):
        return next(s for s in at.slider if s.label == label)

    def wacc_caption() -> str:
        return next(c.value for c in at.caption if c.value.startswith("Weighted average"))

    slider("Equity risk premium").set_value(6.0).run()
    assert not at.exception, exceptions(at)
    assert slider("WACC (discount rate)").value == pytest.approx(seeded_wacc(6.0)), (
        "an untouched WACC slider should keep following the premium"
    )
    assert "matches the slider" in wacc_caption()

    slider("WACC (discount rate)").set_value(12.0).run()
    slider("Equity risk premium").set_value(4.5).run()
    assert not at.exception, exceptions(at)
    assert slider("WACC (discount rate)").value == 12.0, (
        "moving the premium threw away the WACC the person chose"
    )
    expected = run_valuation(
        company, report.E2E_RISK_FREE, {"erp": 4.5, "wacc": 12.0}
    ).result.fair_value_per_share
    shown = next(m.value for m in at.metric if m.label == "Fair value / share")
    assert shown == f"${expected:,.2f}", "the valuation did not use the 12% WACC"
    assert "overridden it to 12.00%" in wacc_caption()
    assert "matches the slider" not in wacc_caption()

    # A different company starts again from its own seeds, whatever was
    # dragged for the last one.
    at.text_input[0].set_value("AAPL")
    at.button[0].click().run()
    assert not at.exception, exceptions(at)
    company = at.session_state["company"]
    assert company.ticker == "AAPL"
    assert slider("WACC (discount rate)").value == pytest.approx(
        seeded_wacc(ASSUMPTIONS["erp"].fallback)
    )
    assert_page_shows_the_reports_fair_value(at, "AAPL")

    # A company that reports and trades in yen is valued the same way, and the
    # page says yen rather than dollars. Toyota's price, fair value and
    # enterprise value all carry the code, and a US ticker still carries "$".
    at.text_input[0].set_value("7203.T")
    at.button[0].click().run()
    assert not at.exception, exceptions(at)
    company = at.session_state["company"]
    assert company.currency == "JPY"
    expected = report.evaluate("7203.T").valuation.result.fair_value_per_share
    money = {m.label: m.value for m in at.metric}
    assert money["Fair value / share"] == f"JPY {expected:,.2f}"
    assert money["Current price"] == f"JPY {company.current_price:,.2f}"
    for label in ("Enterprise value", "Equity value", "Net debt"):
        assert money[label].startswith(("JPY ", "-JPY ")), (label, money[label])
    assert "$" not in " ".join(
        v for k, v in money.items() if k not in ("Implied upside",)
    ), money


def test_typing_a_ticker_and_clicking_load_gets_you_a_valuation():
    """The same journey as above, with the ticker typed the way people type it.

    Every journey loads through the button; this one keeps the handler's own
    behaviour in view -- whitespace and case are normalised, nothing errors,
    and the loaded company lands in session state.
    """
    at = load_via_the_button(" jnj ")

    assert not at.error, "; ".join(e.value for e in at.error)
    assert at.session_state["company"].ticker == "JNJ"
    assert "Valuation" in page_text(at)
    assert len(at.dataframe) == 1


def test_a_volatile_company_is_discounted_far_below_market_and_says_why():
    """Apple: the case the model is bad at, failing in the documented way.

    A high beta feeds a high discount rate, the model lands far under market,
    and the reverse DCF reports the growth the market must be assuming. The
    app is expected to show that gap rather than hide it — the number being
    wrong is fine, the number being unexplained is not.
    """
    row = report.evaluate("AAPL")
    v = row.valuation

    assert row.error is None
    assert v.wacc is not None
    assert v.inputs.wacc * 100 > 9, "a beta near 1 should not discount like a utility"
    assert v.upside < -0.40, f"expected a large discount to market, got {v.upside:+.1%}"
    assert v.verdict == "overvalued"

    # The whole point: the market's implied growth is far above delivered growth.
    assert v.implied.gap_vs_history > 0.10, (
        "the reverse DCF should show the market assuming much more growth than "
        "this company has delivered"
    )
    assert v.implied.reading == "optimistic"

    at = load_via_the_button("AAPL")
    assert_page_shows_the_reports_fair_value(at, "AAPL")
    assert_page_shows_the_verdict(at, "overvalued")
    assert "Reverse DCF" in page_text(at)
    assert "pp above" in page_text(at), "the page should say the market is the optimist"

    # Then the person drags the margin to the floor: the model stops giving a
    # number at all, instead of calling a negative value "overvalued".
    kinds = [c.kind for c in v.caveats]
    assert CaveatKind.NEGATIVE_EQUITY not in kinds, "AAPL opens on a positive value"
    next(s for s in at.slider if s.label == "Operating margin").set_value(-20.0).run()
    assert not at.exception, exceptions(at)

    tile = next(m for m in at.metric if m.label == "Fair value / share")
    assert tile.value == "Not meaningful"
    assert not tile.delta, f"no verdict delta on a meaningless value, got {tile.delta!r}"
    upside = next(m for m in at.metric if m.label == "Implied upside")
    assert upside.value == "—"
    warnings = " ".join(w.value for w in at.warning)
    assert "Equity value is zero or negative" in warnings
    assert "final projection year is negative" in warnings

    # The same company with no quote still reports the value as meaningless
    # rather than as a missing price.
    unpriced = run_valuation(
        replace(build_company("AAPL"), current_price=None),
        report.E2E_RISK_FREE,
        {"operating_margin": -20.0},
    )
    assert unpriced.verdict == "not_meaningful"
    assert unpriced.upside is None


def test_a_price_the_model_cannot_reach_reports_nothing_rather_than_guessing():
    """Tesla: the bisection has no root in its search range.

    This is the regression test for the reverse-DCF bug. The solver used to
    assume price rises monotonically with growth, which is false when margins
    are thin enough that growth destroys value; it would return a confident
    -50% — the bracket floor — instead of admitting defeat. It must return
    None here, and the page must still render.
    """
    row = report.evaluate("TSLA")
    v = row.valuation

    assert row.error is None
    assert v.implied is not None and v.implied.rate is None, (
        f"expected no solution, got {v.implied!r} — if the solver has "
        "started finding a root here, check it is a real one and re-record"
    )
    assert v.implied.reading is None, "no growth rate means nothing to compare with history"

    at = load_via_the_button("TSLA")
    assert_page_shows_the_reports_fair_value(at, "TSLA")
    assert_page_shows_the_verdict(at, "overvalued")
    assert "Valuation" in page_text(at)
    assert "between −50% and +100%" in page_text(at), "the page should name the range it searched"
    assert len(at.dataframe) == 1


def test_a_bank_is_flagged_as_the_wrong_shape_for_this_model():
    """JPMorgan: FCFF is not a meaningful number for a bank.

    The app does not refuse to value it — refusing would be worse, because
    the user would not learn why. It values it and says the tool is wrong.
    """
    bank = build_company("JPM")

    assert bank.is_dcf_inappropriate
    caveats = [c.kind for c in report.evaluate("JPM").valuation.caveats]
    assert caveats.count(CaveatKind.WRONG_SECTOR) == 1, (
        "an inappropriate sector must be reported once, not once per surface"
    )

    at = load_via_the_button("JPM")
    assert_page_shows_the_reports_fair_value(at, "JPM")
    assert report.evaluate("JPM").valuation.verdict == "fair", "JPM lands within 10% of its price"
    assert_page_shows_the_verdict(at, "fair")
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


@pytest.mark.parametrize(
    "symbol, needles",
    [
        pytest.param("KRTX", ["KRTX", "revenue"], id="delisted"),
        pytest.param("TSM", ["TSM", "TWD", "USD"], id="adr-in-another-currency"),
    ],
)
def test_data_that_cannot_be_valued_fails_with_a_message_a_human_can_act_on(
    symbol, needles
):
    """Karuna: acquired by Bristol-Myers in March 2024, no longer reporting.
    TSMC: its ADR trades in USD but its statements are in Taiwan dollars.

    Two failures share the first path -- a company that never had revenue, and
    one that stopped having it. Neither can be valued by a model whose first
    line compounds a revenue base. The ADR is the same shape from the other
    side: the data is there, but revenue in TWD against a price in USD would
    come out about 30x too high, and the tool does not convert. What matters
    is that the failure names the problem instead of surfacing a KeyError or a
    confident wrong number.
    """
    with pytest.raises(DataFetchError) as excinfo:
        build_company(symbol)

    message = str(excinfo.value)
    for needle in needles:
        assert needle.lower() in message.lower()

    # And a user who types the ticker and clicks Load sees that sentence as an
    # error, with the app still standing and still asking for a ticker.
    at = load_via_the_button(symbol)

    errors = " ".join(e.value for e in at.error).lower()
    for needle in needles:
        assert needle.lower() in errors
    assert at.session_state["company"] is None
    assert any("Enter a ticker" in i.value for i in at.info)

    # A refused ticker is reported as not valued and has no valuation row.
    assert report.evaluate(symbol).valuation is None


def everything_on_the_page(at: AppTest) -> str:
    """Every string any element on the page carries, not just the ones the
    helpers above know to read: markdown, captions, errors, metrics, tables."""
    parts = [str(e.value) for e in at.exception]

    def walk(block):
        for child in block.children.values():
            if hasattr(child, "children"):
                walk(child)
            else:
                parts.append(str(child.proto))

    walk(at.main)
    walk(at.sidebar)
    return " ".join(parts)


@pytest.mark.parametrize("balance_sheet", [True, False], ids=["fallback", "no-shares"])
@pytest.mark.parametrize(
    "info_mode, reason",
    [
        pytest.param("1", "RuntimeError: 401 Client Error", id="info-raised"),
        pytest.param(replay.INFO_EMPTY, "had no name, price or share count", id="info-near-empty"),
    ],
)
def test_yahoo_not_answering_for_company_info_keeps_the_valuation_or_says_so(
    monkeypatch, info_mode, reason, balance_sheet
):
    """On a blocked datacenter IP the statements load and `.info` is refused,
    either by raising or by handing back a near-empty dict.

    The old app reported that it could not find shares outstanding for AAPL,
    which sent people off to check a ticker that was fine. Now the valuation
    carries on: shares, debt and cash come from the balance sheet, the price
    from the last close, and a caveat says so (and that the sector check could
    not run). When the balance sheet has no share count either, the screen
    names the source instead of the ticker. Either way the crumb and cookie
    that ride along in the request URL never reach the page.
    """
    monkeypatch.setenv(replay.INFO_REFUSED_ENV_VAR, info_mode)
    if not balance_sheet:
        monkeypatch.setenv(replay.NO_BALANCE_SHEET_ENV_VAR, "1")

    at = load_via_the_button("AAPL")

    everything = everything_on_the_page(at)
    assert "AAPL" in everything, "the page walk should see the page"
    assert "abc123" not in everything and "xyz" not in everything

    if not balance_sheet:
        errors = " ".join(e.value for e in at.error)
        assert "Yahoo Finance didn't answer" in errors
        assert reason in errors
        assert "shares outstanding" not in errors
        assert at.session_state["company"] is None
        assert any("Enter a ticker" in i.value for i in at.info)
        return

    recorded = replay.load("AAPL")
    company = at.session_state["company"]
    assert company is not None, "a dead .info must not cost the valuation"
    assert not at.error, "; ".join(e.value for e in at.error)
    fair = next(m.value for m in at.metric if m.label == "Fair value / share")
    assert fair.startswith("$"), fair

    # Shares come from the balance sheet, the price from the last close.
    shares = recorded.balance_sheet.loc["Ordinary Shares Number"].dropna().iloc[0]
    last_close = recorded.history("5d")["Close"].iloc[-1]
    assert company.shares_outstanding == shares
    assert company.current_price == pytest.approx(last_close)
    shown_price = next(m.value for m in at.metric if m.label == "Current price")
    assert shown_price == f"${last_close:,.2f}"
    assert company.name == "AAPL" and company.sector == "" and company.beta is None
    # The fallback has no currency code, so it keeps the dollar sign.
    assert company.currency is None

    caveats = [c.kind for c in report_caveats(at)]
    assert caveats.count(CaveatKind.QUOTE_FALLBACK) == 1
    warnings = " ".join(w.value for w in at.warning)
    assert "balance sheet" in warnings and "price history" in warnings
    assert reason in warnings
    assert "sector" in warnings and "could not run" in warnings


def report_caveats(at: AppTest):
    """The caveats of the valuation the page is showing, for the loaded company."""
    from dcf.market import fetch_risk_free_rate, load_source
    from dcf.valuation import run_valuation

    risk_free, _ = fetch_risk_free_rate(load_source())
    return run_valuation(at.session_state["company"], risk_free).caveats


@pytest.mark.parametrize(
    "quote, is_live",
    [
        pytest.param("4.10,4.96", True, id="normal-quote"),
        pytest.param("error", False, id="dead-network"),
        pytest.param("nan,nan", False, id="all-nan"),
        pytest.param("empty", False, id="empty"),
        pytest.param("0.0", False, id="zero"),
        pytest.param("-1.5", False, id="negative"),
        pytest.param("15.0", False, id="at-the-ceiling"),
        pytest.param("518.0", False, id="units-changed"),
    ],
)
def test_a_dead_treasury_quote_costs_the_rate_and_not_the_valuation(
    monkeypatch, quote, is_live
):
    """The 10-year Treasury feeds every WACC, and it fails on its own.

    Runs the real fetch_risk_free_rate behind the real page, with only the
    recorded source's Treasury quote changed (E2E_TREASURY_CLOSE). The ways it
    goes wrong: a percent read as a decimal (^TNX quotes 4.96 for
    4.96%, and a missing /100 prices every company at nothing); a NaN or
    empty series that raises instead of falling back; a garbage quote passed
    through as a real rate; an exception that takes the valuation with it; or
    a stale fallback presented as today's yield. So the page must still
    value the company, and its caption must say which rate it used and
    whether that rate is live.
    """
    monkeypatch.setenv(replay.TREASURY_ENV_VAR, quote)

    at = load_via_the_button("JNJ")

    assert "Valuation" in page_text(at), "a bad Treasury quote must not cost the valuation"
    caption = next(c.value for c in at.caption if "10-year Treasury" in c.value)
    if is_live:
        assert "10-year Treasury at 4.96%" in caption
        assert "fallback" not in caption
    else:
        assert f"a {FALLBACK_RISK_FREE_RATE * 100:.2f}% fallback is in use" in caption


def test_a_dead_treasury_quote_is_not_remembered_once_the_quote_recovers(monkeypatch):
    """A fallback rate must never be cached, but a dead quote is not hammered.

    Failure modes: a dead quote that is cached pins the fallback (and every
    WACC built on it) for an hour after Treasury recovers; a live rate that is
    not cached is refetched on every rerun; a fallback that loses its caveat;
    a dead quote refetched on every rerun, so each slider drag waits on a
    failing request; a dead quote remembered past the retry window. The
    Streamlit cache is deliberately not cleared between runs, and time is moved
    by replacing the clock dcf.market reads, not by sleeping.
    """
    now = [1000.0]
    monkeypatch.setattr(market, "_clock", lambda: now[0])

    def caption(at):
        return next(c.value for c in at.caption if "10-year Treasury" in c.value)

    monkeypatch.setenv(replay.TREASURY_ENV_VAR, "error")
    fetches = replay.treasury_fetches
    at = load_via_the_button("JNJ")
    assert f"a {FALLBACK_RISK_FREE_RATE * 100:.2f}% fallback is in use" in caption(at)
    assert replay.treasury_fetches == fetches + 1

    # within the window a second dead run does not ask the source again
    now[0] += market._RETRY_DEAD_QUOTE_AFTER_S - 1
    at = load_via_the_button("JNJ")
    assert "fallback is in use" in caption(at)
    assert replay.treasury_fetches == fetches + 1

    # even if the quote has recovered, the window is honoured...
    monkeypatch.setenv(replay.TREASURY_ENV_VAR, "4.10,4.96")
    at = load_via_the_button("JNJ")
    assert "fallback is in use" in caption(at)
    assert replay.treasury_fetches == fetches + 1

    # ...and once it has passed, the source is retried and the rate is live
    now[0] += 2
    at = load_via_the_button("JNJ")
    assert "10-year Treasury at 4.96%" in caption(at)
    assert "fallback" not in caption(at)
    assert replay.treasury_fetches == fetches + 2

    # a live rate IS cached: a later dead quote does not displace it, and the
    # source is not asked again
    monkeypatch.setenv(replay.TREASURY_ENV_VAR, "error")
    at = load_via_the_button("JNJ")
    assert "10-year Treasury at 4.96%" in caption(at)
    assert replay.treasury_fetches == fetches + 2


# ----- the artifact ---------------------------------------------------------


def test_the_valuation_report_matches_the_committed_snapshot():
    """Regenerate the report and diff it against the committed copy.

    This is the repeatable artifact. The seven journeys above assert on
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
