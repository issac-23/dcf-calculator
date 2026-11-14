# Working in this repo

Notes for anyone — human or agent — writing code here.

## Testing

**Prefer end-to-end tests. They are the default, not the fallback.**

`tests/e2e/` holds eight journey tests, plus the snapshot test below. Between
them they cover every path a person can take through this app: a valuation
that works, the same one with the ticker typed sloppily, one that works but
produces an absurd number, a reverse solve with no answer, a company that is
the wrong shape for the model, data that isn't there (or that Yahoo's
company info won't answer, which the app works around from the balance sheet
and price history when the statements loaded), and a Treasury quote that is
dead. Every journey loads through
the real Load button. Each starts at a recorded Yahoo Finance payload and
ends at a rendered Streamlit page or a fair value per share, with nothing
stubbed in between except the network call itself.

When you add a feature, the question to answer first is which of those eight
journeys changes, or whether you have created a ninth. Adding one is a real
decision — this is close to the ceiling for a tool that asks one question.

**Every E2E run produces a reviewable artifact.**
`tests/e2e/__snapshots__/valuation_report.md` is regenerated on every run and
diffed against the committed copy. It is markdown, not a blob, because the
point is that a person reads it and notices Apple is suddenly worth $400. A
change to it is a change in behaviour, and the diff belongs in the pull
request.

To refresh it deliberately:

```bash
python tests/e2e/record.py          # re-record fixtures from the live API
E2E_UPDATE_SNAPSHOT=1 pytest tests/e2e
```

Then read the diff before committing. A re-recording that produces *no* diff
is the suspicious case, not the reassuring one.

**Never write unit tests after you write the code.**

A unit test written afterwards tests what the code does, which you already
know, and it passes on the first run — that is the tell. It has confirmed your
implementation matches itself. Almost every low-value test in a codebase got
there this way.

**If a system genuinely needs testing in isolation, write down the failure
modes first, then write the code.**

In a comment or the PR body, list the ways the thing can be wrong before you
implement it. Then implement it. Then write one test per listed failure mode.
If you cannot name a way it fails, you do not need a test — you need to
understand the problem better.

The valuation engine earns isolated tests under this rule, and its tests
should stay. `dcf/engine.py` is pure arithmetic with known failure modes:
terminal growth exceeding WACC, a bisection that assumes monotonicity it does
not have, division by zero shares. Those were written down and then tested.
`tests/test_engine.py::test_textbook_case_matches_hand_computed_answer` is the
load-bearing one — it checks the engine against a hand-computed textbook
answer, which is an oracle the E2E suite cannot provide, because the E2E suite
only knows what this code produces.

**Tests never touch the network, and a guard enforces it.** An autouse fixture
in `tests/conftest.py` (built in `tests/offline_guard.py`) blocks everything
except loopback and unix sockets. It patches Python's sockets and also
curl_cffi's `Session.request`, because yfinance goes through libcurl and never
touches the socket module. `dcf/data.py` swallows exceptions, so the guard
records every blocked attempt and fails the test at teardown naming the host,
whether or not the error surfaced. If a test goes red with "tried to reach the
network", something is reaching for Yahoo; fix that instead of loosening the
guard. `tests/e2e/record.py` is the sole exception and pytest never runs it.

Tests that really must talk to Yahoo carry `@pytest.mark.live` and are exempt.
Plain `pytest` skips them (`addopts = -m "not live"` in `pytest.ini`); a
command-line `-m` replaces that, so `pytest -m live` runs only the live ones.

A scheduled workflow, `.github/workflows/live.yml`, runs `pytest -m live` every
day against the real Yahoo from a GitHub runner, which is a datacenter IP like
Streamlit Cloud. It exists because Yahoo once stopped answering `.info` to
datacenter IPs while financials kept working, and the app was broken for weeks
behind a green suite. A red run there means Yahoo or the deploy broke, not the
code: check the live app at https://dcf-calculator-issac.streamlit.app before
touching anything. The failure message says which of `.info`, `.financials`,
`.cashflow` and `.balance_sheet` came back empty, and the test also fails when
the app had to use its balance-sheet fallback because `.info` did not answer.
GitHub pauses schedules on a repo with no activity for 60 days; start it by hand from the Actions tab (`workflow_dispatch`) if
that happens. Locally, `pytest -m live`.

Deploys: Streamlit Cloud redeploys on every push to main by rerunning `app.py`,
but it can keep the `dcf/` modules it already imported. A push where `app.py`
imports a new name from `dcf/` therefore crashed the live app with an
ImportError until it was rebooted (this happened on 2026-11-10 when
`describe_failure` was added). After merging a change like that, open
https://dcf-calculator-issac.streamlit.app and, if it shows an ImportError,
reboot it from Manage app, then Reboot.

## Code

- `dcf/` is pure Python with no Streamlit and no I/O beyond `dcf/data.py`.
  Keep it that way; it is why the model is testable at all.
- Logic that seeds or shapes the UI's inputs belongs in `dcf/assumptions.py`,
  not in `app.py`. If it lives in the Streamlit script, the only way to test it
  is to reimplement it in the test, and then the test agrees with itself
  instead of with the app.
- Missing data is `None`, never `0.0`. A zero is a measurement; a `None` is an
  absence. Conflating them is how a company with no reported interest expense
  silently acquires free debt.

## Documentation

`LIMITATIONS.md` records where the model's output is weakest, with measured
numbers. If a change alters one of those numbers, update it in the same pull
request. It is the most useful file in the repo and it goes stale fastest.
