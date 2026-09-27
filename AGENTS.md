# Working in this repo

Notes for anyone — human or agent — writing code here.

## Testing

**Prefer end-to-end tests. They are the default, not the fallback.**

`tests/e2e/` holds six journey tests. Between them they cover every path a
person can take through this app: a valuation that works, the same one driven
through the actual widgets, one that works but produces an absurd number, a
reverse solve with no answer, a company that is the wrong shape for the model,
and data that isn't there. Each starts at a recorded Yahoo Finance payload and
ends at a rendered Streamlit page or a fair value per share, with nothing
stubbed in between except the network call itself.

When you add a feature, the question to answer first is which of those six
journeys changes, or whether you have created a seventh. Adding one is a real
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

**Tests never touch the network.** `tests/e2e/record.py` is the sole exception
and pytest never runs it. If a test run gets slower for no reason, something
started reaching for Yahoo — that has happened before and it cost 30 seconds a
run while failing silently.

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
