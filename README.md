# DCF Calculator

[![tests](https://github.com/issac-23/dcf-calculator/actions/workflows/tests.yml/badge.svg)](https://github.com/issac-23/dcf-calculator/actions/workflows/tests.yml)

**[Try it →](https://dcf-calculator-issac.streamlit.app)**

Type in a ticker. It pulls the company's real financials, projects their cash
flow out five years, discounts it back to today, and tells you what a share
looks like it's worth.

Then it does the harder thing: it takes the price the market is actually
charging and works backwards to the growth rate that would justify it. Apple
needs 36% revenue growth a year to make sense at today's price. It has
delivered 1.8%. That gap is the whole point — not the fair value number, the
gap.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests need one extra package:

```bash
pip install -r requirements-dev.txt
pytest
```

They never touch the network. Real Yahoo responses are recorded in
`tests/e2e/fixtures/` and replayed, so the suite runs the same way in a tunnel
as it does on your couch.

## How it works

Five years of projected free cash flow, plus a terminal value for everything
after that:

```
FCFF  = NOPAT + D&A - Capex - change in working capital
EV    = each year's FCFF discounted back, + terminal value discounted back
TV    = next year's FCFF / (WACC - terminal growth)
Price = (EV - net debt) / shares outstanding
```

The discount rate comes from CAPM, using the company's own beta and capital
structure rather than a round number someone picked.

All the math lives in `dcf/engine.py` — pure Python, no network, no Streamlit.
It's checked against a textbook problem worked by hand, so the tests can tell
you the answer is *wrong*, not just that it changed.

## What it can't do

Sort the tickers by beta and you've sorted them by how wrong the model is.
Steady companies land near their market price; volatile ones come out 55–95%
below. That's CAPM showing through, not a fact about the businesses.

[LIMITATIONS.md](LIMITATIONS.md) has the measurements, including the companies
this model can't value at all. Worth reading before you trust a number it
gives you.
