# What this model can't do

A DCF is a chain of assumptions, and this one is no exception. This document
records where the output is weakest, with numbers rather than hedging.

Every figure below was produced by running the app's own defaults headlessly,
with the 10-year Treasury at 5.18% (`^TNX` on the recording day), a 5% equity
risk premium, 2.5% terminal growth, and a 5-year projection. Re-running later
will move the numbers; the conclusions have held across every run so far.

The JNJ, AAPL and TSLA figures come from the recorded Yahoo fixtures in
`tests/e2e/fixtures` (recorded 2026-09-27) and match
`tests/e2e/__snapshots__/valuation_report.md`, so anyone can reproduce them.
PG, KO, MSFT and NVDA have no fixture; their figures are from the original live
run on 2026-09-26 and cannot be regenerated from the repo.

| Ticker | Beta | Derived WACC | Hist. growth | Market | Model | Gap | TV share of EV |
|---|---|---|---|---|---|---|---|
| JNJ  | 0.23 | 6.03%  | 5.6%   | $271.22 | $310.25 | +14.4% | 85.2% |
| PG   | 0.38 | 6.59%  | 2.0%   | $146.23 | $150.88 | +3.2%  | 82.1% |
| KO   | 0.34 | 6.49%  | 3.7%   | $87.81  | $62.24  | −29.1% | 82.9% |
| AAPL | 1.08 | 10.49% | 1.8%   | $341.07 | $89.43  | −73.8% | 68.4% |
| MSFT | 1.11 | 10.44% | 16.1%  | $516.17 | $228.61 | −55.7% | 74.0% |
| NVDA | 2.22 | 16.16% | 100.0% | $225.07 | $84.22  | −62.6% | 65.0% |
| TSLA | 1.84 | 14.27% | 5.2%   | $372.11 | $19.61  | −94.7% | 59.4% |

The ordering is the finding. The three low-beta names land within 30% of the
market; every high-beta name is 55–95% below it. That is not a coincidence, and
it is not really a statement about the businesses.

## 1. Beta drives the output more than the business does

The discount rate comes from CAPM: `Re = risk_free + beta × ERP`. Beta is
measured from how much the stock price has historically swung relative to the
market — it is a fact about the shareholders, not the company. Nothing about
competitive position, patent cliffs, or customer concentration reaches the
model.

Swapping PG's 6.59% WACC into the high-gap names, changing nothing else:

- **MSFT**: −55.7% → −9.9%. The discount rate accounts for 45.8 of the 55.7
  points. Swapping PG's 2.0% growth in *instead* makes MSFT worse (−75.7%),
  because MSFT's real 16.1% growth was helping.
- **AAPL**: −73.8% → −48.4%. Worth 25.4 points. Growth is worth 0.2 points here
  — AAPL's 1.8% historical growth is already close to PG's 2.0%.

So MSFT's gap is almost entirely the discount rate. AAPL's is the discount rate
plus something the model cannot see at all, since its remaining 48-point gap
survives both swaps.

## 2. One point of discount rate is worth 11–23% of the answer

Holding everything else fixed and moving WACC up a single percentage point:

- JNJ: $310.25 → $238.55 (**−23.1%**)
- AAPL: $89.43 → $79.36 (**−11.3%**)

Moving terminal growth from 2.5% to 3.5% is comparably violent: JNJ +36.2%,
AAPL +10.7%. Low-WACC companies are the most sensitive, because the
`WACC − g` denominator in the terminal value is small to begin with — JNJ's is
3.5 points wide, so a one-point move changes it by nearly a third.

This is why the app seeds the WACC slider from CAPM rather than replacing the
slider with it. The derived number is a starting point, not an answer.

## 3. Most of the answer is the terminal value

The terminal value is 59–85% of enterprise value across every ticker above, and
over 80% for all three defensive names. The explicit 5-year projection — the
part with actual revenue, margin, and capex modelling in it — is a minority of
the result.

That means the model is mostly a Gordon Growth formula with a DCF attached. The
five years of projections are doing less work than they appear to be doing.

## 4. It cannot value a company without a revenue history

`KRTX` (Karuna Therapeutics) raises `DataFetchError: Could not find revenue data
for 'KRTX'`. Two separate problems are hiding behind that one message:

- Karuna was acquired by Bristol-Myers Squibb in March 2024, so it no longer
  trades and Yahoo no longer serves its financials.
- Even with the data, the model could not have valued it. Karuna was
  pre-revenue; a model whose first line is `Revenue_t = Revenue_base × (1 + g)^t`
  has nothing to compound.

The same applies to banks and insurers, where FCFF is not the right cash flow
measure, and to any company whose value is a single binary outcome — a drug
trial, a lawsuit, a merger vote.

## 5. Historical averages are a weak forecast, and the sliders clip them

Defaults are 4-year historical averages, clamped to each slider's range. NVDA's
historical revenue growth is **100.0%**, which the growth slider clamps to 30%.
The model is therefore not valuing the company its own data describes. The app
now says so: the sidebar names every input whose default was clamped. (NVDA has
no fixture; the recorded case is JPM, whose working-capital history clamps.)

More generally, "the last four years" is a bad estimator for the next five for
anything cyclical, anything post-acquisition, or anything that just changed
business model. The reverse DCF exists partly to make this visible: it reports
the growth rate the market is implying, so the gap against history is explicit.
For AAPL that gap is 36.2% implied vs 1.8% delivered — the market is not
extrapolating history, and neither should the user.

One honest success here: for TSLA the reverse DCF returns `None` rather than a
number. No growth rate in the −50% to +100% search bracket reproduces the
market price. An earlier version of `implied_revenue_growth` assumed price rises
monotonically with growth and returned a confident −50% in cases like this; it
now reads the slope off the bracket endpoints and gives up when the price is
genuinely unreachable.

## 6. The working-capital default is a cash-flow line read as a balance

`dcf/data.py` seeds the "Working capital / revenue" slider from the cash-flow
statement's `Change In Working Capital` divided by revenue. That is a *flow*: how
much cash working capital absorbed or released in a year. `dcf/engine.py` treats
the same number as a *balance*: `prev_wc = revenue_base × wc_pct`, with
ΔWC the change in that balance, and the slider's help text says "Working capital
balance". Two things follow, both confirmed against the recorded fixtures:

- **It is the wrong quantity.** A balance of `w` percent of revenue growing at
  `g` absorbs about `w × g` of revenue in cash. A yearly flow of that size is a
  different number, so the default describes nothing the engine uses.
- **The sign is inverted.** yfinance reports cash impact, so a *negative*
  `Change In Working Capital` means working capital grew and cash was consumed.
  JNJ's 2025 line is −$12.7B alongside negative inventory (−$1.5B) and receivable
  (−$1.8B) changes. The engine reads a negative `wc_pct` as a balance that
  releases cash as revenue grows.

Seeded defaults from the recordings: JNJ −3.4%, AAPL −1.6%, TSLA −1.5%, and
JPM −49.6%, which clamps to the slider's −10%. Year to year the JNJ ratio swings
from −13.5% to +2.9%, which is a flow's behaviour, not a balance's.

The effect on the answer is small for ordinary companies because the engine only
charges `wc_pct × Δrevenue`. Setting the default to zero moves JNJ −0.8%, AAPL
−0.1%, TSLA −0.9%; flipping its sign moves them −1.5%, −0.2% and −1.8%. It is
large only for JPM (−6.8% at zero, −13.7% flipped), where the cash-flow lines are
not meaningful for a bank anyway.

This is not fixed. A correct default needs the balance sheet (current assets less
current liabilities, over revenue), which neither `build_company_data` nor the
recorded fixtures contain, and re-recording is out of bounds for a review fix.
Until then treat the slider as a plain assumption and leave it near zero.

## What this model is for

Not price targets. It is a tool for asking what a price implies: run the reverse
DCF, compare the implied growth against what the company has actually delivered,
and decide whether you believe the difference. The forward valuation is most
trustworthy for mature, low-beta, positive-FCF companies — which is exactly the
set where it happens to agree with the market.
