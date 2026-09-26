# What this model can't do

A DCF is a chain of assumptions, and this one is no exception. This document
records where the output is weakest, with numbers rather than hedging.

Every figure below was produced by running the app's own defaults headlessly on
**2026-09-26**, with the 10-year Treasury at 5.18% (live from `^TNX`), a 5%
equity risk premium, 2.5% terminal growth, and a 5-year projection. Re-running
later will move the numbers; the conclusions have held across every run so far.

| Ticker | Beta | Derived WACC | Hist. growth | Market | Model | Gap | TV share of EV |
|---|---|---|---|---|---|---|---|
| JNJ  | 0.23 | 6.03%  | 5.6%   | $271.22 | $309.90 | +14.3% | 85.2% |
| PG   | 0.38 | 6.59%  | 2.0%   | $146.23 | $150.88 | +3.2%  | 82.1% |
| KO   | 0.34 | 6.49%  | 3.7%   | $87.81  | $62.24  | −29.1% | 82.9% |
| AAPL | 1.08 | 10.50% | 1.8%   | $341.07 | $89.39  | −73.8% | 68.4% |
| MSFT | 1.11 | 10.44% | 16.1%  | $516.17 | $228.61 | −55.7% | 74.0% |
| NVDA | 2.22 | 16.16% | 100.0% | $225.07 | $84.22  | −62.6% | 65.0% |
| TSLA | 1.84 | 14.28% | 5.2%   | $372.11 | $19.61  | −94.7% | 59.4% |

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
- **AAPL**: −73.8% → −48.5%. Worth 25.3 points. Growth is worth 0.2 points here
  — AAPL's 1.8% historical growth is already close to PG's 2.0%.

So MSFT's gap is almost entirely the discount rate. AAPL's is the discount rate
plus something the model cannot see at all, since its remaining 48-point gap
survives both swaps.

## 2. One point of discount rate is worth 11–23% of the answer

Holding everything else fixed and moving WACC up a single percentage point:

- JNJ: $310.14 → $238.48 (**−23.1%**)
- AAPL: $89.35 → $79.29 (**−11.3%**)

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
The model is therefore not valuing the company its own data describes, and
nothing in the UI says so.

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

## What this model is for

Not price targets. It is a tool for asking what a price implies: run the reverse
DCF, compare the implied growth against what the company has actually delivered,
and decide whether you believe the difference. The forward valuation is most
trustworthy for mature, low-beta, positive-FCF companies — which is exactly the
set where it happens to agree with the market.
