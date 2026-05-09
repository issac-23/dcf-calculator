# DCF Calculator

A discounted cash flow valuation tool for public equities. Pulls real
financials from Yahoo Finance, runs an intermediate two-stage DCF, and reports
fair value per share alongside a full year-by-year projection and a WACC ×
terminal-growth sensitivity table.

Live: _coming soon (Streamlit Community Cloud)_

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pytest
```

## Model

Two-stage DCF over a 5-year explicit projection plus a Gordon Growth terminal
value:

```
FCFF_t  = NOPAT_t + D&A_t - Capex_t - dWC_t
NOPAT_t = Revenue_t * operating_margin * (1 - tax_rate)
EV      = sum_t FCFF_t / (1 + WACC)^t  +  TV / (1 + WACC)^N
TV      = FCFF_{N+1} / (WACC - g_terminal)
Equity  = EV - net_debt
Price   = Equity / shares_outstanding
```

The valuation engine in `dcf/engine.py` is pure Python with no I/O, validated
against a hand-computed textbook problem in `tests/test_engine.py`.
