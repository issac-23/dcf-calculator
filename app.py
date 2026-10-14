"""DCF Calculator — Streamlit UI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dcf.assumptions import (
    PROJECTION_YEARS,
    SLIDER_BOUNDS,
    WACC_MAX_PCT,
    WACC_MIN_PCT,
    seed_inputs,
    seeded_wacc_pct,
)
from dcf.data import (
    CompanyData,
    DataFetchError,
    fetch_company_data,
    fetch_risk_free_rate,
)
from dcf.valuation import run_valuation
from dcf.wacc import DEFAULT_EQUITY_RISK_PREMIUM

st.set_page_config(
    page_title="DCF Calculator",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- styling -------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    #MainMenu, footer, header [data-testid="stToolbar"] {
        visibility: hidden;
    }
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stMetric"] {
        background: #15181c;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border: 1px solid #23262c;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        opacity: 0.65;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    .stPlotlyChart {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Shared Plotly layout for the dark theme
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6e8eb", family="sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#23262c", zerolinecolor="#23262c"),
    yaxis=dict(gridcolor="#23262c", zerolinecolor="#23262c"),
)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(ticker: str) -> CompanyData:
    return fetch_company_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_risk_free() -> tuple[float, bool]:
    return fetch_risk_free_rate()


# Slider labels by SLIDER_BOUNDS field, shared by the widgets and the notice
# that names which seeded defaults had to be clamped.
_SLIDER_LABELS = {
    "revenue_growth": "Revenue growth (annual)",
    "operating_margin": "Operating margin",
    "tax_rate": "Tax rate",
    "capex_pct": "Capex / revenue",
    "da_pct": "D&A / revenue",
    "wc_pct": "Working capital / revenue",
}


def _format_money(x: float) -> str:
    if x is None:
        return "—"
    abs_x = abs(x)
    sign = "-" if x < 0 else ""
    if abs_x >= 1e12:
        return f"{sign}${abs_x / 1e12:.2f}T"
    if abs_x >= 1e9:
        return f"{sign}${abs_x / 1e9:.2f}B"
    if abs_x >= 1e6:
        return f"{sign}${abs_x / 1e6:.2f}M"
    return f"{sign}${abs_x:,.0f}"


def _format_pct(x):
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


# ----- sidebar: ticker -----------------------------------------------------

with st.sidebar:
    st.markdown("### Ticker")
    ticker_input = st.text_input(
        "Stock symbol",
        value="AAPL",
        max_chars=10,
        label_visibility="collapsed",
        placeholder="AAPL",
    ).strip().upper()
    load_clicked = st.button("Load financials", width="stretch", type="primary")


# ----- main ----------------------------------------------------------------

st.title("DCF Valuation")
st.caption("Two-stage discounted cash flow model · live financials from Yahoo Finance")

if "company" not in st.session_state:
    st.session_state.company = None

if load_clicked and ticker_input:
    try:
        with st.spinner(f"Fetching {ticker_input}..."):
            st.session_state.company = _cached_fetch(ticker_input)
    except DataFetchError as e:
        st.session_state.company = None
        st.error(str(e))
    except Exception as e:
        st.session_state.company = None
        st.error(f"Unexpected error fetching {ticker_input}: {e}")

company: CompanyData | None = st.session_state.company

if company is None:
    st.info("Enter a ticker in the sidebar and click **Load financials** to begin.")
    st.stop()


# ----- sidebar: assumptions (only after a company is loaded) ----------------

with st.sidebar:
    st.markdown("---")
    st.markdown("### Assumptions")
    st.caption(
        f"Pre-filled with {company.ticker}'s 4-year historical averages where available. "
        "Drag any slider to test your own thesis."
    )

    # Defaults and bounds both come from dcf/assumptions.py, so this page and
    # the E2E valuation report start from the same inputs by construction.
    risk_free, rf_is_live = _cached_risk_free()
    seed = seed_inputs(company, risk_free, DEFAULT_EQUITY_RISK_PREMIUM)
    seeded = seed.inputs

    def _pct_slider(field: str, default: float, **kwargs) -> float:
        _, lo, hi = SLIDER_BOUNDS[field]
        return st.slider(
            _SLIDER_LABELS[field], lo, hi, default * 100, 0.5, format="%.1f%%", **kwargs
        )

    revenue_growth = _pct_slider("revenue_growth", seeded.revenue_growth)
    operating_margin = _pct_slider("operating_margin", seeded.operating_margin)
    tax_rate = _pct_slider("tax_rate", seeded.tax_rate)
    capex_pct = _pct_slider("capex_pct", seeded.capex_pct)
    da_pct = _pct_slider("da_pct", seeded.da_pct)
    wc_pct_input = _pct_slider(
        "wc_pct",
        seeded.wc_pct,
        help="Working capital balance as a percent of revenue. Year-over-year changes drive ΔWC in the FCF calc.",
    )

    if seed.clamped:
        names = ", ".join(
            label for field, label in _SLIDER_LABELS.items() if field in seed.clamped
        )
        st.info(
            f"{company.ticker}'s historical average for {names} falls outside the "
            "slider's range, so the default was clamped to the nearest bound. "
            "The model is not using the figure the history actually shows."
        )

    st.markdown("---")
    st.markdown("##### Discount rate & terminal value")

    erp_pct = st.slider(
        "Equity risk premium", 3.0, 8.0, DEFAULT_EQUITY_RISK_PREMIUM * 100, 0.25,
        format="%.2f%%",
        help=(
            "The only CAPM input that cannot be looked up — it is an assumption "
            "about future returns, not a measurement. Damodaran's implied "
            "premium for mature markets has sat in the 4–6% band for years."
        ),
    )
    erp_seed = seed_inputs(company, risk_free, erp_pct / 100)

    # The derived figure seeds the slider rather than replacing it. WACC is
    # the most consequential input in the model, so the user keeps the final
    # say — but the starting point is now computed instead of a round 9%.
    derived_pct = seeded_wacc_pct(erp_seed.wacc)
    wacc = st.slider(
        "WACC (discount rate)", WACC_MIN_PCT, WACC_MAX_PCT, derived_pct, 0.25,
        format="%.2f%%",
        help=(
            "Seeded from CAPM using this company's beta and capital structure. "
            "Drag to override."
            if erp_seed.wacc
            else "Could not be derived for this ticker, so this is a generic default."
        ),
    )
    if erp_seed.wacc:
        st.caption(f"Derived: **{erp_seed.wacc.wacc * 100:.2f}%**")
    else:
        st.caption(f"Not derived — {erp_seed.wacc_reason}")
    tg_default, tg_lo, tg_hi = SLIDER_BOUNDS["terminal_growth"]
    terminal_growth = st.slider(
        "Terminal growth rate", tg_lo, tg_hi, tg_default, 0.25, format="%.2f%%"
    )
    years_default, years_lo, years_hi = PROJECTION_YEARS
    projection_years = st.slider("Projection years", years_lo, years_hi, years_default)


# ----- the valuation ---------------------------------------------------------

try:
    valuation = run_valuation(
        company,
        risk_free,
        {
            "revenue_growth": revenue_growth,
            "operating_margin": operating_margin,
            "tax_rate": tax_rate,
            "capex_pct": capex_pct,
            "da_pct": da_pct,
            "wc_pct": wc_pct_input,
            "erp": erp_pct,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "projection_years": projection_years,
        },
    )
except ValueError as e:
    st.error(f"Invalid assumptions: {e}")
    st.stop()

result = valuation.result
wacc_breakdown = valuation.wacc
wacc_reason = valuation.wacc_reason


# ----- company header ------------------------------------------------------

header_left, header_right = st.columns([3, 1])
with header_left:
    st.subheader(f"{company.ticker} · {company.name}")
    if company.sector or company.industry:
        st.caption(f"{company.sector} · {company.industry}")
with header_right:
    if company.current_price:
        st.metric("Current price", f"${company.current_price:,.2f}")

for w in company.warnings:
    st.warning(w)


# ----- valuation summary ---------------------------------------------------

st.markdown("### Valuation")

fair_value = result.fair_value_per_share
current = company.current_price or 0.0
upside = valuation.upside

if valuation.verdict == "undervalued":
    verdict = f"▲ {upside * 100:.1f}% undervalued"
elif valuation.verdict == "overvalued":
    verdict = f"▼ {abs(upside) * 100:.1f}% overvalued"
elif valuation.verdict == "fair":
    verdict = f"≈ Fairly valued ({upside * 100:+.1f}%)"
else:
    verdict = "Current price unavailable"

v1, v2, v3 = st.columns(3)
v1.metric("Fair value / share", f"${fair_value:,.2f}", verdict if current > 0 else None)
v2.metric("Current price", f"${current:,.2f}" if current > 0 else "—")
v3.metric(
    "Implied upside",
    f"{upside * 100:+.1f}%" if upside is not None else "—",
)

e1, e2, e3 = st.columns(3)
e1.metric("Enterprise value", _format_money(result.enterprise_value))
e2.metric("Equity value", _format_money(result.equity_value))
e3.metric("Net debt", _format_money(company.net_debt))


# ----- cost of capital -----------------------------------------------------

st.markdown("### Cost of capital")

if wacc_breakdown is None:
    st.info(
        f"WACC could not be derived for {company.ticker} — {wacc_reason} "
        f"The model is using the **{wacc:.2f}%** set in the sidebar."
    )
else:
    b = wacc_breakdown
    st.caption(
        "Built from CAPM rather than assumed. WACC discounts every projected "
        "cash flow and sets the terminal value denominator, so it moves the "
        "answer more than any other input — it is worth seeing where it comes from."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Cost of equity",
        f"{b.cost_of_equity * 100:.2f}%",
        f"{risk_free * 100:.2f}% risk-free + {company.beta:.2f}β × {erp_pct:.2f}%",
        delta_color="off",
    )
    if b.cost_of_debt_after_tax is not None:
        c2.metric(
            "Cost of debt (after tax)",
            f"{b.cost_of_debt_after_tax * 100:.2f}%",
            f"{b.cost_of_debt_pretax * 100:.2f}% pre-tax",
            delta_color="off",
        )
    else:
        c2.metric("Cost of debt (after tax)", "—")
    c3.metric(
        "Weights",
        f"{b.weight_equity * 100:.1f}% / {b.weight_debt * 100:.1f}%",
        "equity / debt",
        delta_color="off",
    )

    used_note = (
        "matches the slider"
        if abs(wacc - b.wacc * 100) < 0.01
        else f"you have overridden it to {wacc:.2f}%"
    )
    st.caption(
        f"Weighted average: **{b.wacc * 100:.2f}%** — {used_note}. "
        f"Risk-free rate is the 10-year Treasury"
        + (
            f" at {risk_free * 100:.2f}%."
            if rf_is_live
            else f", which could not be fetched, so a {risk_free * 100:.2f}% fallback is in use."
        )
    )

    if b.debt_cost_unknown:
        st.warning(
            f"{company.ticker} carries {_format_money(company.total_debt)} of debt, but "
            "no interest expense was reported, so its cost could not be established. "
            "The figure above is the cost of equity only, which understates WACC for "
            "a leveraged company."
        )

    with st.expander("Why this is an estimate"):
        st.markdown(
            """
**Cost of debt is backward-looking.** It divides reported interest expense by
total debt, which gives the average rate the company is *already* paying, not
the marginal rate it would pay to borrow today. Those diverge once rates move.
The marginal rate is the theoretically correct input and is not disclosed.

**The equity risk premium is not observable.** It is an assumption about future
returns. The slider exists because your view is as legitimate as the default.

**Beta is backward-looking too.** It is regressed on historical returns, so it
describes how the stock *has* moved, not how it will.

None of this makes a derived WACC worse than a guessed one — it makes the
guess visible.
            """
        )


# ----- year-by-year projection table ---------------------------------------

st.markdown("### Free cash flow projection")

table = pd.DataFrame(
    {
        "Year": [f"Year {p.year}" for p in result.projections],
        "Revenue": [_format_money(p.revenue) for p in result.projections],
        "EBIT": [_format_money(p.ebit) for p in result.projections],
        "NOPAT": [_format_money(p.nopat) for p in result.projections],
        "+ D&A": [_format_money(p.da) for p in result.projections],
        "− Capex": [_format_money(-p.capex) for p in result.projections],
        "− ΔWC": [_format_money(-p.delta_wc) for p in result.projections],
        "FCF": [_format_money(p.fcf) for p in result.projections],
        "Discount factor": [p.discount_factor for p in result.projections],
        "PV(FCF)": [_format_money(p.pv_fcf) for p in result.projections],
    }
)

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Discount factor": st.column_config.NumberColumn(format="%.4f"),
    },
)

s1, s2, s3 = st.columns(3)
s1.metric("Sum of PV(FCF)", _format_money(result.sum_pv_fcf))
s2.metric("Terminal value", _format_money(result.terminal_value))
s3.metric("PV of terminal value", _format_money(result.pv_terminal_value))

tv_share_text = (
    ""
    if valuation.tv_share is None
    else f"Terminal value contributes {valuation.tv_share * 100:.0f}% of enterprise "
    "value — the higher this share, the more your valuation depends on "
    "assumptions about the distant future."
)
st.caption(
    f"Terminal value computed via Gordon Growth: "
    f"FCF × (1 + g) / (WACC − g), with g = {terminal_growth:.2f}% and WACC = {wacc:.2f}%. "
    + tv_share_text
)


# ----- FCF chart -----------------------------------------------------------

st.markdown("### Cash flow visualization")

years = [f"Year {p.year}" for p in result.projections]
fcfs = [p.fcf for p in result.projections]
pv_fcfs = [p.pv_fcf for p in result.projections]

fcf_fig = go.Figure()
fcf_fig.add_bar(
    x=years,
    y=fcfs,
    name="Undiscounted FCF",
    marker_color="#1f3a5f",
    hovertemplate="%{x}<br>FCF: $%{y:,.0f}<extra></extra>",
)
fcf_fig.add_bar(
    x=years,
    y=pv_fcfs,
    name="Present value",
    marker_color="#3b82f6",
    hovertemplate="%{x}<br>PV: $%{y:,.0f}<extra></extra>",
)
fcf_fig.update_layout(
    barmode="overlay",
    bargap=0.25,
    title=dict(text="Free cash flow vs. present value", font=dict(size=14)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=320,
    **_PLOT_LAYOUT,
)
fcf_fig.update_traces(opacity=0.95)
st.plotly_chart(fcf_fig, width="stretch")


# ----- reverse DCF ---------------------------------------------------------

if valuation.implied is not None:
    st.markdown("### Reverse DCF")
    st.caption(
        "Instead of asking *what is this stock worth?*, ask *what growth rate "
        "would the market need to see to justify the current price?* All other "
        "assumptions above are held fixed."
    )

    implied = valuation.implied.rate

    r1, r2, r3 = st.columns(3)
    r1.metric("Current price", f"${company.current_price:,.2f}")
    if implied is None:
        r2.metric("Implied revenue growth", "—")
        r3.metric("Your assumption", f"{revenue_growth:.1f}%")
        st.caption(
            "No revenue growth rate between −50% and +100% reproduces the current "
            "price under the other assumptions. Try adjusting margin, WACC, or "
            "terminal growth."
        )
    else:
        implied_pct = implied * 100
        delta = implied_pct - revenue_growth
        r2.metric(
            "Implied revenue growth",
            f"{implied_pct:.1f}%",
            f"{delta:+.1f} pp vs. your assumption",
            delta_color="off",
        )
        hist = company.historical_revenue_growth
        if hist is not None:
            r3.metric("Historical (4-yr avg)", f"{hist * 100:.1f}%")
        else:
            r3.metric("Your assumption", f"{revenue_growth:.1f}%")

        gap = valuation.implied.gap_vs_history
        if valuation.implied.reading == "optimistic":
            interp = (
                f"The market is pricing in growth **{gap * 100:.1f} pp above** the "
                "4-year historical average — an optimistic take that assumes "
                "the business accelerates."
            )
        elif valuation.implied.reading == "pessimistic":
            interp = (
                f"The market is pricing in growth **{abs(gap) * 100:.1f} pp below** "
                "the 4-year historical average — a pessimistic take that "
                "assumes the business decelerates."
            )
        elif valuation.implied.reading == "in_line":
            interp = (
                "The market's implied growth is roughly in line with the "
                "4-year historical average."
            )
        else:
            interp = None
        if interp:
            st.caption(interp)


# ----- sensitivity heatmap -------------------------------------------------

st.markdown("### Sensitivity analysis")
st.caption(
    "Fair value per share across a grid of WACC and terminal growth assumptions. "
    "If small moves change the answer drastically, your valuation is fragile."
)

wacc_grid = valuation.sensitivity.wacc
g_grid = valuation.sensitivity.terminal_growth

heatmap_z = np.array(valuation.sensitivity.fair_value)
text = np.where(
    np.isnan(heatmap_z),
    "—",
    np.vectorize(lambda v: f"${v:,.0f}" if not np.isnan(v) else "—")(heatmap_z),
)

heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_z,
        x=[f"{g * 100:.1f}%" for g in g_grid],
        y=[f"{w * 100:.2f}%" for w in wacc_grid],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="Blues",
        showscale=False,
        hovertemplate="WACC %{y} · g %{x}<br>Fair value: %{text}<extra></extra>",
    )
)
heatmap.update_layout(
    title=dict(text="Fair value per share (WACC × terminal growth)", font=dict(size=14)),
    xaxis=dict(title="Terminal growth", side="top"),
    yaxis=dict(title="WACC", autorange="reversed"),
    height=380,
    **{k: v for k, v in _PLOT_LAYOUT.items() if k not in ("xaxis", "yaxis")},
)
st.plotly_chart(heatmap, width="stretch")


# ----- methodology expander ------------------------------------------------

with st.expander("Methodology"):
    st.markdown(
        """
**Two-stage discounted cash flow.** Free cash flow to the firm (FCFF) is
projected explicitly for the next *N* years using assumptions on the left.
Beyond year *N*, a Gordon Growth terminal value captures all remaining cash
flows in perpetuity. Everything is discounted to today at the weighted
average cost of capital.

```
FCFF_t  = NOPAT_t + D&A_t − Capex_t − ΔWC_t
NOPAT_t = Revenue_t × operating margin × (1 − tax rate)
EV      = Σ FCFF_t / (1 + WACC)^t  +  TV / (1 + WACC)^N
TV      = FCFF_{N+1} / (WACC − g_terminal)
Equity  = EV − net debt
Price   = Equity / shares outstanding
```

**Limitations.** DCFs are unreliable for banks (no meaningful FCF — those
are flagged at load), REITs (use FFO/NAV instead), and very early-stage
companies (no historical signal). Even where the model applies, terminal
value typically dominates enterprise value, so the answer is sensitive to
small changes in WACC and terminal growth — see the sensitivity grid above.
        """
    )
