"""DCF Calculator — Streamlit UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dcf import DCFInputs, run_dcf
from dcf.data import CompanyData, DataFetchError, fetch_company_data

st.set_page_config(
    page_title="DCF Calculator",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(ticker: str) -> CompanyData:
    return fetch_company_data(ticker)


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


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _default_pct(historical, fallback_pct: float, lo: float, hi: float) -> float:
    """Pick a slider default in percent points, clamped to the slider's range."""
    if historical is None:
        return fallback_pct
    return float(_clamp(historical * 100, lo, hi))


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
    load_clicked = st.button("Load financials", use_container_width=True, type="primary")


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

    revenue_growth = st.slider(
        "Revenue growth (annual)", -10.0, 30.0,
        _default_pct(company.historical_revenue_growth, 5.0, -10.0, 30.0),
        0.5, format="%.1f%%",
    )
    operating_margin = st.slider(
        "Operating margin", -20.0, 60.0,
        _default_pct(company.historical_operating_margin, 15.0, -20.0, 60.0),
        0.5, format="%.1f%%",
    )
    tax_rate = st.slider(
        "Tax rate", 0.0, 40.0,
        _default_pct(company.historical_tax_rate, 21.0, 0.0, 40.0),
        0.5, format="%.1f%%",
    )
    capex_pct = st.slider(
        "Capex / revenue", 0.0, 30.0,
        _default_pct(company.historical_capex_pct, 5.0, 0.0, 30.0),
        0.5, format="%.1f%%",
    )
    da_pct = st.slider(
        "D&A / revenue", 0.0, 30.0,
        _default_pct(company.historical_da_pct, 5.0, 0.0, 30.0),
        0.5, format="%.1f%%",
    )
    wc_pct_input = st.slider(
        "Working capital / revenue", -10.0, 30.0,
        _default_pct(company.historical_wc_pct, 2.0, -10.0, 30.0),
        0.5, format="%.1f%%",
        help="Working capital balance as a percent of revenue. Year-over-year changes drive ΔWC in the FCF calc.",
    )

    st.markdown("---")
    st.markdown("##### Discount rate & terminal value")
    wacc = st.slider("WACC (discount rate)", 4.0, 20.0, 9.0, 0.25, format="%.2f%%")
    terminal_growth = st.slider("Terminal growth rate", 0.0, 5.0, 2.5, 0.25, format="%.2f%%")
    projection_years = st.slider("Projection years", 3, 10, 5)


# ----- build inputs --------------------------------------------------------

inputs = DCFInputs(
    revenue_base=company.revenue_base,
    shares_outstanding=company.shares_outstanding,
    net_debt=company.net_debt,
    revenue_growth=revenue_growth / 100,
    operating_margin=operating_margin / 100,
    tax_rate=tax_rate / 100,
    capex_pct=capex_pct / 100,
    da_pct=da_pct / 100,
    wc_pct=wc_pct_input / 100,
    terminal_growth=terminal_growth / 100,
    wacc=wacc / 100,
    projection_years=projection_years,
)

try:
    result = run_dcf(inputs)
except ValueError as e:
    st.error(f"Invalid assumptions: {e}")
    st.stop()


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

if current > 0:
    upside = (fair_value - current) / current
    if upside > 0.10:
        verdict = f"▲ {upside * 100:.1f}% undervalued"
    elif upside < -0.10:
        verdict = f"▼ {abs(upside) * 100:.1f}% overvalued"
    else:
        verdict = f"≈ Fairly valued ({upside * 100:+.1f}%)"
else:
    upside = None
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


# ----- year-by-year projection table ---------------------------------------

st.markdown("### Free cash flow projection")

table = pd.DataFrame(
    {
        "Year": [f"Year {p.year}" for p in result.projections],
        "Revenue": [p.revenue for p in result.projections],
        "EBIT": [p.ebit for p in result.projections],
        "NOPAT": [p.nopat for p in result.projections],
        "+ D&A": [p.da for p in result.projections],
        "− Capex": [-p.capex for p in result.projections],
        "− ΔWC": [-p.delta_wc for p in result.projections],
        "FCF": [p.fcf for p in result.projections],
        "Discount factor": [p.discount_factor for p in result.projections],
        "PV(FCF)": [p.pv_fcf for p in result.projections],
    }
)

st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Revenue": st.column_config.NumberColumn(format="$%.0f"),
        "EBIT": st.column_config.NumberColumn(format="$%.0f"),
        "NOPAT": st.column_config.NumberColumn(format="$%.0f"),
        "+ D&A": st.column_config.NumberColumn(format="$%.0f"),
        "− Capex": st.column_config.NumberColumn(format="$%.0f"),
        "− ΔWC": st.column_config.NumberColumn(format="$%.0f"),
        "FCF": st.column_config.NumberColumn(format="$%.0f"),
        "Discount factor": st.column_config.NumberColumn(format="%.4f"),
        "PV(FCF)": st.column_config.NumberColumn(format="$%.0f"),
    },
)

s1, s2, s3 = st.columns(3)
s1.metric("Sum of PV(FCF)", _format_money(result.sum_pv_fcf))
s2.metric("Terminal value", _format_money(result.terminal_value))
s3.metric("PV of terminal value", _format_money(result.pv_terminal_value))

st.caption(
    f"Terminal value computed via Gordon Growth: "
    f"FCF × (1 + g) / (WACC − g), with g = {terminal_growth:.2f}% and WACC = {wacc:.2f}%. "
    f"Terminal value contributes "
    f"{result.pv_terminal_value / result.enterprise_value * 100:.0f}% of enterprise value — "
    f"the higher this share, the more your valuation depends on assumptions about the distant future."
)
