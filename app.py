"""DCF Calculator — Streamlit UI."""

from __future__ import annotations

import streamlit as st

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


# ----- sidebar -------------------------------------------------------------

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

    st.markdown("---")
    st.caption(
        "Enter a ticker and click **Load financials** to pull the latest "
        "fundamentals from Yahoo Finance. Assumption sliders will appear "
        "next, pre-filled with the company's historical averages."
    )


# ----- main ----------------------------------------------------------------

st.title("DCF Valuation")
st.caption("Two-stage discounted cash flow model · pulls live financials from Yahoo Finance")

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

# --- company header ---
header_left, header_right = st.columns([3, 1])
with header_left:
    st.subheader(f"{company.ticker} · {company.name}")
    if company.sector or company.industry:
        st.caption(f"{company.sector} · {company.industry}")
with header_right:
    if company.current_price:
        st.metric("Current price", f"${company.current_price:,.2f}")

# --- warnings ---
for w in company.warnings:
    st.warning(w)

# --- fundamentals row ---
st.markdown("#### Fundamentals (latest fiscal year)")
c1, c2, c3 = st.columns(3)
c1.metric("Revenue", _format_money(company.revenue_base))
c2.metric("Shares outstanding", _format_money(company.shares_outstanding).replace("$", ""))
c3.metric(
    "Net debt",
    _format_money(company.net_debt),
    help="Total debt minus cash and equivalents.",
)

# --- historical averages ---
st.markdown("#### Historical 4-year averages")
h1, h2, h3 = st.columns(3)
h1.metric("Revenue growth (CAGR)", _format_pct(company.historical_revenue_growth))
h2.metric("Operating margin", _format_pct(company.historical_operating_margin))
h3.metric("Effective tax rate", _format_pct(company.historical_tax_rate))

h4, h5, h6 = st.columns(3)
h4.metric("Capex / revenue", _format_pct(company.historical_capex_pct))
h5.metric("D&A / revenue", _format_pct(company.historical_da_pct))
h6.metric("ΔWC / revenue", _format_pct(company.historical_wc_pct))

st.caption(
    "These will become the default values for the DCF assumption sliders "
    "in the next iteration."
)
