import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from functools import lru_cache

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="NIFTY 50 Contribution & Scenario Simulator",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background: radial-gradient(circle at top, #0f172a 0, #020617 40%, #000000 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }
    /* Cards */
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.35);
        backdrop-filter: blur(10px);
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: .12em;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    /* Tables */
    .row_heading, .blank {
        display: none;
    }
    .stDataFrame table {
        border-radius: 0.8rem;
        overflow: hidden;
    }
    thead tr th {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        font-size: 0.85rem !important;
    }
    tbody tr td {
        font-size: 0.8rem !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Constants
# -----------------------------
NIFTY_CONSTITUENT_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

TARGET_POINTS = [100, 200, 250, 500]  # we’ll also show the same negatives

# -----------------------------
# Data Loaders (cached)
# -----------------------------
@st.cache_data(show_spinner="Fetching NIFTY 50 constituents from NSE…")
def load_nifty_constituents() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(NIFTY_CONSTITUENT_URL, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text))
    # Standardize column names if needed
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner="Fetching market caps from Yahoo Finance…")
def fetch_market_caps(symbols: list) -> pd.DataFrame:
    """
    Fetch market cap for each NSE symbol via yfinance.
    Assumes NSE symbols become 'SYMBOL.NS' on Yahoo.
    """
    records = []
    for sym in symbols:
        yahoo_ticker = f"{sym}.NS"
        try:
            tkr = yf.Ticker(yahoo_ticker)
            info = tkr.fast_info
            mcap = info.get("market_cap", None)
            if mcap is None:
                # fallback to slower get_info (older yfinance versions)
                full_info = tkr.get_info()
                mcap = full_info.get("marketCap", None)
        except Exception:
            mcap = None

        records.append(
            {
                "Symbol": sym,
                "YF_Ticker": yahoo_ticker,
                "MarketCap": mcap
            }
        )
    df = pd.DataFrame(records)
    # Drop rows with missing market cap
    df = df.dropna(subset=["MarketCap"])
    return df


@st.cache_data(show_spinner="Fetching latest NIFTY 50 level…")
def get_nifty_level() -> float:
    nifty = yf.Ticker("^NSEI")
    hist = nifty.history(period="5d")
    if hist.empty:
        return np.nan
    # last close
    return float(hist["Close"].iloc[-1])


# -----------------------------
# Helper Functions
# -----------------------------
def compute_weights(nifty_df: pd.DataFrame, mcap_df: pd.DataFrame) -> pd.DataFrame:
    merged = nifty_df.merge(mcap_df, on="Symbol", how="left")
    merged = merged.dropna(subset=["MarketCap"])
    total_mcap = merged["MarketCap"].sum()
    merged["Weight"] = merged["MarketCap"] / total_mcap
    merged["WeightPct"] = merged["Weight"] * 100
    return merged


def build_scenario_table(weighted_df: pd.DataFrame, index_level: float) -> pd.DataFrame:
    df = weighted_df.copy()
    df = df[["Company Name", "Symbol", "Industry", "Weight", "WeightPct"]].copy()
    df["WeightPct"] = df["WeightPct"].round(2)

    for pts in TARGET_POINTS:
        # positive scenario
        col_pos = f"+{pts} pts"
        df[col_pos] = (pts / (index_level * df["Weight"])) * 100
        # negative scenario
        col_neg = f"-{pts} pts"
        df[col_neg] = (-pts / (index_level * df["Weight"])) * 100

    # round all scenario columns
    scenario_cols = [c for c in df.columns if "pts" in c]
    df[scenario_cols] = df[scenario_cols].round(2)

    return df


def sector_summary(weighted_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        weighted_df
        .groupby("Industry", as_index=False)["Weight"]
        .sum()
        .sort_values("Weight", ascending=False)
    )
    grp["WeightPct"] = (grp["Weight"] * 100).round(2)
    return grp


# -----------------------------
# UI Layout
# -----------------------------
st.title("📊 NIFTY 50 Contribution & Scenario Simulator")
st.caption(
    "See NIFTY 50 constituents, sector-wise weights, and how much each stock "
    "must move (in %) to move the index by a given number of points."
)

# Load data
try:
    nifty_df = load_nifty_constituents()
except Exception as e:
    st.error(f"Failed to load NIFTY 50 constituents from NSE: {e}")
    st.stop()

symbols = nifty_df["Symbol"].unique().tolist()
mcap_df = fetch_market_caps(symbols)
index_level = get_nifty_level()

if np.isnan(index_level):
    st.warning("Could not fetch latest NIFTY 50 level from Yahoo Finance. Scenario calculations may not work.")
    st.stop()

weighted_df = compute_weights(nifty_df, mcap_df)
scenario_df = build_scenario_table(weighted_df, index_level)
sector_df = sector_summary(weighted_df)

# -----------------------------
# Top Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">NIFTY 50 Level</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{index_level:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Last close (Yahoo Finance)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Constituents (with market cap)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(weighted_df):d} / 50</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Some may be missing if market cap was not available</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    top_stock = weighted_df.sort_values("Weight", ascending=False).iloc[0]
    st.markdown('<div class="metric-title">Top Weighted Stock</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{top_stock["Company Name"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="metric-sub">{top_stock["Symbol"]} • {top_stock["WeightPct"]:.2f}% of index</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

sector_options = ["All"] + sorted(sector_df["Industry"].tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector (Industry)", sector_options)

min_weight = st.sidebar.slider(
    "Minimum stock weight (%) to show in scenario table",
    min_value=0.0,
    max_value=float(sector_df["WeightPct"].max()) if not sector_df.empty else 10.0,
    value=0.5,
    step=0.1
)

custom_points = st.sidebar.number_input(
    "Custom NIFTY move (points)",
    min_value=-1000,
    max_value=1000,
    value=0,
    step=10
)

# -----------------------------
# Sector-wise View
# -----------------------------
st.subheader("Sector-wise NIFTY 50 Weight")

col_left, col_right = st.columns([1.2, 2])

with col_left:
    st.dataframe(
        sector_df[["Industry", "WeightPct"]],
        hide_index=True,
        use_container_width=True
    )

with col_right:
    import plotly.express as px

    fig = px.bar(
        sector_df,
        x="Industry",
        y="WeightPct",
        title="Sector Weights in NIFTY 50 (approx.)",
    )
    fig.update_layout(
        xaxis_title="Sector (Industry)",
        yaxis_title="Weight (%)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Scenario Simulator
# -----------------------------
st.subheader("Scenario: Required Stock % Move for Given NIFTY Move")

st.write(
    "Assuming **only one stock moves and all others stay flat**, the required % move in that stock "
    "to move the index by a given number of points is:"
)
st.latex(r"r_i = \frac{\Delta I}{L \times w_i}")

# Apply filters
filtered = scenario_df[scenario_df["WeightPct"] >= min_weight].copy()
if selected_sector != "All":
    filtered = filtered[filtered["Industry"] == selected_sector]

# Add custom points column if user entered something non-zero
if custom_points != 0:
    col_name = f"{custom_points:+d} pts"
    filtered[col_name] = (custom_points / (index_level * filtered["Weight"])) * 100
    filtered[col_name] = filtered[col_name].round(2)

display_cols = [
    "Company Name",
    "Symbol",
    "Industry",
    "WeightPct",
] + [c for c in filtered.columns if "pts" in c]

st.dataframe(
    filtered[display_cols].sort_values("WeightPct", ascending=False),
    hide_index=True,
    use_container_width=True
)

st.info(
    "⚠️ These are approximate calculations based on full market capitalisation from Yahoo Finance. "
    "NIFTY uses free-float market cap, so actual weights & contributions will differ slightly."
)
