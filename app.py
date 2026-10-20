"""Streamlit web app: rank Borsa Istanbul stocks by technical, valuation and quality scores,
with statistical 1- and 3-month trend projections.

Run with:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from data import fetch_fundamentals, fetch_prices, parse_tickers
from indicators import sma
from projection import projection_path
from scoring import RATINGS, build_table

DEFAULT_TICKERS = (
    "THYAO, ASELS, GARAN, AKBNK, YKBNK, ISCTR, KCHOL, SAHOL, BIMAS, EREGL, "
    "TUPRS, FROTO, TOASO, SISE, TCELL, PGSUS, KRDMD, PETKM, ENKAI, ASTOR"
)

DISCLAIMER = (
    "For information and research only, not investment advice. Scores and projections are "
    "calculated mechanically from historical prices and reported financial ratios. They do "
    "not account for news, market manipulation or other external events, and past trends "
    "do not guarantee future results."
)

# Rating label -> (arrow, badge colour, chart colour)
RATING_STYLE = {
    "Strong": ("▲", "green", "#22C55E"),
    "Positive": ("↗", "green", "#86EFAC"),
    "Neutral": ("→", "gray", "#94A3B8"),
    "Weak": ("↘", "orange", "#F59E0B"),
    "Very weak": ("▼", "red", "#EF4444"),
    "No data": ("·", "gray", "#475569"),
}

SIGNAL_LABELS = {
    "sig_rsi": "RSI in healthy range (45-65)",
    "sig_above_sma50": "Price above 50-day average",
    "sig_above_sma200": "Price above 200-day average",
    "sig_sma50_above_sma200": "50-day average above 200-day average",
    "sig_momentum_3m": "Positive 3-month return",
}

SERIES_COLORS = alt.Scale(
    domain=["Close", "SMA 50", "SMA 200", "Projection"],
    range=["#60A5FA", "#F59E0B", "#A78BFA", "#34D399"],
)


# ---------- Small formatting helpers ----------

def fmt_pct(value, decimals=1):
    return "–" if pd.isna(value) else f"{value:+.{decimals}f}%"


def fmt_num(value, decimals=1):
    return "–" if pd.isna(value) else f"{value:.{decimals}f}"


def rating_text(label):
    return f"{RATING_STYLE[label][0]} {label}"


# ---------- Charts ----------

def score_bar_chart(table):
    data = table.reset_index(names="ticker")
    labels = [label for _, label in RATINGS]
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("total:Q", title="Total score", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("ticker:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color(
            "rating:N",
            scale=alt.Scale(domain=labels, range=[RATING_STYLE[x][2] for x in labels]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("ticker:N", title="Stock"),
            alt.Tooltip("total:Q", title="Total score", format=".0f"),
            alt.Tooltip("rating:N", title="Rating"),
        ],
    ).properties(height=max(320, 26 * len(data)))


def score_vs_projection_chart(table):
    data = table.reset_index(names="ticker")
    labels = [label for _, label in RATINGS]
    base = alt.Chart(data).encode(
        x=alt.X("total:Q", title="Total score", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("proj_3m:Q", title="3-month trend projection (%)"),
    )
    points = base.mark_circle(size=110, opacity=0.9).encode(
        color=alt.Color(
            "rating:N",
            scale=alt.Scale(domain=labels, range=[RATING_STYLE[x][2] for x in labels]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("ticker:N", title="Stock"),
            alt.Tooltip("total:Q", title="Total score", format=".0f"),
            alt.Tooltip("proj_3m:Q", title="3M projection %", format="+.1f"),
            alt.Tooltip("up_prob_3m:Q", title="P(higher) 3M %", format=".0f"),
        ],
    )
    names = base.mark_text(align="left", dx=8, fontSize=11, color="#CBD5E1").encode(text="ticker:N")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="#64748B").encode(y="y:Q")
    return (zero + points + names).properties(height=max(320, 26 * len(data)))


def price_chart(close):
    history = pd.DataFrame({"Close": close, "SMA 50": sma(close, 50), "SMA 200": sma(close, 200)})
    history = history.rename_axis("date").reset_index().melt(
        "date", var_name="series", value_name="price").dropna()
    path = projection_path(close).rename_axis("date").reset_index().assign(series="Projection")

    y_axis = alt.Y("price:Q", title="Price (TRY)", scale=alt.Scale(zero=False))
    lines = alt.Chart(history).mark_line(strokeWidth=1.8).encode(
        x=alt.X("date:T", title=None),
        y=y_axis,
        color=alt.Color("series:N", scale=SERIES_COLORS, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
                 alt.Tooltip("price:Q", format=".2f")],
    )
    band = alt.Chart(path).mark_area(opacity=0.18, color="#34D399").encode(
        x="date:T",
        y=alt.Y("low:Q", title="Price (TRY)", scale=alt.Scale(zero=False)),
        y2="high:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("low:Q", title="Likely low", format=".2f"),
                 alt.Tooltip("high:Q", title="Likely high", format=".2f")],
    )
    projected = alt.Chart(path).mark_line(strokeDash=[6, 4], strokeWidth=2).encode(
        x="date:T",
        y=alt.Y("expected:Q", title="Price (TRY)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=SERIES_COLORS),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("expected:Q", title="Projected", format=".2f")],
    )
    return (band + lines + projected).properties(height=430).interactive(bind_y=False)


# ---------- Page setup and sidebar ----------

st.set_page_config(page_title="BIST Stock Screener", page_icon="📈", layout="wide")

with st.sidebar:
    st.header("Settings")
    tickers = parse_tickers(st.text_area("Tickers (comma separated)", DEFAULT_TICKERS, height=170))
    st.subheader("Score weights")
    weights = {
        "technical": st.slider("Technical", 0, 100, 40),
        "valuation": st.slider("Valuation", 0, 100, 30),
        "quality": st.slider("Quality", 0, 100, 30),
    }
    st.caption("Weights are relative: 40/30/30 and 4/3/3 give the same result.")
    st.divider()
    st.caption("Data: Yahoo Finance, cached for one hour.")

st.title("BIST Stock Screener")
st.markdown(
    "Multi-factor ranking of Borsa Istanbul stocks by **technical trend**, **valuation** and "
    "**quality**, with statistical 1- and 3-month trend projections."
)
st.info(DISCLAIMER, icon=":material/info:")

if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()
if sum(weights.values()) == 0:
    st.warning("At least one weight must be above zero.")
    st.stop()


# ---------- Data (cached for one hour so Yahoo is not called on every click) ----------

@st.cache_data(ttl=3600, show_spinner="Downloading data from Yahoo Finance...")
def load_data(tickers: tuple[str, ...]):
    return fetch_prices(list(tickers)), fetch_fundamentals(list(tickers))


try:
    prices, fundamentals = load_data(tuple(tickers))
except Exception as error:
    st.error(f"Could not download data: {error}")
    st.stop()

if prices.empty:
    st.error("Yahoo Finance returned no prices. Check the tickers or try again later.")
    st.stop()

missing = sorted(set(tickers) - set(prices.columns))
if missing:
    st.warning("No price data for: " + ", ".join(missing))

table = build_table(prices, fundamentals, weights)
last_close = prices.index.max()


# ---------- Market overview ----------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Stocks analysed", len(table), border=True)
k2.metric("Above 200-day average", f"{table['sig_above_sma200'].mean() * 100:.0f}%",
          help="Share of stocks trading above their 200-day moving average (market breadth).",
          border=True)
k3.metric("Average total score", f"{table['total'].mean():.0f} / 100", border=True)
k4.metric("Median 1M projection", fmt_pct(table["proj_1m"].median()),
          help="Median of the 1-month trend projections across all stocks.", border=True)
k5.metric("Last close", last_close.strftime("%d %b %Y"), border=True)

tab_ranking, tab_detail, tab_method = st.tabs(
    [":material/leaderboard: Ranking", ":material/candlestick_chart: Stock detail",
     ":material/menu_book: Methodology"]
)


# ---------- Tab 1: ranking ----------

with tab_ranking:
    st.subheader("Highest total scores")
    for column, (ticker, row) in zip(st.columns(3), table.head(3).iterrows()):
        with column.container(border=True):
            st.markdown(f"#### #{int(row['rank'])} · {ticker}")
            arrow, colour, _ = RATING_STYLE[row["rating"]]
            st.badge(f"{arrow} {row['rating']}", color=colour)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total score", fmt_num(row["total"], 0))
            c2.metric("1M projection", fmt_pct(row["proj_1m"]))
            c3.metric("3M projection", fmt_pct(row["proj_3m"]))
            st.caption(
                f"Probability of a higher price: {fmt_num(row['up_prob_1m'], 0)}% in 1M · "
                f"{fmt_num(row['up_prob_3m'], 0)}% in 3M"
            )

    lowest = table.tail(3).index[::-1]
    st.caption("Lowest total scores: " + " · ".join(
        f"{t} ({fmt_num(table.loc[t, 'total'], 0)})" for t in lowest))

    left, right = st.columns(2)
    with left:
        st.markdown("##### Total score by stock")
        st.altair_chart(score_bar_chart(table))
    with right:
        st.markdown("##### Score vs. 3-month trend projection")
        st.altair_chart(score_vs_projection_chart(table))
        st.caption("Top-right: high score and a rising trend. Bottom-left: low score and a falling trend.")

    st.markdown("##### Full ranking")
    view = table.reset_index(names="ticker")
    view["rating"] = view["rating"].map(rating_text)
    view["roe"] *= 100  # Yahoo gives ROE as a fraction (0.25 = 25%)

    def score_column(label, width="small"):
        return st.column_config.ProgressColumn(label, min_value=0, max_value=100,
                                               format="%.0f", width=width)

    def number_column(label, fmt="%.1f", help=None):
        return st.column_config.NumberColumn(label, format=fmt, help=help)

    st.dataframe(
        view,
        height=(len(view) + 1) * 35 + 3,  # show every row without scrolling
        hide_index=True,
        placeholder="–",
        column_order=["rank", "ticker", "rating", "total", "technical", "valuation", "quality",
                      "proj_1m", "up_prob_1m", "proj_3m", "up_prob_3m", "price",
                      "change_1d", "rsi", "pe", "roe"],
        column_config={
            "rank": number_column("#", "%d"),
            "ticker": st.column_config.TextColumn("Stock", pinned=True),
            "rating": st.column_config.TextColumn("Rating"),
            "total": score_column("Total"),
            "technical": score_column("Technical"),
            "valuation": score_column("Valuation"),
            "quality": score_column("Quality"),
            "proj_1m": number_column("1M proj.", "%+.1f%%", "Expected change if the 6-month trend continues"),
            "up_prob_1m": number_column("P(up) 1M", "%.0f%%", "Probability of a higher price in 1 month"),
            "proj_3m": number_column("3M proj.", "%+.1f%%", "Expected change if the 6-month trend continues"),
            "up_prob_3m": number_column("P(up) 3M", "%.0f%%", "Probability of a higher price in 3 months"),
            "price": number_column("Price ₺", "%.2f"),
            "change_1d": number_column("1D", "%+.2f%%"),
            "rsi": number_column("RSI 14"),
            "pe": number_column("P/E"),
            "roe": number_column("ROE %"),
        },
    )
    st.caption("More metrics (P/B, net margin, 3-month return, signals) are in the "
               "Stock detail tab and in the CSV export.")
    st.download_button(":material/download: Download CSV", table.to_csv().encode("utf-8"),
                       file_name="bist_screener.csv", mime="text/csv")


# ---------- Tab 2: one stock in detail ----------

with tab_detail:
    choice = st.selectbox(
        "Select a stock", table.index,
        format_func=lambda t: f"{t}  ·  #{int(table.loc[t, 'rank'])}  ·  {table.loc[t, 'rating']}",
    )
    row = table.loc[choice]
    close = prices[choice].dropna()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last price", f"{row['price']:.2f} ₺",
              delta=None if pd.isna(row["change_1d"]) else fmt_pct(row["change_1d"], 2), border=True)
    m2.metric("Total score", f"{fmt_num(row['total'], 0)} / 100",
              help=f"Rank {int(row['rank'])} of {len(table)}", border=True)
    m3.metric("Rating", rating_text(row["rating"]), border=True)
    m4.metric("1M projection", fmt_pct(row["proj_1m"]),
              help=f"Likely range {fmt_pct(row['proj_1m_low'])} to {fmt_pct(row['proj_1m_high'])}",
              border=True)
    m5.metric("3M projection", fmt_pct(row["proj_3m"]),
              help=f"Likely range {fmt_pct(row['proj_3m_low'])} to {fmt_pct(row['proj_3m_high'])}",
              border=True)

    st.markdown("##### Price, moving averages and 3-month projection")
    st.altair_chart(price_chart(close))
    st.caption("Dashed line: projected path if the last 6 months' trend continues. "
               "Shaded area: likely range (±1 standard deviation, about 68% of outcomes).")

    col_scores, col_signals, col_fundamentals = st.columns(3)
    with col_scores, st.container(border=True):
        st.markdown("**Score breakdown**")
        for key in ["technical", "valuation", "quality", "total"]:
            value = row[key]
            st.progress(0 if pd.isna(value) else int(round(value)),
                        text=f"{key.title()}: {fmt_num(value, 0)}")
    with col_signals, st.container(border=True):
        st.markdown("**Technical signals**")
        for key, label in SIGNAL_LABELS.items():
            value = row[key]
            if pd.isna(value):
                icon = ":gray[:material/remove:]"
            elif value == 1:
                icon = ":green[:material/check_circle:]"
            elif value == 0.5:
                icon = ":orange[:material/adjust:]"
            else:
                icon = ":red[:material/cancel:]"
            st.markdown(f"{icon} {label}")
        st.caption(f"RSI 14: {fmt_num(row['rsi'])} · 3-month return: {fmt_pct(row['return_3m'])}")
    with col_fundamentals, st.container(border=True):
        st.markdown("**Fundamentals and projection**")
        st.markdown(
            f"P/E **{fmt_num(row['pe'])}** · P/B **{fmt_num(row['pb'])}** · "
            f"ROE **{fmt_num(row['roe'] * 100)}%** · Net margin **{fmt_num(row['margin'] * 100)}%**"
        )
        st.markdown(
            f"- **1 month:** {fmt_pct(row['proj_1m'])} "
            f"(range {fmt_pct(row['proj_1m_low'])} to {fmt_pct(row['proj_1m_high'])}), "
            f"P(higher) {fmt_num(row['up_prob_1m'], 0)}%\n"
            f"- **3 months:** {fmt_pct(row['proj_3m'])} "
            f"(range {fmt_pct(row['proj_3m_low'])} to {fmt_pct(row['proj_3m_high'])}), "
            f"P(higher) {fmt_num(row['up_prob_3m'], 0)}%"
        )
        st.caption("Statistical scenario, not a forecast.")


# ---------- Tab 3: how it works ----------

with tab_method:
    st.markdown(
        """
#### Scores (0-100)

| Score | Inputs | Logic |
|---|---|---|
| **Technical** | RSI-14, 50/200-day moving averages, 3-month return | Share of positive signals across short, medium and long horizons |
| **Valuation** | P/E, P/B | Percentile rank against the other stocks: cheaper is better. Loss-making companies get 0 on P/E |
| **Quality** | ROE, net margin, debt/equity | Percentile rank: higher profitability and lower debt is better |
| **Total** | The three scores | Weighted average using the sidebar weights. Missing data is skipped, not counted as zero |

#### Rating

| Total score | 70+ | 55-70 | 45-55 | 30-45 | below 30 |
|---|---|---|---|---|---|
| Rating | ▲ Strong | ↗ Positive | → Neutral | ↘ Weak | ▼ Very weak |

#### 1- and 3-month projections

1. A straight line is fitted to the logarithm of the last 6 months of prices. Its slope is the daily trend.
2. Daily volatility is the standard deviation of daily log returns over the same period.
3. **Projection** = trend × number of trading days (21 for 1 month, 63 for 3 months).
4. **Likely range** = projection ± volatility × √days. Uncertainty grows with the square root of time.
5. **P(up)** = probability of a higher price, assuming normally distributed returns.

#### Limitations

- Projections assume the recent trend continues. Trends often reverse, especially after strong moves.
- The score has not been backtested yet, so a higher score is not proven to lead to higher returns.
- Yahoo Finance data for BIST can be delayed or incomplete, especially fundamentals.
"""
    )
    st.info(DISCLAIMER, icon=":material/info:")

st.divider()
st.caption(f"Data: Yahoo Finance · last close {last_close:%d %b %Y} · Not investment advice.")
