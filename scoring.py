"""Turns raw prices and fundamentals into 0-100 scores, ratings and one ranked table."""

import numpy as np
import pandas as pd

from indicators import pct_return, rsi, sma
from projection import HORIZONS, project

DEFAULT_WEIGHTS = {"technical": 40, "valuation": 30, "quality": 30}
FUNDAMENTAL_COLUMNS = ["pe", "pb", "roe", "margin", "debt_to_equity"]

# Total score -> rating label (checked from the top down)
RATINGS = [(70, "Strong"), (55, "Positive"), (45, "Neutral"), (30, "Weak"), (0, "Very weak")]


# ---------- Technical score (one stock at a time) ----------

def rsi_signal(value: float) -> float:
    """1 = healthy momentum (45-65), 0.5 = borderline, 0 = overbought / oversold."""
    if pd.isna(value):
        return np.nan
    if 45 <= value <= 65:
        return 1.0
    if 30 <= value <= 70:
        return 0.5
    return 0.0


def technical_signals(close: pd.Series) -> dict:
    """Latest indicator values, the individual signals and a 0-100 technical score.

    Three horizons: short (RSI-14), medium (50-day average)
    and long (200-day average and 3-month return).
    """
    close = close.dropna()
    if close.empty:
        return {"price": np.nan, "rsi": np.nan, "return_3m": np.nan, "technical": np.nan}

    price = float(close.iloc[-1])
    rsi_now = rsi(close).iloc[-1]
    sma50 = sma(close, 50).iloc[-1]
    sma200 = sma(close, 200).iloc[-1]
    ret_3m = pct_return(close, 63)  # about 63 trading days in 3 months

    signals = {
        "sig_rsi": rsi_signal(rsi_now),
        "sig_above_sma50": np.nan if pd.isna(sma50) else float(price > sma50),
        "sig_above_sma200": np.nan if pd.isna(sma200) else float(price > sma200),
        "sig_sma50_above_sma200": (np.nan if pd.isna(sma50) or pd.isna(sma200)
                                   else float(sma50 > sma200)),
        "sig_momentum_3m": np.nan if pd.isna(ret_3m) else float(ret_3m > 0),
    }
    # Ignore signals we could not calculate (e.g. a newly listed stock has no 200-day average)
    available = [s for s in signals.values() if not pd.isna(s)]
    score = float(np.mean(available) * 100) if available else np.nan

    return {
        "price": price,
        "change_1d": pct_return(close, 1),
        "rsi": rsi_now,
        "sma50": sma50,
        "sma200": sma200,
        "return_3m": ret_3m,
        **signals,
        "technical": score,
    }


def projections(close: pd.Series) -> dict:
    """1- and 3-month trend projections, flattened into table columns."""
    result = {}
    for name, days in HORIZONS.items():
        p = project(close, days)
        result[f"proj_{name}"] = p["expected"]
        result[f"proj_{name}_low"] = p["low"]
        result[f"proj_{name}_high"] = p["high"]
        result[f"up_prob_{name}"] = p["prob_up"]
    return result


# ---------- Valuation and quality scores (stocks compared with each other) ----------

def rank_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile rank (0-1] among the screened stocks. Missing values stay missing."""
    return values.rank(pct=True, ascending=higher_is_better)


def valuation_score(table: pd.DataFrame) -> pd.Series:
    """Cheaper than the other stocks on P/E and P/B -> higher score (0-100).

    A negative P/E means the company lost money, so it gets 0 for that metric.
    """
    parts = []
    for column in ["pe", "pb"]:
        values = table[column]
        score = rank_score(values.where(values > 0), higher_is_better=False)
        parts.append(score.mask(values <= 0, 0.0))
    return pd.concat(parts, axis=1).mean(axis=1) * 100


def quality_score(table: pd.DataFrame) -> pd.Series:
    """Higher ROE and profit margin, lower debt/equity -> higher score (0-100)."""
    parts = [
        rank_score(table["roe"], higher_is_better=True),
        rank_score(table["margin"], higher_is_better=True),
        rank_score(table["debt_to_equity"], higher_is_better=False),
    ]
    return pd.concat(parts, axis=1).mean(axis=1) * 100


# ---------- Combine ----------

def total_score(table: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted average of the three scores.

    If a stock is missing a score (e.g. no fundamentals), the remaining weights
    are scaled up instead of counting the missing part as zero.
    """
    parts = table[list(weights)]
    w = pd.Series(weights, dtype=float)
    weighted_sum = parts.mul(w, axis=1).sum(axis=1, min_count=1)
    used_weight = parts.notna().mul(w, axis=1).sum(axis=1)
    return weighted_sum / used_weight.replace(0, np.nan)


def rating(score: float) -> str:
    """Turn a 0-100 total score into a label: Strong, Positive, Neutral, Weak, Very weak."""
    if pd.isna(score):
        return "No data"
    for threshold, label in RATINGS:
        if score >= threshold:
            return label
    return RATINGS[-1][1]


def build_table(prices: pd.DataFrame, fundamentals: pd.DataFrame,
                weights: dict | None = None) -> pd.DataFrame:
    """One row per stock: indicators, projections, scores, rating and rank (best first)."""
    weights = weights or DEFAULT_WEIGHTS
    per_stock = {
        ticker: {**technical_signals(prices[ticker]), **projections(prices[ticker].dropna())}
        for ticker in prices.columns
    }
    technical = pd.DataFrame(per_stock).T.astype(float)

    table = technical.join(fundamentals.reindex(columns=FUNDAMENTAL_COLUMNS), how="left")
    table["valuation"] = valuation_score(table)
    table["quality"] = quality_score(table)
    table["total"] = total_score(table, weights)
    table["rating"] = table["total"].map(rating)

    table = table.sort_values("total", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table
