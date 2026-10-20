"""Statistical 1- and 3-month price projections from the recent trend and volatility.

This is a scenario ("if the last 6 months' trend and volatility continue"),
not a prediction. News, market manipulation and other events are not modelled.
"""

import math

import numpy as np
import pandas as pd

HORIZONS = {"1m": 21, "3m": 63}  # trading days
LOOKBACK = 126                   # about 6 months of trading days
MIN_DAYS = 30                    # below this the trend estimate is too unreliable


def normal_cdf(x: float) -> float:
    """Probability that a standard normal value is below x."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def trend_and_volatility(close: pd.Series, lookback: int = LOOKBACK) -> tuple[float, float]:
    """Daily trend and daily volatility of the log price over the last `lookback` days.

    Trend = slope of a straight line fitted to log prices (less sensitive to the
    first and last day than a simple start-to-end return).
    Volatility = standard deviation of daily log returns.
    """
    close = close.dropna().tail(lookback)
    if len(close) < MIN_DAYS:
        return float("nan"), float("nan")

    log_price = np.log(close.to_numpy(dtype=float))
    slope = np.polyfit(np.arange(len(log_price)), log_price, 1)[0]
    volatility = np.diff(log_price).std(ddof=1)
    return float(slope), float(volatility)


def project(close: pd.Series, horizon: int, lookback: int = LOOKBACK) -> dict:
    """Expected % change after `horizon` trading days, a likely range and P(price higher).

    The range is +/- one standard deviation, which covers about 68% of outcomes
    if returns behave like a normal distribution.
    """
    trend, volatility = trend_and_volatility(close, lookback)
    if math.isnan(trend):
        return {"expected": np.nan, "low": np.nan, "high": np.nan, "prob_up": np.nan}

    mean = trend * horizon                      # expected log return
    spread = volatility * math.sqrt(horizon)    # uncertainty grows with sqrt(time)
    if spread > 0:
        prob_up = normal_cdf(mean / spread)
    else:                                       # no volatility at all: direction is certain
        prob_up = 1.0 if mean > 0 else 0.0 if mean < 0 else 0.5

    def to_pct(log_return):
        return (math.exp(log_return) - 1) * 100

    return {
        "expected": to_pct(mean),
        "low": to_pct(mean - spread),
        "high": to_pct(mean + spread),
        "prob_up": prob_up * 100,
    }


def projection_path(close: pd.Series, horizon: int = HORIZONS["3m"],
                    lookback: int = LOOKBACK) -> pd.DataFrame:
    """Day-by-day expected price and likely range for the chart (empty if not enough data)."""
    close = close.dropna()
    trend, volatility = trend_and_volatility(close, lookback)
    if math.isnan(trend):
        return pd.DataFrame(columns=["expected", "low", "high"])

    steps = np.arange(1, horizon + 1)
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    last_price = float(close.iloc[-1])
    spread = volatility * np.sqrt(steps)
    return pd.DataFrame(
        {
            "expected": last_price * np.exp(trend * steps),
            "low": last_price * np.exp(trend * steps - spread),
            "high": last_price * np.exp(trend * steps + spread),
        },
        index=dates,
    )
