"""Technical indicators calculated from a series of daily closing prices."""

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average: the mean of the last `window` closing prices."""
    return close.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's method), between 0 and 100.

    Above 70 is usually read as overbought, below 30 as oversold.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Wilder's smoothing is an exponential moving average with alpha = 1 / period
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss  # only gains -> division by zero -> inf -> RSI 100
    result = 100 - 100 / (1 + rs)

    # Completely flat prices give 0 / 0; treat that as neutral (50)
    flat = (avg_gain == 0) & (avg_loss == 0)
    return result.mask(flat, 50.0)


def pct_return(close: pd.Series, days: int) -> float:
    """Percentage change over the last `days` trading days (NaN if not enough data)."""
    close = close.dropna()
    if len(close) <= days:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)
