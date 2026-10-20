import math

import pandas as pd

from indicators import pct_return, rsi, sma


def test_sma_is_mean_of_last_window():
    close = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(close, 3).iloc[-1] == 4.0  # (3 + 4 + 5) / 3


def test_rsi_is_100_when_price_only_rises():
    close = pd.Series(range(1, 31), dtype=float)
    assert rsi(close).iloc[-1] == 100.0


def test_rsi_is_0_when_price_only_falls():
    close = pd.Series(range(30, 0, -1), dtype=float)
    assert rsi(close).iloc[-1] == 0.0


def test_rsi_is_50_when_price_is_flat():
    close = pd.Series([10.0] * 30)
    assert rsi(close).iloc[-1] == 50.0


def test_pct_return():
    close = pd.Series([100, 110, 120], dtype=float)
    assert math.isclose(pct_return(close, 2), 20.0)


def test_pct_return_without_enough_data_is_nan():
    close = pd.Series([100, 110], dtype=float)
    assert math.isnan(pct_return(close, 5))
