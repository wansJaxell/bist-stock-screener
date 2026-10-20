import math

import numpy as np
import pandas as pd

from projection import normal_cdf, project, projection_path


def make_prices(daily_log_return: float, days: int = 150, noise: float = 0.0, seed: int = 1):
    """Synthetic price series with a fixed daily trend and optional random noise."""
    rng = np.random.default_rng(seed)
    steps = daily_log_return + rng.normal(0, noise, days) if noise else np.full(days, daily_log_return)
    index = pd.bdate_range("2025-01-01", periods=days)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=index)


def test_normal_cdf_midpoint():
    assert normal_cdf(0) == 0.5


def test_smooth_uptrend_projects_trend_forward():
    # +0.1% per day with no noise -> after 21 days exactly exp(0.021) - 1
    result = project(make_prices(0.001), horizon=21)
    assert math.isclose(result["expected"], (math.exp(0.021) - 1) * 100, rel_tol=1e-6)
    assert result["prob_up"] == 100.0


def test_downtrend_has_low_probability_of_rise():
    result = project(make_prices(-0.002, noise=0.01), horizon=21)
    assert result["expected"] < 0
    assert result["prob_up"] < 50


def test_longer_horizon_has_wider_range():
    close = make_prices(0.0005, noise=0.02)
    one_month = project(close, horizon=21)
    three_months = project(close, horizon=63)
    assert (three_months["high"] - three_months["low"]) > (one_month["high"] - one_month["low"])


def test_not_enough_data_gives_nan():
    result = project(make_prices(0.001, days=10), horizon=21)
    assert math.isnan(result["expected"])


def test_projection_path_has_one_row_per_future_day():
    close = make_prices(0.0005, noise=0.02)
    path = projection_path(close, horizon=63)
    assert len(path) == 63
    assert path.index[0] > close.index[-1]
    assert (path["low"] <= path["expected"]).all() and (path["expected"] <= path["high"]).all()
