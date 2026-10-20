import math

import numpy as np
import pandas as pd
import pytest

from scoring import build_table, rating, rsi_signal, total_score, valuation_score


@pytest.mark.parametrize(
    "value, expected",
    [(55, 1.0), (40, 0.5), (68, 0.5), (25, 0.0), (80, 0.0)],
)
def test_rsi_signal_bands(value, expected):
    assert rsi_signal(value) == expected


def test_rsi_signal_missing_value():
    assert math.isnan(rsi_signal(np.nan))


def test_lower_pe_gets_higher_valuation_score():
    table = pd.DataFrame({"pe": [5.0, 10.0, 20.0], "pb": [1.0, 1.0, 1.0]}, index=["A", "B", "C"])
    scores = valuation_score(table)
    assert scores["A"] > scores["B"] > scores["C"]


def test_loss_making_company_gets_zero_for_pe():
    table = pd.DataFrame({"pe": [-4.0, 10.0], "pb": [np.nan, np.nan]}, index=["LOSS", "OK"])
    assert valuation_score(table)["LOSS"] == 0.0


def test_total_score_skips_missing_parts():
    # Quality is missing, so the total is the weighted average of the other two only
    table = pd.DataFrame({"technical": [80.0], "valuation": [40.0], "quality": [np.nan]})
    weights = {"technical": 50, "valuation": 50, "quality": 50}
    assert total_score(table, weights).iloc[0] == 60.0


def test_build_table_ranks_uptrend_above_downtrend():
    days = pd.bdate_range("2025-01-01", periods=250)
    prices = pd.DataFrame(
        {
            "UP": np.linspace(50, 100, 250),
            "DOWN": np.linspace(100, 50, 250),
        },
        index=days,
    )
    fundamentals = pd.DataFrame(index=["UP", "DOWN"])  # no fundamentals at all

    table = build_table(prices, fundamentals)

    assert list(table.index) == ["UP", "DOWN"]
    assert list(table["rank"]) == [1, 2]
    assert table.loc["UP", "technical"] > table.loc["DOWN", "technical"]
    assert table.loc["UP", "proj_1m"] > 0 > table.loc["DOWN", "proj_1m"]


@pytest.mark.parametrize(
    "score, expected",
    [(85, "Strong"), (70, "Strong"), (60, "Positive"), (50, "Neutral"),
     (35, "Weak"), (10, "Very weak"), (np.nan, "No data")],
)
def test_rating_labels(score, expected):
    assert rating(score) == expected
