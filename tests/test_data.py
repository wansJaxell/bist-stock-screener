from data import parse_tickers, to_yahoo


def test_parse_tickers_cleans_input():
    assert parse_tickers(" thyao, asels ,,THYAO\nbimas ") == ["THYAO", "ASELS", "BIMAS"]


def test_parse_tickers_empty_text():
    assert parse_tickers("  ,  ") == []


def test_to_yahoo_adds_istanbul_suffix():
    assert to_yahoo("THYAO") == "THYAO.IS"
