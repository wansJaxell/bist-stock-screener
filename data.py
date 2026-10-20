"""Downloads prices and fundamentals for Borsa Istanbul stocks from Yahoo Finance."""

import pandas as pd
import yfinance as yf

# Our column name -> field name in Yahoo Finance's company info
FUNDAMENTAL_FIELDS = {
    "pe": "trailingPE",
    "pb": "priceToBook",
    "roe": "returnOnEquity",
    "margin": "profitMargins",
    "debt_to_equity": "debtToEquity",
}


def parse_tickers(text: str) -> list[str]:
    """'thyao, asels ,,THYAO' -> ['THYAO', 'ASELS'] (trimmed, upper-case, no duplicates)."""
    tickers = [t.strip().upper() for t in text.replace("\n", ",").split(",")]
    return list(dict.fromkeys(t for t in tickers if t))


def to_yahoo(ticker: str) -> str:
    """Yahoo Finance lists Borsa Istanbul stocks with a '.IS' suffix, e.g. THYAO.IS."""
    return f"{ticker}.IS"


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Daily closing prices: one column per ticker, one row per trading day."""
    data = yf.download([to_yahoo(t) for t in tickers], period=period,
                       auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()

    close = data["Close"]
    if isinstance(close, pd.Series):  # some yfinance versions return a Series for one ticker
        close = close.to_frame(to_yahoo(tickers[0]))
    close.columns = [str(c).removesuffix(".IS") for c in close.columns]
    return close.dropna(axis=1, how="all")  # drop tickers Yahoo returned nothing for


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Valuation and quality figures, one row per ticker (missing values become NaN)."""
    rows = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(to_yahoo(ticker)).info
        except Exception:  # network error or unknown ticker: keep the row, leave it empty
            info = {}
        rows[ticker] = {name: info.get(field) for name, field in FUNDAMENTAL_FIELDS.items()}

    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(FUNDAMENTAL_FIELDS))
    return table.apply(pd.to_numeric, errors="coerce")
