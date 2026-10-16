"""Getting prices in: Yahoo Finance (optional, needs `pip install yfinance`) or a CSV file."""
from __future__ import annotations

import pandas as pd


def load_prices_csv(path: str) -> pd.DataFrame:
    """CSV with dates in the first column and one price column per ticker."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all").ffill().dropna()


def fetch_prices(tickers: list[str], start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Yahoo data needs one more package:  pip install yfinance")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all").ffill().dropna()
    if close.empty:
        raise SystemExit(f"No price data returned for {tickers}. Check the ticker symbols and your internet connection.")
    return close
