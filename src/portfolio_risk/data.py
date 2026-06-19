"""Market data download helpers."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    period: str = "5y",
    price_field: str = "Close",
) -> pd.DataFrame:
    """Download historical prices and return one column per ticker."""
    data = yf.download(" ".join(tickers), period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError("No price data was downloaded.")
    if price_field not in data.columns.get_level_values(0):
        raise ValueError(f"Price field {price_field!r} is not available.")

    prices = data[price_field].copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers)
    return prices.dropna(how="all")


def download_risk_free_rate(
    index_ticker: str = "^IRX",
    period: str = "5y",
    target_index: pd.Index | None = None,
) -> pd.Series:
    """Download the annualized risk-free proxy from Yahoo Finance.

    Yahoo reports ``^IRX`` as a percentage yield. The returned series is a
    decimal annualized rate, optionally aligned and forward-filled to a target
    price/return index.
    """
    rate = yf.Ticker(index_ticker).history(period=period)["Close"] / 100
    if rate.empty:
        raise ValueError(f"No data was downloaded for {index_ticker}.")
    rate.index = rate.index.tz_localize(None)
    rate.name = "risk_free_annual"
    if target_index is not None:
        rate = rate.reindex(target_index).ffill()
    return rate

