"""Return calculations and annualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage returns from prices."""
    if prices.isna().all().any():
        raise ValueError("At least one price column contains only missing values.")
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from prices."""
    if (prices <= 0).any().any():
        raise ValueError("Log returns require strictly positive prices.")
    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualize_return(
    daily_return: float | np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | np.ndarray | pd.Series:
    """Compound a daily return into an annualized return."""
    return (1 + daily_return) ** periods_per_year - 1


def deannualize_return(
    annual_return: float | np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | np.ndarray | pd.Series:
    """Convert an annualized return into an equivalent daily return."""
    return (1 + annual_return) ** (1 / periods_per_year) - 1


def annualize_volatility(
    daily_volatility: float | np.ndarray | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | np.ndarray | pd.Series:
    """Scale daily volatility into annualized volatility."""
    return daily_volatility * np.sqrt(periods_per_year)


def annualize_covariance(
    daily_covariance: pd.DataFrame | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame | np.ndarray:
    """Scale a daily covariance matrix into annualized covariance."""
    return daily_covariance * periods_per_year

