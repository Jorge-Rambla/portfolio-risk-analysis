"""Monte Carlo simulation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_risk.portfolios import validate_covariance, validate_weights


def simulate_univariate_portfolio(
    mean_daily_return: float,
    daily_volatility: float,
    initial_capital: float = 100.0,
    days: int = 252,
    n_simulations: int = 1_000,
    seed: int | None = None,
) -> pd.Series:
    """Simulate terminal capital using normally distributed portfolio returns."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(mean_daily_return, daily_volatility, size=(n_simulations, days))
    terminal_capital = initial_capital * np.prod(1 + daily_returns, axis=1)
    return pd.Series(terminal_capital, name="terminal_capital")


def simulate_multivariate_portfolio(
    mean_daily_returns: np.ndarray,
    daily_covariance: np.ndarray,
    weights: np.ndarray,
    initial_capital: float = 100.0,
    days: int = 252,
    n_simulations: int = 1_000,
    seed: int | None = None,
) -> pd.Series:
    """Simulate terminal capital using multivariate normal asset returns."""
    mean_daily_returns = np.asarray(mean_daily_returns, dtype=float)
    weights = validate_weights(weights, n_assets=len(mean_daily_returns))
    daily_covariance = validate_covariance(daily_covariance, len(mean_daily_returns))
    rng = np.random.default_rng(seed)

    asset_returns = rng.multivariate_normal(mean_daily_returns, daily_covariance, size=(n_simulations, days))
    portfolio_returns = asset_returns @ weights
    terminal_capital = initial_capital * np.prod(1 + portfolio_returns, axis=1)
    return pd.Series(terminal_capital, name="terminal_capital")

