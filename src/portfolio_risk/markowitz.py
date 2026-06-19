"""Markowitz optimization and Capital Market Line utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize as opt

from portfolio_risk.portfolios import (
    portfolio_return,
    portfolio_volatility,
    validate_covariance,
    validate_weights,
)


def sharpe_ratio(
    portfolio_expected_return: float,
    portfolio_vol: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the Sharpe ratio for matching return/volatility units."""
    if portfolio_vol <= 0:
        raise ValueError("Portfolio volatility must be positive.")
    return float((portfolio_expected_return - risk_free_rate) / portfolio_vol)


def minimum_variance_portfolio(covariance: np.ndarray, long_only: bool = True) -> opt.OptimizeResult:
    """Find the long-only minimum variance portfolio."""
    n_assets = np.asarray(covariance).shape[0]
    validate_covariance(covariance, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets)) if long_only else None
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial_weights = np.ones(n_assets) / n_assets
    return opt.minimize(
        lambda w: portfolio_volatility(w, covariance),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def tangency_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
    long_only: bool = True,
) -> opt.OptimizeResult:
    """Find the portfolio with the highest Sharpe ratio."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = validate_covariance(covariance, len(expected_returns))
    bounds = tuple((0, 1) for _ in range(len(expected_returns))) if long_only else None
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

    def objective(weights: np.ndarray) -> float:
        ret = portfolio_return(weights, expected_returns)
        vol = portfolio_volatility(weights, covariance)
        return -sharpe_ratio(ret, vol, risk_free_rate)

    return opt.minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    n_points: int = 200,
    long_only: bool = True,
) -> pd.DataFrame:
    """Compute a constrained minimum-volatility frontier over target returns."""
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance = validate_covariance(covariance, len(expected_returns))
    bounds = tuple((0, 1) for _ in range(len(expected_returns))) if long_only else None
    initial_weights = np.ones(len(expected_returns)) / len(expected_returns)
    targets = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    rows = []

    for target_return in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target_return: w @ expected_returns - target},
        ]
        result = opt.minimize(
            lambda w: portfolio_volatility(w, covariance),
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            rows.append(
                {
                    "daily_return": target_return,
                    "daily_volatility": result.fun,
                    "weights": result.x,
                }
            )

    return pd.DataFrame(rows)


def capital_market_line(
    risk_free_rate_annual: float,
    tangency_return_annual: float,
    tangency_volatility_annual: float,
    n_points: int = 100,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    """Build Capital Market Line points from the computed tangency portfolio."""
    if tangency_volatility_annual <= 0:
        raise ValueError("Tangency volatility must be positive.")
    slope = sharpe_ratio(tangency_return_annual, tangency_volatility_annual, risk_free_rate_annual)
    volatilities = np.linspace(0, max_leverage * tangency_volatility_annual, n_points)
    returns = risk_free_rate_annual + slope * volatilities
    return pd.DataFrame({"annual_volatility": volatilities, "annual_return": returns})

