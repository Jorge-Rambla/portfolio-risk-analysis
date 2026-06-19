"""Portfolio construction and validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(
    weights: np.ndarray | pd.Series,
    n_assets: int | None = None,
    long_only: bool = True,
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate and return portfolio weights as a one-dimensional array."""
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array.")
    if n_assets is not None and len(weights_array) != n_assets:
        raise ValueError(f"Expected {n_assets} weights, got {len(weights_array)}.")
    if long_only and np.any(weights_array < -atol):
        raise ValueError("Long-only portfolio weights cannot be negative.")
    if not np.isclose(weights_array.sum(), 1.0, atol=atol):
        raise ValueError("Portfolio weights must sum to 1.")
    return weights_array


def validate_covariance(covariance: pd.DataFrame | np.ndarray, n_assets: int) -> np.ndarray:
    """Validate covariance matrix shape and return it as an array."""
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance matrix must have shape {(n_assets, n_assets)}.")
    return covariance_array


def portfolio_return(weights: np.ndarray | pd.Series, expected_returns: np.ndarray | pd.Series) -> float:
    """Compute portfolio expected return."""
    expected_returns_array = np.asarray(expected_returns, dtype=float)
    weights_array = validate_weights(weights, n_assets=len(expected_returns_array))
    return float(weights_array @ expected_returns_array)


def portfolio_volatility(weights: np.ndarray | pd.Series, covariance: pd.DataFrame | np.ndarray) -> float:
    """Compute portfolio volatility from a covariance matrix."""
    covariance_array = np.asarray(covariance, dtype=float)
    weights_array = validate_weights(weights, n_assets=covariance_array.shape[0])
    validate_covariance(covariance_array, len(weights_array))
    return float(np.sqrt(weights_array @ covariance_array @ weights_array))


def portfolio_return_series(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    columns: list[str] | None = None,
) -> pd.Series:
    """Compute a historical portfolio return series."""
    if returns.isna().any().any():
        raise ValueError("Return data contains missing values.")
    selected_returns = returns if columns is None else returns.loc[:, columns]
    weights_array = validate_weights(weights, n_assets=selected_returns.shape[1])
    return selected_returns @ weights_array


def random_long_only_weights(
    n_assets: int,
    n_portfolios: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random long-only weights that sum to 1."""
    rng = np.random.default_rng(seed)
    raw_weights = rng.random((n_portfolios, n_assets))
    return raw_weights / raw_weights.sum(axis=1, keepdims=True)


def random_long_only_portfolios(
    expected_returns: np.ndarray | pd.Series,
    covariance: pd.DataFrame | np.ndarray,
    asset_names: list[str],
    n_portfolios: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random long-only portfolios and their return/volatility."""
    expected_returns_array = np.asarray(expected_returns, dtype=float)
    covariance_array = validate_covariance(covariance, len(expected_returns_array))
    weights = random_long_only_weights(len(expected_returns_array), n_portfolios, seed)

    portfolio_returns = weights @ expected_returns_array
    portfolio_volatilities = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance_array, weights))
    portfolios = pd.DataFrame(weights, columns=asset_names)
    portfolios.insert(0, "daily_return", portfolio_returns)
    portfolios.insert(1, "daily_volatility", portfolio_volatilities)
    portfolios["sharpe_ratio"] = portfolios["daily_return"] / portfolios["daily_volatility"]
    return portfolios

