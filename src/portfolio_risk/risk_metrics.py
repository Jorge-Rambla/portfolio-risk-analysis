"""Portfolio risk metrics reported as positive losses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return historical VaR as a positive loss and the underlying return quantile."""
    if returns.isna().any():
        raise ValueError("Returns contain missing values.")
    tail_probability = 1 - confidence_level
    return_quantile = float(returns.quantile(tail_probability))
    return max(0.0, -return_quantile), return_quantile


def parametric_var(
    mean_return: float,
    volatility: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return normal VaR as a positive loss and the underlying return quantile."""
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")
    tail_probability = 1 - confidence_level
    return_quantile = float(mean_return + volatility * stats.norm.ppf(tail_probability))
    return max(0.0, -return_quantile), return_quantile


def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return historical CVaR as a positive loss and the underlying tail mean."""
    _, return_quantile = historical_var(returns, confidence_level)
    tail = returns[returns <= return_quantile]
    if tail.empty:
        raise ValueError("No observations found in the historical tail.")
    tail_mean = float(tail.mean())
    return max(0.0, -tail_mean), tail_mean


def parametric_cvar(
    mean_return: float,
    volatility: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return normal CVaR as a positive loss and the underlying tail mean."""
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")
    tail_probability = 1 - confidence_level
    z_score = stats.norm.ppf(tail_probability)
    tail_mean = float(mean_return - volatility * stats.norm.pdf(z_score) / tail_probability)
    return max(0.0, -tail_mean), tail_mean

