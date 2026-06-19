"""Plotting helpers for the portfolio analysis notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_efficient_frontier(
    random_portfolios: pd.DataFrame,
    efficient_frontier: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot random portfolios and the efficient frontier in daily units."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.scatter(
        random_portfolios["daily_volatility"],
        random_portfolios["daily_return"],
        c="lightgray",
        label="Simulated portfolios",
    )
    ax.scatter(
        efficient_frontier["daily_volatility"],
        efficient_frontier["daily_return"],
        c="red",
        label="Efficient frontier",
    )
    ax.set_xlabel("Daily volatility")
    ax.set_ylabel("Daily return")
    ax.legend()
    return ax


def plot_capital_market_line(
    annual_portfolios: pd.DataFrame,
    annual_frontier: pd.DataFrame,
    cml: pd.DataFrame,
    tangency_return_annual: float,
    tangency_volatility_annual: float,
    risk_free_rate_annual: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot annualized portfolios, efficient frontier, and Capital Market Line."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.scatter(
        annual_portfolios["annual_volatility"],
        annual_portfolios["annual_return"],
        c="lightgray",
        label="Simulated portfolios",
    )
    ax.scatter(
        annual_frontier["annual_volatility"],
        annual_frontier["annual_return"],
        c="red",
        label="Efficient frontier",
    )
    ax.plot(cml["annual_volatility"], cml["annual_return"], c="black", label="CML")
    ax.scatter(tangency_volatility_annual, tangency_return_annual, label="Tangency portfolio")
    ax.scatter(0, risk_free_rate_annual, label="Risk-free asset")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.legend()
    return ax


def plot_return_distribution(
    returns: pd.Series,
    historical_var_quantile: float,
    historical_cvar_tail_mean: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot portfolio returns with historical VaR and CVaR return levels."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    returns.hist(bins=40, ax=ax, color="lightgray")
    ax.axvline(historical_var_quantile, color="red", label="VaR return quantile")
    ax.axvline(historical_cvar_tail_mean, color="black", label="CVaR tail mean")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax

