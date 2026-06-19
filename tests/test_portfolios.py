import numpy as np
import pandas as pd

from portfolio_risk.portfolios import portfolio_return, portfolio_return_series, portfolio_volatility


def test_portfolio_return_and_volatility():
    weights = np.array([0.6, 0.4])
    expected_returns = np.array([0.01, 0.02])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    assert np.isclose(portfolio_return(weights, expected_returns), 0.014)
    assert np.isclose(portfolio_volatility(weights, covariance), np.sqrt(0.0348))


def test_portfolio_return_series():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.01]})
    weights = np.array([0.25, 0.75])

    result = portfolio_return_series(returns, weights)

    expected = pd.Series([0.025, -0.0025])
    pd.testing.assert_series_equal(result, expected)

