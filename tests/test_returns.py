import numpy as np
import pandas as pd

from portfolio_risk.returns import (
    annualize_covariance,
    annualize_return,
    annualize_volatility,
    simple_returns,
)


def test_simple_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 55.0]})

    returns = simple_returns(prices)

    expected = pd.DataFrame(
        {"A": [0.10, 0.10], "B": [0.10, 0.00]},
        index=[1, 2],
    )
    pd.testing.assert_frame_equal(returns, expected)


def test_annualization_helpers():
    assert np.isclose(annualize_return(0.01, periods_per_year=2), 0.0201)
    assert np.isclose(annualize_volatility(0.02, periods_per_year=4), 0.04)

    cov = np.array([[0.01, 0.002], [0.002, 0.04]])
    np.testing.assert_allclose(annualize_covariance(cov, periods_per_year=2), cov * 2)

