import numpy as np

from portfolio_risk.markowitz import sharpe_ratio


def test_sharpe_ratio():
    assert np.isclose(sharpe_ratio(0.12, 0.20, 0.02), 0.5)

