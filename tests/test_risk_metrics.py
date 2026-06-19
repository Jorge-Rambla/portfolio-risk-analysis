import numpy as np
import pandas as pd

from portfolio_risk.risk_metrics import historical_cvar, historical_var, parametric_cvar, parametric_var


def test_historical_var_reports_positive_loss_and_return_quantile():
    returns = pd.Series([-0.10, -0.05, 0.0, 0.02, 0.04])

    loss, quantile = historical_var(returns, confidence_level=0.80)

    assert np.isclose(quantile, -0.06)
    assert np.isclose(loss, 0.06)


def test_historical_cvar_reports_positive_loss_and_tail_mean():
    returns = pd.Series([-0.10, -0.05, 0.0, 0.02, 0.04])

    loss, tail_mean = historical_cvar(returns, confidence_level=0.80)

    assert np.isclose(tail_mean, -0.10)
    assert np.isclose(loss, 0.10)


def test_parametric_var_reports_positive_loss_and_return_quantile():
    loss, quantile = parametric_var(0.001, 0.02, confidence_level=0.95)

    assert quantile < 0
    assert np.isclose(loss, -quantile)


def test_parametric_cvar_reports_positive_loss_and_tail_mean():
    loss, tail_mean = parametric_cvar(0.001, 0.02, confidence_level=0.95)

    assert tail_mean < 0
    assert np.isclose(loss, -tail_mean)
