# Portfolio Risk Analysis

This project is a compact quantitative finance portfolio project built with Python. It downloads market data with `yfinance`, builds long-only portfolios, runs Monte Carlo simulations, estimates the Markowitz efficient frontier, computes the tangency portfolio and Capital Market Line, and reports VaR/CVaR risk metrics.

The code is intentionally simple and educational. Reusable logic lives in `src/portfolio_risk/`, while the notebook provides a clean narrative analysis.

## Objective

Analyze a four-asset portfolio made of:

- `SPY`
- `QQQ`
- `MSFT`
- `KO`

The analysis compares sample portfolios, simulated portfolios, the efficient frontier, the tangency portfolio, and downside risk metrics.

## Methodology

1. Download five years of daily close prices with `yfinance`.
2. Compute simple daily returns.
3. Build example long-only portfolios.
4. Estimate daily mean returns and the daily covariance matrix.
5. Run univariate and multivariate Monte Carlo simulations.
6. Generate random long-only portfolios.
7. Compute the constrained Markowitz efficient frontier.
8. Download `^IRX` as an annualized risk-free rate proxy.
9. Convert quantities explicitly between daily and annualized units.
10. Optimize the annualized tangency portfolio.
11. Build the Capital Market Line from the computed tangency portfolio.
12. Compute historical and parametric VaR/CVaR.

## Risk Metric Convention

VaR and CVaR are reported as positive losses.

The underlying return quantiles and tail means are also documented in the notebook. For example, a daily return quantile of `-0.025` is reported as a VaR loss of `0.025`.

## Assumptions

- The portfolio is long-only.
- Portfolio weights must sum to 1.
- Simple returns are used as the main return measure.
- One trading year has 252 trading days.
- Monte Carlo returns are sampled from normal or multivariate normal distributions.
- `^IRX` is used as a risk-free proxy and interpreted as an annualized decimal yield after dividing Yahoo's percentage quote by 100.
- Close prices are used consistently through the analysis.

## Limitations

- Historical estimates are sample-based and sensitive to the selected period.
- Normal Monte Carlo and parametric VaR/CVaR do not fully capture fat tails, skew, volatility clustering, or regime changes.
- The optimizer is constrained to long-only weights.
- Transaction costs, taxes, dividends, rebalancing rules, and liquidity constraints are not modeled.
- Results may change when market data is refreshed.

## Project Structure

```text
portfolio-risk-analysis/
├── README.md
├── requirements.txt
├── portfolio_risk_analysis.ipynb
├── src/
│   └── portfolio_risk/
│       ├── __init__.py
│       ├── data.py
│       ├── returns.py
│       ├── portfolios.py
│       ├── monte_carlo.py
│       ├── markowitz.py
│       ├── risk_metrics.py
│       └── plotting.py
└── tests/
    ├── test_returns.py
    ├── test_portfolios.py
    ├── test_markowitz.py
    └── test_risk_metrics.py
```

## How To Run

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
PYTHONPATH=src pytest
```

Open the notebook:

```bash
jupyter notebook portfolio_risk_analysis.ipynb
```

On Windows PowerShell, run tests with:

```powershell
$env:PYTHONPATH = "src"
pytest
```

## Main Results

The notebook computes the main results from downloaded data rather than hardcoding them:

- annualized expected returns,
- annualized covariance matrix,
- minimum variance portfolio,
- tangency portfolio weights,
- tangency portfolio annualized return and volatility,
- Sharpe ratio,
- Capital Market Line,
- historical VaR and CVaR,
- parametric VaR and CVaR.

Because the data is downloaded live from Yahoo Finance, exact numerical results can vary over time.
