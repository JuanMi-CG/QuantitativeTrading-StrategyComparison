# Quant Trading Library

A modular Python framework for backtesting, optimizing, and analyzing trading strategies.

## Features

* **Data Management**

  * Download, cache, and load historical market data via `yfinance`.
  * `DataManager` handles data retrieval and storage.
* **Trading Strategies**

  * Base class `TradingStrategy` for signal generation and backtesting.
  * Concrete strategies including:

    * Moving Average Crossover
    * Dollar-Cost Averaging (DCA)
    * Donchian Channel Breakout
    * ADX Trend Filter
    * Rate of Change (ROC)
    * MACD
    * Bollinger Band Mean Reversion
    * RSI
    * Pair Trading
    * VWAP
* **Risk Management**

  * Position sizing: fixed size, percentage of capital, ATR-based stop-loss.
* **Performance Analysis**

  * Key metrics: total/annual return, volatility, Sharpe ratio, max drawdown, win rate, profit factor, expectancy.
* **Optimization**

  * Grid search
  * Bayesian optimization (Optuna)
  * Differential Evolution (SciPy)
* **Reporting**

  * Generate CSV reports, tables, bar charts, equity curves, drawdowns, and rolling performance.
* **Trading System**

  * `TradingSystem` and `StrategyCollection` for orchestrating backtests across symbols and parameter sets.

## Installation

```bash
git clone https://github.com/JuanMi-CG/quant_trading.git
cd quant_trading
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Project Structure

```
├── trading_lib/                # Core library modules
│   ├── data_manager.py         # DataManager
│   ├── strategies/             # Strategy implementations
│   ├── risk_manager.py         # RiskManager
│   ├── performance_analyzer.py # PerformanceAnalyzer
│   ├── optimizer.py            # Optimizer (Grid, Bayesian, DE)
│   ├── report_manager.py       # ReportManager
│   └── trading_system.py       # TradingSystem & StrategyCollection
├── notebooks/                  # Jupyter notebooks
│   ├── Strategies comparison.ipynb
│   └── Testing Features.ipynb
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## Quickstart

```python
from trading_lib.data_manager import DataManager
from trading_lib.optimizer import Optimizer
from trading_lib.report_manager import ReportManager
from trading_lib.trading_system import TradingSystem
from trading_lib.strategies import MovingAverageCrossStrategy, DcaStrategy

# 1) Load data
dm = DataManager(data_dir="data", max_file_size=50)
data = dm.load_data(symbols="BTC-USD", period="2y", interval="1d")

# 2) Find best strategy & parameters
best_strat, best_params, perf_df, equity_map = Optimizer.find_best_strategy(
    strategies=[MovingAverageCrossStrategy, DcaStrategy],
    data=data,
    method="bayes",
    metric="Sharpe",
    n_trials=50,
    seed=42
)
print("Best:", best_strat.name, best_params)

# 3) Visualize and report
rm = ReportManager()
rm.plot_metrics(perf_df)
rm.plot_equity_curves(equity_map)
```

## Notebooks

* **Strategies comparison.ipynb**: Compare multiple strategies side-by-side.
* **Testing Features.ipynb**: Demonstrate and test individual components of the library.

## Contributing

1. Open an issue or discussion to propose enhancements or report bugs.
2. Fork the repository and create a new branch for your feature.
3. Implement changes and add tests where appropriate.
4. Submit a pull request and link to the related issue.

## Disclaimer

This framework is provided **as-is** for **educational and research purposes**. Not intended for live trading or financial advice. Use at your own risk.

---

© 2025 Juan Miguel Coll / Personal Project
