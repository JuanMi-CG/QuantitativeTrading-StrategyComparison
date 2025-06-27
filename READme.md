# Trading Environment

A modular Python framework for backtesting, optimizing, and analyzing trading strategies. This repository provides:

* **DataManager**: download, cache, and load historical market data.
* **TradingStrategy** base class & concrete strategies: MA crossover, DCA, Donchian breakout, ADX trend filter, ROC, MACD, Bollinger mean reversion, RSI, pair trading, VWAP.
* **RiskManager**: flexible position sizing (fixed, percentage of capital, ATR-based).
* **PerformanceAnalyzer**: compute key metrics (total/annual return, volatility, Sharpe, drawdown, win rate, profit factor, expectancy).
* **Optimizer**: grid search, Bayesian (Optuna), and differential evolution hyperparameter tuning.
* **ReportManager**: generate tables, bar charts, equity curves, drawdowns, rolling metrics, and save CSV reports.
* **TradingSystem** & **StrategyCollection**: orchestrate loading data, running backtests, and aggregating multiple strategies.

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-org/trading-environment.git
   cd trading-environment
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── data/                # cached CSV data
├── strategies/          # saved strategy pickles
├── reports/             # output CSV and figures
├── trading_environment.py
├── strategies.py        # concrete strategy implementations
└── notebook.ipynb       # example usage
```

## Quickstart

```python
from trading_environment import (
    DataManager, Optimizer, ReportManager, TradingSystem
)
from strategies import MovingAverageCrossStrategy, DcaStrategy, ...

# 1) Load data
dm   = DataManager(data_dir=DATA_DIR, max_file_size=MAX_FILE_SIZE)
data = dm.load_data(symbols='BTC-USD', period='2y', interval='1d')

# 2) Find best strategy & parameters
top_strat, top_params, perf_df, equity_map = Optimizer.find_best_strategy(
    strategies=[MovingAverageCrossStrategy, DcaStrategy, ...],
    data=data,
    method='bayes', metric='Sharpe', n_trials=50, seed=42
)
print("Best:", top_strat.name, top_params)
print(perf_df)

# 3) Visualize results
rm = ReportManager()
rm.plot_metrics(perf_df)
rm.plot_equity_curves(equity_map)
```

## Components

### DataManager

* Downloads via `yfinance`, caches splits under `data/`.
* Returns a MultiIndex `DataFrame` (`date`, `ticker`).

### TradingStrategy

* Base class: `generate_signals(data) -> Series` and `backtest(...)` => equity curve.
* Each concrete strategy defines a `param_config` for auto‐tuning.

### RiskManager

* `fixed`: static position size.
* `pct`: trade size = `risk_pct` \* capital / price.
* `atr`: uses Average True Range and a stop-loss level.

### PerformanceAnalyzer

* `summary()` returns Series: Total Return, Ann. Return/Vol, Sharpe, Max Drawdown, Win Rate, Profit Factor, Expectancy.

### Optimizer

* **Grid**: brute‐force all combinations.
* **Bayes**: Optuna TPE, returns `(best_params, best_value, study)`.
* **DE**: SciPy Differential Evolution.
* **find\_best\_strategy** (updated): returns `(best_strategy, best_params, perf_df, equity_map)`.

### ReportManager

* `plot_metrics`: bar charts per metric.
* `plot_equity_curves`: overlay equity lines.
* `plot_price_and_indicators`, `plot_signals`, `plot_performance`.
* Save/load CSV reports under `reports/`.

### TradingSystem & StrategyCollection

* `TradingSystem`: per-symbol backtest runner combining a strategy+risk manager.
* `StrategyCollection`: generate & batch-backtest many param combos, save strategies.

## Logging and Verbosity

```python
import logging
# suppress INFO logs (cache messages)
logging.getLogger().setLevel(logging.WARNING)

import optuna
from optuna.logging import set_verbosity, ERROR
# suppress Optuna trial warnings
set_verbosity(ERROR)
```

## Disclaimer of Responsibility

This framework is provided **as-is** for **educational and research purposes only**. It is **not** intended for live trading, financial advice, or any production use. **Use at your own risk.** The authors and contributors accept no liability for any losses or damages arising from its use.

## Contributing

1. Create an issue or discussion outlining your idea.
2. Fork the repo & create a feature branch.
3. Write tests where applicable.
4. Submit a PR and reference the issue.

---

© 2025 Juan Miguel Coll / Personal Project

