from imports import *

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directorios principales
DATA_DIR = Path('data')
STRAT_DIR = Path('strategies')
REPORT_DIR = Path('reports')
for d in (DATA_DIR, STRAT_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Límite de tamaño de archivo (500 MB por defecto)
MAX_FILE_SIZE = 500 * 1024**2

def _extract_dates(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Extrae nivel de fecha de un índice, manejando MultiIndex.
    """
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(0))
    else:
        return pd.to_datetime(idx)

class DataManager:
    """
    Gestor de caché de datos: descarga y cache particionado CSV en data/.
    Devuelve siempre MultiIndex ['date','ticker'].
    """
    def __init__(self,
                 data_dir: Path,
                 max_file_size: int):
        self.data_dir = data_dir
        self.max_file_size = max_file_size

    def load_data(self,
                  symbols: Union[str, List[str]],
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  period: Optional[str] = None,
                  interval: Optional[str] = None) -> pd.DataFrame:
        # Normalizar lista de símbolos
        if isinstance(symbols, str):
            sym_list = [symbols]
        else:
            sym_list = symbols
        basename = '_'.join(filter(None, [
            '-'.join(sym_list), start or '', end or '', period or '', interval or ''
        ]))
        parts = sorted(self.data_dir.glob(f"{basename}_part*.csv"))
        if parts:
            dfs = [pd.read_csv(p, index_col=[0,1], parse_dates=[0]) for p in parts]
            df = pd.concat(dfs).sort_index()
            logging.info(f"Cargados {len(df)} registros de caché ({len(parts)} archivos)")
            return df

        logging.info(f"Descargando tickers={sym_list}, start={start}, end={end}, period={period}, interval={interval}")
        group_by = 'ticker' if len(sym_list) > 1 else 'column'
        raw = yf.download(
            tickers=sym_list if len(sym_list) > 1 else sym_list[0],
            start=start,
            end=end,
            period=period,
            interval=interval,
            progress=False,
            group_by=group_by,
            auto_adjust=False
        )

        # Procesar raw a DataFrame largo con MultiIndex ['date','ticker']
        if isinstance(raw.columns, pd.MultiIndex):
            if len(sym_list) > 1:
                df = raw.stack(level=0)
            else:
                df = raw.copy()
                df.columns = df.columns.droplevel(1)
                df.index.name = 'date'
                df = df.reset_index()
                df['ticker'] = sym_list[0]
                df = df.set_index(['date', 'ticker'])
        else:
            df = raw.copy()
            df = df.reset_index()
            df['ticker'] = sym_list[0]
            df = df.set_index(['date', 'ticker'])

        df.columns = df.columns.str.lower()
        self._save_split(df, basename)
        return df

    def _save_split(self, df: pd.DataFrame, basename: str) -> None:
        n = len(df)
        mem = df.memory_usage(deep=True).sum()
        avg = mem / (n or 1)
        chunk = max(int(self.max_file_size / avg), 1)
        for i in range(0, n, chunk):
            part = df.iloc[i:i + chunk]
            path = self.data_dir / f"{basename}_part{(i // chunk) + 1}.csv"
            part.to_csv(path)
            logging.info(f"Guardado caché {path} ({len(part)} filas)")

# --- RiskManager ---
class RiskManager:
    """
    Tamaño de posición: fixed, pct, atr.
    """
    def __init__(self,
                 method: str = 'fixed',
                 fixed_size: float = 1.0,
                 risk_pct: float = 0.01,
                 atr_window: int = 14):
        self.method = method
        self.fixed_size = fixed_size
        self.risk_pct = risk_pct
        self.atr_window = atr_window

    def calculate_size(self,
                       capital: float,
                       price: float,
                       data: pd.DataFrame,
                       stop_loss: Optional[float] = None) -> float:
        if self.method == 'fixed':
            return self.fixed_size
        if self.method == 'pct':
            return (capital * self.risk_pct) / price
        if self.method == 'atr':
            if stop_loss is None:
                raise ValueError('Se requiere stop_loss para ATR')
            atr = self._compute_atr(data)
            return (capital * self.risk_pct) / abs(price - stop_loss)
        raise ValueError(f"Método desconocido {self.method}")

    def _compute_atr(self, data: pd.DataFrame) -> float:
        high, low, close = data['high'], data['low'], data['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean().iloc[-1]

# --- TradingStrategy base ---
class TradingStrategy:
    def __init__(self, name: str, price_col: str = 'close'):
        self.name = name
        self.price_col = price_col
        self.positions: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.equity_curve: Optional[pd.Series] = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def backtest(self,
                 data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.0,
                 risk_manager: Optional[RiskManager] = None,
                 stop_loss: Optional[float] = None) -> pd.Series:
        # make initial_capital available to generate_signals (for DCA)
        self._initial_capital = initial_capital

        price = data[self.price_col]
        sig   = self.generate_signals(data)
        pos   = sig.shift(1).fillna(0)
        ret   = price.pct_change().fillna(0)
        strat_ret = pos * ret - sig.diff().abs().fillna(0) * transaction_cost
        equity = initial_capital * (1 + strat_ret).cumprod()

        self.positions = sig
        self.returns   = strat_ret
        self.equity_curve = equity
        return equity

# --- PerformanceAnalyzer ---
class PerformanceAnalyzer:
    def __init__(self, equity: pd.Series, returns: pd.Series):
        self.equity, self.returns = equity, returns.dropna()

    def summary(self) -> pd.Series:
        tot = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        days = len(self.returns)
        ann = (1 + tot)**(252/days) - 1 if days else np.nan
        vol = self.returns.std() * np.sqrt(252)
        sr  = self.returns.mean()/self.returns.std()*np.sqrt(252) if self.returns.std() else np.nan
        dd  = (self.equity - self.equity.cummax())/self.equity.cummax()
        maxdd = dd.min()
        wr = (self.returns>0).mean()
        pf = self.returns[self.returns>0].sum()/ -self.returns[self.returns<0].sum() if (self.returns<0).any() else np.nan
        exp = wr * (self.returns[self.returns>0].mean() if (self.returns>0).any() else 0) - \
              (1-wr) * (-self.returns[self.returns<0].mean() if (self.returns<0).any() else 0)
        return pd.Series({
            'Total Return': tot, 'Ann. Return': ann, 'Ann. Vol': vol,
            'Sharpe': sr, 'Max Drawdown': maxdd,
            'Win Rate': wr, 'Profit Factor': pf, 'Expectancy': exp
        })



import logging
from itertools import product
from typing import Type, Dict, Any, List, Tuple, Optional

import pandas as pd
from scipy.optimize import differential_evolution
import optuna

from trading_environment import TradingStrategy, PerformanceAnalyzer

class Optimizer:
    """
    Three optimization routines over your strategy’s parameter space,
    plus helpers to extract best params and build the best strategy safely.
    """

    def __init__(
        self,
        strategy_cls: Type[TradingStrategy],
        param_grid: Dict[str, List[Any]]
    ):
        self.strategy_cls = strategy_cls
        self.param_grid   = param_grid

    def optimize_grid(
        self,
        data: pd.DataFrame,
        metric: str = 'Sharpe'
    ) -> pd.DataFrame:
        """Brute‐force grid search."""
        records: List[Dict[str, Any]] = []
        for vals in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), vals))
            try:
                strat = self.strategy_cls(**params)
                eq    = strat.backtest(data)
                perf  = PerformanceAnalyzer(eq, strat.returns).summary().to_dict()
                perf.update(params)
                records.append(perf)
            except Exception as e:
                logging.warning(f"Grid skip {params}: {e}")
        if not records:
            raise ValueError(f"No valid grid combinations for {self.strategy_cls.__name__}")
        return pd.DataFrame(records).sort_values(by=metric, ascending=False).reset_index(drop=True)

    def optimize_bayesian(
        self,
        data: pd.DataFrame,
        metric: str = 'Sharpe',
        n_trials: int = 50,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float, optuna.Study]:
        """Bayesian optimization via Optuna."""
        def _objective(trial: optuna.Trial) -> float:
            kwargs: Dict[str, Any] = {}
            for k, choices in self.param_grid.items():
                if all(isinstance(c, int) for c in choices):
                    low, high = min(choices), max(choices)
                    kwargs[k] = trial.suggest_int(k, low, high)
                elif all(isinstance(c, float) for c in choices):
                    low, high = min(choices), max(choices)
                    kwargs[k] = trial.suggest_float(k, low, high)
                else:
                    kwargs[k] = trial.suggest_categorical(k, choices)
            strat = self.strategy_cls(**kwargs)
            eq    = strat.backtest(data)
            perf  = PerformanceAnalyzer(eq, strat.returns).summary()[metric]
            return -perf  # minimize negative Sharpe

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(_objective, n_trials=n_trials)

        best_params = study.best_params
        best_metric = -study.best_value
        return best_params, best_metric, study

    def optimize_de(
        self,
        data: pd.DataFrame,
        metric: str = 'Sharpe',
        maxiter: int = 30,
        popsize: int = 10
    ) -> Tuple[Dict[str, Any], float, object]:
        """Differential Evolution via SciPy."""
        keys = list(self.param_grid.keys())
        bounds: List[Tuple[float,float]] = []
        int_flags: List[bool] = []
        for k in keys:
            vals = self.param_grid[k]
            low, high = min(vals), max(vals)
            bounds.append((low, high))
            int_flags.append(all(isinstance(v, int) for v in vals))

        def _func(x: List[float]) -> float:
            kwargs: Dict[str, Any] = {}
            for xi, k, is_int in zip(x, keys, int_flags):
                kwargs[k] = int(round(xi)) if is_int else float(xi)
            try:
                strat = self.strategy_cls(**kwargs)
                eq    = strat.backtest(data)
                perf  = PerformanceAnalyzer(eq, strat.returns).summary()[metric]
                return -perf
            except Exception:
                return 1e6

        result = differential_evolution(_func, bounds, maxiter=maxiter, popsize=popsize)
        best_x = result.x

        best_kwargs: Dict[str, Any] = {}
        for xi, k, is_int in zip(best_x, keys, int_flags):
            best_kwargs[k] = int(round(xi)) if is_int else float(xi)

        best_strat  = self.strategy_cls(**best_kwargs)
        best_eq     = best_strat.backtest(data)
        # best_metric = PerformanceAnalyzer(best_eq, best_strategies.returns).summary()[metric]
        best_metric = PerformanceAnalyzer(best_eq, best_strat.returns).summary()[metric]

        return best_kwargs, best_metric, result

    def best_params(
        self,
        results: pd.DataFrame,
        idx: int = 0
    ) -> Dict[str, Any]:
        """Extract native‐typed best parameter combo from a results DataFrame."""
        row = results.iloc[idx]
        out: Dict[str, Any] = {}
        for k in self.param_grid.keys():
            v = row[k]
            if hasattr(v, 'item'):
                v = v.item()
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            out[k] = v
        return out

    def best_strategy(
        self,
        data: pd.DataFrame,
        metric: str = 'Sharpe',
        idx: int = 0
    ) -> Tuple[Optional[TradingStrategy], Optional[pd.Series], Optional[pd.DataFrame]]:
        """
        1) Run grid search
        2) Pick best params
        3) Safely instantiate & backtest that single strategy
        Returns (strat, equity, results_df) or (None, None, results_df) on failure.
        """
        results = self.optimize_grid(data, metric)
        params  = self.best_params(results, idx)

        try:
            strat = self.strategy_cls(**params)
            eq    = strat.backtest(data)
        except Exception as e:
            logging.warning(f"best_strategy failed for {params}: {e}")
            return None, None, results

        return strat, eq, results



class StrategyManager:
    """
    Guarda y carga estrategias en strategies/ como pickle.
    """
    def __init__(self, directory: Path = STRAT_DIR):
        self.directory = directory

    def save(self, strategy: TradingStrategy) -> None:
        fname = f"{strategy.name.replace(' ', '_').replace('/', '_')}.pkl"
        path  = self.directory / fname
        with open(path, 'wb') as f:
            pickle.dump(strategy, f)
        logging.info(f"Estrategia '{strategy.name}' guardada en {path}")

    def load(self, name: str) -> TradingStrategy:
        fname = f"{name.replace(' ', '_').replace('/', '_')}.pkl"
        path  = self.directory / fname
        if not path.exists():
            raise FileNotFoundError(f"No existe estrategia guardada con nombre '{name}'")
        with open(path, 'rb') as f:
            strat = pickle.load(f)
        logging.info(f"Estrategia '{name}' cargada desde {path}")
        return strat

class ReportManager:
    def __init__(self, directory: Path = REPORT_DIR):
        self.directory = directory

    def load(self, report_name: str) -> pd.Series:
        path = self.directory / f"{report_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Reporte '{report_name}' no encontrado")
        df = pd.read_csv(path, index_col=0)
        return df.squeeze("columns")

    def list_reports(self) -> List[str]:
        return [p.stem for p in self.directory.glob("*.csv")]

    def plot_price_and_indicators(self,
                                  data: pd.DataFrame,
                                  short_window: int = 20,
                                  long_window: int = 50,
                                  price_col: str = 'close',
                                  benchmark: Optional[pd.Series] = None):
        price = data[price_col]
        dates = _extract_dates(price.index)
        sma_s = price.rolling(short_window).mean()
        sma_l = price.rolling(long_window).mean()

        plt.figure()
        plt.plot(dates, price.values, label='Price')
        plt.plot(dates, sma_s.values, label=f'SMA{short_window}')
        plt.plot(dates, sma_l.values, label=f'SMA{long_window}')
        if benchmark is not None:
            bdates = _extract_dates(benchmark.index)
            plt.plot(bdates, benchmark.values, label='Benchmark')
        plt.title('Price & Moving Averages')
        plt.legend()
        plt.show()

    def plot_signals(self,
                     data: pd.DataFrame,
                     signals: pd.Series,
                     price_col: str = 'close'):
        price = data[price_col]
        dates = _extract_dates(price.index)
        plt.figure()
        plt.plot(dates, price.values, label='Price')

        buys = signals[signals==1].index
        buys_dates = _extract_dates(buys)
        buys_vals = price.loc[buys].values
        plt.scatter(buys_dates, buys_vals, marker='^', s=100, label='Buy')

        sells = signals[signals==-1].index
        sells_dates = _extract_dates(sells)
        sells_vals = price.loc[sells].values
        plt.scatter(sells_dates, sells_vals, marker='v', s=100, label='Sell')

        plt.title('Trading Signals')
        plt.legend()
        plt.show()

    def plot_performance(self,
                         equity: pd.Series,
                         returns: pd.Series,
                         benchmark_equity: Optional[pd.Series] = None,
                         rolling_window: int = 30):
        edates = _extract_dates(equity.index)
        plt.figure()
        plt.plot(edates, equity.values, label='Strategy Equity')
        if benchmark_equity is not None:
            bdates = _extract_dates(benchmark_equity.index)
            plt.plot(bdates, benchmark_equity.values, label='Benchmark Equity')
        plt.title('Equity Curve')
        plt.legend()
        plt.show()

        dd = (equity - equity.cummax()) / equity.cummax()
        ddates = _extract_dates(dd.index)
        plt.figure()
        plt.plot(ddates, dd.values)
        plt.title('Drawdown')
        plt.show()

        sr = returns.rolling(rolling_window).mean() / returns.rolling(rolling_window).std() * np.sqrt(252)
        vol = returns.rolling(rolling_window).std() * np.sqrt(252)
        rdates = _extract_dates(sr.index)
        plt.figure()
        plt.plot(rdates, sr.values, label='Rolling Sharpe')
        plt.plot(rdates, vol.values, label='Rolling Vol')
        plt.title(f'Rolling Metrics ({rolling_window} periods')
        plt.legend()
        plt.show()

    def compare_performance(
        self,
        analyzers: Dict[str, PerformanceAnalyzer],
        save: bool = False,
        report_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute a combined performance table for N strategies.

        Parameters:
        - analyzers: dict mapping strategy name to PerformanceAnalyzer
        - save: whether to save the table to CSV
        - report_name: filename (without .csv) to save under REPORT_DIR

        Returns:
        - DataFrame: rows are strategy names, columns are metrics
        """
        # build summary Series for each
        records = {}
        for name, pa in analyzers.items():
            rec = pa.summary()
            rec.name = name
            records[name] = rec
        df = pd.DataFrame(records).T

        if save:
            fname = report_name or "comparison_report"
            path = self.directory / f"{fname}.csv"
            df.to_csv(path)
            logging.info(f"Saved comparison report to {path}")

        print("Combined Performance Report:")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        return df

    def plot_metrics(
        self,
        perf_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Plot bar charts of selected metrics, one plot per metric.

        - perf_df: DataFrame from compare_performance
        - metrics: list of metric column names; defaults to key set
        """
        if metrics is None:
            metrics = [
                'Total Return', 'Sharpe', 'Max Drawdown',
                'Win Rate', 'Profit Factor'
            ]
        n = len(metrics)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        for ax, metric in zip(axes, metrics):
            if metric not in perf_df.columns:
                continue
            perf_df[metric].plot(
                kind='bar',
                ax=ax,
                legend=False
            )
            ax.set_title(metric)
        # hide any unused axes
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.suptitle("Strategy Metrics Comparison")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_equity_curves(
        self,
        equity_dict: Dict[str, pd.Series],
        figsize: tuple = (10, 5)
    ) -> None:
        """
        Overlay equity curves of multiple strategies.

        - equity_dict: mapping name to equity Series
        """
        plt.figure(figsize=figsize)
        for name, eq in equity_dict.items():
            dates = _extract_dates(eq.index)
            plt.plot(dates, eq.values, label=name)
        plt.title("Equity Curves Comparison")
        plt.legend()
        plt.show()

    def save_summary(self, name: str, summary: pd.Series) -> None:
        path = self.directory / f"{name}.csv"
        summary.to_csv(path)
        logging.info(f"Saved report to {path}")

    def print_summary(self, name: str, summary: pd.Series) -> None:
        print(f"Performance Report ({name}):")
        print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    def summarize(self,
                  analyzers: Union[PerformanceAnalyzer, Dict[str,PerformanceAnalyzer]],
                  save: bool = False,
                  do_print: bool = False):
        if isinstance(analyzers, PerformanceAnalyzer):
            summaries = { analyzers.equity.name: analyzers.summary() }
        else:
            summaries = {name: pa.summary() for name, pa in analyzers.items()}

        # single or multi
        if len(summaries) == 1:
            name, summary = next(iter(summaries.items()))
            if do_print: self.print_summary(name, summary)
            if save:  self.save_summary(name, summary)
            return summary
        else:
            df = pd.DataFrame(summaries).T
            if do_print:
                print("Combined Performance Report:")
                print(df.to_string(float_format=lambda x: f"{x:.4f}"))
            if save:
                self.save_summary("comparison_report", df)
            return df

# --- TradingSystem ---
class TradingSystem:
    def __init__(self,
                 symbol: str,
                 strategy: TradingStrategy,
                 risk_manager: Optional[RiskManager] = None):
        self.symbol = symbol
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.data_mgr = DataManager(
            data_dir=DATA_DIR,
            max_file_size=MAX_FILE_SIZE
        )
        self.data: Optional[pd.DataFrame] = None

    def load(self,
             start: Optional[str] = None,
             end: Optional[str] = None,
             period: Optional[str] = '1y',
             interval: Optional[str] = '1d') -> pd.DataFrame:
        self.data = self.data_mgr.load_data(
            symbols=self.symbol,
            start=start,
            end=end,
            period=period,
            interval=interval
        )
        return self.data

    def run_backtest(self,
                     initial_capital: float = 10000.0,
                     transaction_cost: float = 0.0,
                     stop_loss: Optional[float] = None,
                     save_report: bool = False,
                     do_print: bool = False,
                     report_name: Optional[str] = None) -> pd.Series:
        if self.data is None:
            raise RuntimeError("Datos no cargados. Llama a load() primero.")
        eq = self.strategy.backtest(
            self.data,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            risk_manager=self.risk_manager,
            stop_loss=stop_loss
        )
        analyzer = PerformanceAnalyzer(eq, self.strategy.returns)
        # analyzer.report(save=save_report, do_print=do_print, report_name=report_name)
        rm = ReportManager()
        rm.summarize(analyzer, save=save_report, do_print=do_print)
        return eq



class StrategyCollection:
    """
    Genera y agrupa múltiples instancias de una misma estrategia con diferentes parámetros.
    Permite iterar, guardar y backtestear todas las estrategias de la colección de forma centralizada.
    Estrategias inválidas (por ejemplo, parámetros que lanzan errores) se omiten con advertencia.
    """
    def __init__(self,
                 strategy_cls: Type['TradingStrategy'] = None,
                 param_grid: Dict[str, List[Any]] = None,
                 strategies: Optional[List[TradingStrategy]] = None
    ):
        self.strategies: Dict[str, TradingStrategy] = {}

        if strategies is not None:
            # load from provided list
            for strat in strategies:
                if strat.name in self.strategies:
                    logging.warning(f"Duplicate strategy name '{strat.name}', skipping")
                    continue
                self.strategies[strat.name] = strat
            return

        # otherwise, must have strategy_cls & param_grid
        if strategy_cls is None or param_grid is None:
            raise ValueError("Must provide either `strategies` or (`strategy_cls` and `param_grid`)")
        self.strategy_cls = strategy_cls
        self.param_grid  = param_grid
        self._generate_strategies()

    def _generate_strategies(self) -> None:
        """
        Crea instancias de la estrategia para cada combinación de parámetros.
        Usa el .name que la propia clase haya definido.
        """
        for vals in product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), vals))
            try:
                strat = self.strategy_cls(**params)
            except Exception as e:
                # aquí strat.name aún no existe, así que usamos params para el warning
                pname = f"{self.strategy_cls.__name__}({params})"
                logging.warning(f"Skipping strategy '{pname}': {e}")
                continue

            # Aquí aprovechamos el .name que la estrategia ya haya generado
            name = strat.name

            if name in self.strategies:
                logging.warning(f"Duplicate strategy name '{name}', skipping")
                continue

            self.strategies[name] = strat

    def _format_name(self, params: Dict[str, Any]) -> str:
        """
        Genera un nombre único para la estrategia basado en su clase y parámetros.
        Ejemplo: MovingAverageCross_short_20_long_50
        """
        parts = [f"{k}_{v}" for k, v in params.items()]
        return f"{self.strategy_cls.__name__}_" + "_".join(parts)

    def __iter__(self) -> Iterator['TradingStrategy']:
        return iter(self.strategies.values())

    def __len__(self) -> int:
        return len(self.strategies)

    def names(self) -> List[str]:
        """Devuelve la lista de nombres de las estrategias generadas."""
        return list(self.strategies.keys())

    def backtest_all(
        self,
        symbol: str,
        period: str = '1y',
        interval: str = '1d',
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.0,
        risk_manager: Optional['RiskManager'] = None,
        stop_loss: Optional[float] = None
    ) -> Dict[str, pd.Series]:
        """
        Realiza backtest de todas las estrategias de la colección sobre un símbolo dado.
        Devuelve un diccionario mapping nombre de estrategia a su curva de equity.
        """
        equity_dict: Dict[str, pd.Series] = {}
        for name, strat in self.strategies.items():
            ts = TradingSystem(symbol=symbol, strategy=strat, risk_manager=risk_manager)
            ts.load(period=period, interval=interval)
            eq = ts.run_backtest(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                stop_loss=stop_loss
            )
            equity_dict[name] = eq
        return equity_dict

    def save_all(self, strategy_manager: 'StrategyManager', directory: Optional[Path] = None) -> None:
        """
        Guarda todas las estrategias usando un StrategyManager.
        Opción de especificar un directorio alternativo.
        """
        if directory:
            strategy_manager.directory = directory
        for strat in self.strategies.values():
            strategy_manager.save(strat)

    @classmethod
    def load_all(cls, names: List[str], strategy_manager: StrategyManager):
        strategies = [ strategy_manager.load(n) for n in names ]
        return cls(strategies=strategies)
