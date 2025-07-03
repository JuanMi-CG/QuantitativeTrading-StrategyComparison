from .imports import *
from .trading_environment import *


# --- Trading Strategies classes with auto‐tunable param_config: ---

class Benchmark(TradingStrategy):
    # Siempre en posición larga (1).
    param_config = {}  # sin parámetros a tunear

    def __init__(self, params: dict = None, price_col: str = 'close'):
        super().__init__(name='Benchmark', price_col=price_col)

    def generate_signals(self, data):
        # señal = +1 cada día
        return pd.Series(1, index=data.index)


class MovingAverageCrossStrategy(TradingStrategy):
    # auto‐tunable params: integer windows
    param_config = {
        'short_window': {'type': int,   'bounds': (5, 100),  'step': 5},
        'long_window':  {'type': int,   'bounds': (20, 200), 'step': 10},
    }

    def __init__(self, params: dict):
        required = ['short_window', 'long_window']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"MovingAverageCrossStrategy missing parameters: {missing}")
        short, long_ = params['short_window'], params['long_window']

        super().__init__(name=f"MA {short}/{long_}")
        if short >= long_:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short
        self.long_window  = long_

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        price = data[self.price_col]
        sma_s = price.rolling(self.short_window).mean()
        sma_l = price.rolling(self.long_window).mean()
        up    = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
        down  = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))
        sig   = pd.Series(0, index=data.index)
        sig[up], sig[down] = 1, -1
        return sig.ffill().fillna(0)


class DcaStrategy(TradingStrategy):
    """
    Invierte un monto fijo cada periodo.
    """
    # amount: continuous from 100 to 2000 (10 samples), freq: categorical
    param_config = {
        'amount': {'type': float, 'bounds': (100.0, 2000.0), 'n': 10},
        'freq':   {'choices': ['D', 'W', 'ME']}
    }

    def __init__(self, params: dict, price_col: str = 'close'):
        required = ['amount', 'freq']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"DcaStrategy missing parameters: {missing}")
        amount, freq = params['amount'], params['freq']

        super().__init__(name=f"DCA {amount}@{freq}", price_col=price_col)
        self.amount = amount
        self.freq   = freq

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        dates        = pd.to_datetime(data.index.get_level_values(0))
        px           = pd.Series(data[self.price_col].values, index=dates)
        invest_dates = pd.date_range(px.index.min(), px.index.max(), freq=self.freq)

        df = pd.DataFrame({'price': px, 'invest': 0.0})
        for dt in invest_dates:
            if dt in df.index:
                df.at[dt, 'invest'] = self.amount

        df['cum_invest'] = df['invest'].cumsum()
        frac = (df['cum_invest'] / self._initial_capital).clip(0,1)

        sig_flat = frac.reindex(dates).ffill().fillna(0)
        return pd.Series(sig_flat.values, index=data.index)


class DonchianBreakoutStrategy(TradingStrategy):
    # window: integer from 10 to 100 step 10
    param_config = {
        'window': {'type': int, 'bounds': (10, 100), 'step': 10}
    }

    def __init__(self, params: dict, price_col: str = 'close'):
        required = ['window']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"DonchianBreakoutStrategy missing parameters: {missing}")
        window = params['window']

        super().__init__(name=f"Donchian {window}", price_col=price_col)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        price = data[self.price_col]
        upper = price.rolling(self.window).max()
        lower = price.rolling(self.window).min()
        long  = price > upper.shift(1)
        short = price < lower.shift(1)
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)


class ADXTrendStrategy(TradingStrategy):
    # mix of MA params plus adx window
    param_config = {
        'short_w': {'type': int, 'bounds': (5, 50),  'step': 5},
        'long_w':  {'type': int, 'bounds': (20, 200), 'step': 10},
        'adx_w':   {'type': int, 'bounds': (5, 30),  'step': 1},
    }

    def __init__(self, params: dict):
        required = ['short_w', 'long_w', 'adx_w']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"ADXTrendStrategy missing parameters: {missing}")
        sw, lw, aw = params['short_w'], params['long_w'], params['adx_w']

        super().__init__(name=f"MA {sw}/{lw} + ADX")
        self.ma    = MovingAverageCrossStrategy({'short_window': sw, 'long_window': lw})
        self.adx_w = aw

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        sig_ma           = self.ma.generate_signals(data)
        high, low, close = data['high'], data['low'], data['close']
        adx              = ta.trend.ADXIndicator(high, low, close, self.adx_w).adx()
        strong           = adx > 25
        return sig_ma.where(strong, 0).ffill().fillna(0)


class ROCStrategy(TradingStrategy):
    # window: integer from 5 to 50 step 5
    param_config = {
        'window': {'type': int, 'bounds': (5, 50), 'step': 5}
    }

    def __init__(self, params: dict):
        required = ['window']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"ROCStrategy missing parameters: {missing}")
        window = params['window']

        super().__init__(name=f"ROC {window}")
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        roc = data[self.price_col].pct_change(self.window)
        sig = roc.apply(np.sign)
        return sig.ffill().fillna(0)


class MACDStrategy(TradingStrategy):
    # MACD fast/slow/signal
    param_config = {
        'fast':   {'type': int,   'bounds': (5, 20),  'step': 1},
        'slow':   {'type': int,   'bounds': (20, 100), 'step': 5},
        'signal': {'type': int,   'bounds': (5, 30),  'step': 1},
    }

    def __init__(self, params: dict):
        required = ['fast', 'slow', 'signal']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"MACDStrategy missing parameters: {missing}")
        f, s, sig = params['fast'], params['slow'], params['signal']

        super().__init__(name=f"MACD {f}/{s}/{sig}")
        self.fast, self.slow, self.signal = f, s, sig

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        macd = ta.trend.MACD(data[self.price_col], self.fast, self.slow, self.signal)
        diff = macd.macd_diff()
        long  = (diff > 0) & (diff.shift(1) <= 0)
        short = (diff < 0) & (diff.shift(1) >= 0)
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)


class BollingerMeanRevStrategy(TradingStrategy):
    # window int, n_std float
    param_config = {
        'window': {'type': int,   'bounds': (10, 50),  'step': 5},
        'n_std':  {'type': float, 'bounds': (1.0, 3.0), 'n': 5},
    }

    def __init__(self, params: dict):
        required = ['window', 'n_std']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"BollingerMeanRevStrategy missing parameters: {missing}")
        w, n = params['window'], params['n_std']

        super().__init__(name=f"BB Rev {w}/{n}")
        self.window, self.n_std = w, n

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        price = data[self.price_col]
        ma    = price.rolling(self.window).mean()
        std   = price.rolling(self.window).std()
        upper = ma + self.n_std * std
        lower = ma - self.n_std * std

        long  = price < lower
        short = price > upper
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)


class RSIStrategy(TradingStrategy):
    # window int, low/high int
    param_config = {
        'window': {'type': int, 'bounds': (5, 30), 'step': 1},
        'low':    {'type': int, 'bounds': (10, 50), 'step': 5},
        'high':   {'type': int, 'bounds': (50, 90), 'step': 5},
    }

    def __init__(self, params: dict):
        required = ['window', 'low', 'high']
        missing = [k for k in required if k not in params]
        if missing:
            raise KeyError(f"RSIStrategy missing parameters: {missing}")
        w, low, high = params['window'], params['low'], params['high']

        super().__init__(name=f"RSI {w}")
        self.window, self.low, self.high = w, low, high

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi   = ta.momentum.RSIIndicator(data[self.price_col], self.window).rsi()
        long  = rsi < self.low
        short = rsi > self.high
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)



class PairTradingStrategy(TradingStrategy):
    # expects two symbols and a rolling window
    param_config = {
        's1': {'type': str, 'choices': ['GS', 'AAPL', 'BTC-USD', 'ETH-USD']},
        's2': {'type': str, 'choices': ['GS', 'AAPL', 'BTC-USD', 'ETH-USD']},
        'window': {'type': int, 'bounds': (5, 60), 'step': 5},
    }

    def __init__(self, params: dict):
        required = ['s1', 's2', 'window']
        missing_params = [k for k in required if k not in params]
        if missing_params:
            raise KeyError(f"PairTradingStrategy missing parameters: {missing_params}")
        self.s1 = params['s1']
        self.s2 = params['s2']
        self.window = params['window']
        super().__init__(name=f"Pair {self.s1}/{self.s2} w={self.window}")


    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1) If both legs are the same, no spread→ flat zeros
        if self.s1 == self.s2:
            return pd.Series(0, index=data.index, name='signal')

        # 2) load fresh each series
        dm  = DataManager(data_dir=DATA_DIR, max_file_size=MAX_FILE_SIZE)
        df1 = dm.load_data(symbols=self.s1, period='2y', interval='1d')
        df2 = dm.load_data(symbols=self.s2, period='2y', interval='1d')

        s1 = df1[self.price_col].rename(self.s1)
        s2 = df2[self.price_col].rename(self.s2)

        # align on the same dates
        df = pd.concat([s1, s2], axis=1).dropna()

        p1 = df[self.s1]
        p2 = df[self.s2]

        # if you still end up with a DataFrame (duplicate names),
        # just grab the first column
        if isinstance(p1, pd.DataFrame):
            p1 = p1.iloc[:, 0]
        if isinstance(p2, pd.DataFrame):
            p2 = p2.iloc[:, 0]

        # compute spread stats
        spread = p1 - p2
        mu     = spread.rolling(self.window).mean()
        sigma  = spread.rolling(self.window).std()

        # build your signal
        signal = pd.Series(0, index=spread.index, name='signal')
        mask_long  = spread < (mu - sigma)
        mask_short = spread > (mu + sigma)

        # 3) _always_ use .loc for boolean assignment
        signal.loc[mask_long]  =  1
        signal.loc[mask_short] = -1

        return signal.ffill().fillna(0)




class VWAPStrategy(TradingStrategy):
    # no tunable parameters
    param_config = {}

    def __init__(self, params: dict = None, price_col: str = 'close'):
        super().__init__(name="VWAP", price_col=price_col)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if 'typical_price' in data.columns:
            tp = data['typical_price']
        else:
            tp = data[['high','low', self.price_col]].mean(axis=1)

        volume = data['volume']
        cum_vp = (tp * volume).cumsum()
        cum_v  = volume.cumsum()
        vwap   = cum_vp / cum_v

        price = data[self.price_col]
        sig   = (price > vwap).astype(int) * 2 - 1
        return sig.ffill().fillna(0)
