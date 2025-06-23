from imports import *
from trading_environment import *



# --- Trading Strategies classes: ---
class MovingAverageCrossStrategy(TradingStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(name=f"MA {short_window}/{long_window}")
        if short_window >= long_window:
            raise ValueError("short_window debe ser menor que long_window")
        self.short_window, self.long_window = short_window, long_window

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
    Invierte un monto fijo cada periodo freq (por ejemplo mensual).
    """
    def __init__(self,
                 amount: float = 1000.0,
                 freq: str = 'M',
                 price_col: str = 'close'):
        super().__init__(name=f"DCA {amount}@{freq}", price_col=price_col)
        self.amount = amount
        self.freq   = freq

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # extraemos fecha plana del MultiIndex
        dates = pd.to_datetime(data.index.get_level_values(0))
        px    = pd.Series(data[self.price_col].values, index=dates)
        # programamos fechas de inversión
        invest_dates = pd.date_range(px.index.min(), px.index.max(), freq=self.freq)

        df = pd.DataFrame({'price': px})
        df['invest'] = 0.0
        for dt in invest_dates:
            if dt in df.index:
                df.at[dt, 'invest'] = self.amount

        # acumulado de inversión
        df['cum_invest'] = df['invest'].cumsum()
        # fracción de capital invertido
        frac = (df['cum_invest'] / self._initial_capital).clip(0,1)

        # señal de posición es esa fracción replicada en multiindex
        sig_flat = frac.reindex(dates).ffill().fillna(0)
        return pd.Series(sig_flat.values, index=data.index)







#  --- NEW ONES: ---

class DonchianBreakoutStrategy(TradingStrategy):
    def __init__(self, window: int = 20, price_col: str = 'close'):
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
    def __init__(self, short_w=20, long_w=50, adx_w=14):
        super().__init__(name=f"MA {short_w}/{long_w} + ADX")
        self.ma = MovingAverageCrossStrategy(short_w, long_w)
        self.adx_w = adx_w

    def generate_signals(self, data):
        sig_ma = self.ma.generate_signals(data)
        high, low, close = data['high'], data['low'], data['close']
        adx = ta.trend.ADXIndicator(high, low, close, self.adx_w).adx()
        strong = adx > 25
        return sig_ma.where(strong, 0).ffill().fillna(0)


class ROCStrategy(TradingStrategy):
    def __init__(self, window: int = 10):
        super().__init__(name=f"ROC {window}")
        self.window = window

    def generate_signals(self, data):
        roc = data[self.price_col].pct_change(self.window)
        sig = roc.apply(np.sign)
        return sig.ffill().fillna(0)



class MACDStrategy(TradingStrategy):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(name=f"MACD {fast}/{slow}/{signal}")
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate_signals(self, data):
        macd = ta.trend.MACD(data[self.price_col], self.fast, self.slow, self.signal)
        diff = macd.macd_diff()  # macd - signal
        long  = (diff > 0) & (diff.shift(1) <= 0)
        short = (diff < 0) & (diff.shift(1) >= 0)
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)


class BollingerMeanRevStrategy(TradingStrategy):
    def __init__(self, window=20, n_std=2):
        super().__init__(name=f"BB Rev {window}/{n_std}")
        self.window, self.n_std = window, n_std

    def generate_signals(self, data):
        price = data[self.price_col]
        ma    = price.rolling(self.window).mean()
        std   = price.rolling(self.window).std()
        upper = ma + self.n_std*std
        lower = ma - self.n_std*std

        long  = price < lower
        short = price > upper
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)



class RSIStrategy(TradingStrategy):
    def __init__(self, window=14, low=30, high=70):
        super().__init__(name=f"RSI {window}")
        self.window, self.low, self.high = window, low, high

    def generate_signals(self, data):
        rsi = ta.momentum.RSIIndicator(data[self.price_col], self.window).rsi()
        long  = (rsi < self.low)
        short = (rsi > self.high)
        sig   = pd.Series(0, index=data.index)
        sig[long], sig[short] = 1, -1
        return sig.ffill().fillna(0)



class PairTradingStrategy(TradingStrategy):
    def __init__(self, pair: Tuple[str,str], window=20, entry_z=2, exit_z=0):
        super().__init__(name=f"Pair {pair[0]}/{pair[1]}")
        self.s1, self.s2 = pair
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z

    def generate_signals(self, data):
        p1, p2 = data[self.s1], data[self.s2]
        spread = p1 - p2
        mu = spread.rolling(self.window).mean()
        sigma = spread.rolling(self.window).std()
        z = (spread - mu) / sigma

        long  = z < -self.entry_z
        short = z > self.entry_z
        exit  = z.abs() < self.exit_z

        sig = pd.Series(method='ffill', index=data.index, data=0.0)
        sig[long]  =  1   # long spread: buy p1, sell p2
        sig[short] = -1
        sig[exit]  =  0
        return sig.fillna(method='ffill').fillna(0)



# class ATRStopStrategy(TradingStrategy):
#     def __init__(self, window=14, multiple=3):
#         super().__init__(name=f"ATR Stop {window}x{multiple}")
#         self.window, self.multiple = window, multiple

#     def generate_signals(self, data):
#         entry = MovingAverageCrossStrategy(20,50).generate_signals(data)
#         atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], self.window).average_true_range()
#         stop = entry.copy().astype(float)
#         long_mask = entry == 1
#         short_mask = entry == -1

#         # trailing stop logic (simplified): once in a trade, exit when price breaches entry ± multiple*ATR
#         # ... (would need to track entry price per trade)
#         return entry  # placeholder: combine entry+stop logic in a real implementation




class VWAPStrategy(TradingStrategy):
    def __init__(self, price_col: str = 'close'):
        super().__init__(name="VWAP", price_col=price_col)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1) compute typical price if not present
        if 'typical_price' in data.columns:
            tp = data['typical_price']
        else:
            # assume data has 'high','low','close'
            tp = data[['high','low', self.price_col]].mean(axis=1)

        # 2) cumulative VWAP
        volume = data['volume']
        cum_vp = (tp * volume).cumsum()
        cum_v  = volume.cumsum()
        vwap   = cum_vp / cum_v

        # 3) signal: long when price > VWAP, short when < VWAP
        price = data[self.price_col]
        sig   = (price > vwap).astype(int) * 2 - 1

        return sig.ffill().fillna(0)
