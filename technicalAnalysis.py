import pandas as pd
import numpy as np


# Exponential Moving Average (EMA)
def ema(close, period):
    sma = close.rolling(window=period, min_periods=period).mean()[:period]
    rest = close.ix[period:]
    return pd.concat([sma, rest]).ewm(span=period, adjust=False).mean()


# Bollinger Bands
def bb(close, period):
    bb_sma = close.rolling(period).mean()
    bb_std = close.rolling(period).std(ddof=0)
    return (close - (bb_sma - 2 * bb_std)) / ((bb_sma + 2 * bb_std) - (bb_sma - 2 * bb_std))


# Moving average convergence / divergence (MACD)
# Either (12, 26, 29) or (5, 35, 5)
def macd(close, period):
    macd_line = ema(close, period[0]) - ema(close, period[1])
    signal_line = ema(macd_line, period[2])
    return macd_line - signal_line


# Relative strength index (RSI)
def rsi(returns, period):
    up, down = returns.copy(), returns.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean().abs()
    return 100.0 - (100.0 / (1.0 + roll_up / roll_down))


# Commodity Channel Index (CCI)
def cci(ohlc_df, period):
    ohlc_df['typ'] = (ohlc_df['high'] + ohlc_df['low'] + ohlc_df['close']) / 3
    dev = np.sum(np.absolute(ohlc_df['typ'] - ohlc_df['close'].rolling(period).mean())) / period
    return (ohlc_df['typ'] - ohlc_df['close'].rolling(period).mean()) / (0.15 * dev)


# Williams %R (WR)
def wr(ohlc_df, period):
    high = ohlc_df['high'].rolling(period).max()
    low = ohlc_df['low'].rolling(period).min()
    return -100 * (high - ohlc_df['close']) / (high - low)


# Average true range (ATR)
def atr(ohlc_df, period):
    ohlc_df['tr1'] = abs(ohlc_df['high'] - ohlc_df['low'])
    ohlc_df['tr2'] = abs(ohlc_df['high'] - ohlc_df['close'].shift(1))
    ohlc_df['tr3'] = abs(ohlc_df['low'] - ohlc_df['close'].shift(1))
    ohlc_df['tr'] = ohlc_df[['tr1', 'tr2', 'tr3']].max(axis=1)
    ohlc_df['atr'] = ohlc_df['tr'].rolling(period).mean()
    ohlc_df['atr'].ix[period + 1:] = ((period - 1) * ohlc_df['tr'].ix[period + 1:].shift(1) +
                                      ohlc_df['tr'].ix[period + 1:]) / period
    return ohlc_df['atr']


# On balance volume (OBV)
def obv(ohlcv_df, short_period, long_period):
    ohlcv_df['cum vol'] = 0
    ohlcv_df['cum vol'].ix[1:] = ohlcv_df['cum vol'].shift(1) + np.sign(
        ohlcv_df['close'] - ohlcv_df['close'].shift(1)) * ohlcv_df['volume']
    return ema(ohlcv_df['cum vol'], short_period) - ema(ohlcv_df['cum vol'], long_period)
