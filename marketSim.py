import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utilities as ut
import math
import warnings

import matplotlib.ticker as tkr
from pandas_datareader import data
from matplotlib import style
from scipy.stats.mstats import gmean

style.use('ggplot')
warnings.filterwarnings('ignore')

investment = 10000  # initial investment
spy_fee = 5  # fee trade ($)

df = pd.read_csv('Training_Data/btc final 57_3.csv', index_col=0)
pred = pd.read_csv('Saved_Models/Model Results 57_3.csv', index_col=0)


# Add predictions to correct dates
n = len(df.index)  # number of samples
test = df.ix[int(0.85 * n):]  # test on 15% of samples
test['pred'] = np.asarray(pred['preds'])
test = test[['return', 'move', 'pred']]
# ut.prettyPrint(test.head(20))

# Get S&P 500 returns over same interval
spy = data.DataReader('SPY', 'google', test.index[0], test.index[-1])
spy['dr'] = spy['Close'].shift(-1) / spy['Close'] - 1  # daily returns
spy = spy[(spy.index >= test.index[0]) & (spy.index <= test.index[-1])]

# Calculate S&P 500 returns over same period
spy['val'] = investment
spy['val'].ix[0] = (investment - spy_fee) * (1 + spy['dr'].ix[0])
for i in range(1, len(spy.index)):
    spy['val'].ix[i] = spy['val'].ix[i - 1] * (1 + spy['dr'].ix[i])

# Initialize dataframe
df = test[['pred', 'return']]
df['port val'] = investment
df['btc'] = investment
df.index = pd.to_datetime(df.index, errors='coerce')

# Determine BUY, SELL, and HOLD
df['action'] = df['pred']
df['action'][df['pred'] == 0] = -1  # Change predictions to 1 for BUY and -1 for SELL
pos = df['pred'].ix[0]  # intial position
for i in range(1, len(df.index)):
    if df['action'].ix[i] == pos:
        df['action'].ix[i] = 0  # HOLD
    else:
        pos = df['action'].ix[i]

# kraken trade fees for given volumes
slippage = 10  # (bips)
trade_fee = 1e-4 * (np.asarray([26, 24, 22, 20, 18, 16, 14, 12, 10]) + slippage)
trade_vol = 1e3 * np.array([0, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

# Determine current fee
def getFee(vol, t_fee=trade_fee, t_vol=trade_vol):
    for i, f in enumerate(t_fee):
        if vol >= t_vol[i]:
            cur_fee = f
    return cur_fee


# Calculate portfolio value
vol = investment * (1 - trade_fee[0])
pos = df['action'].ix[0]
df['port val'].ix[0] = (1 - trade_fee[0]) * (1 + df['return'].ix[0] * df['action'].ix[0]) * investment  # inital move
df['btc'].ix[0] = (1 - trade_fee[0]) * (1 + df['return'].ix[0]) * investment  # initial buying of bitcoin
df['vol'] = vol
for i in range(1, len(df.index)):
    act = df['action'].ix[i]  # BUY, SELL, or HOLD
    position = df['port val'].ix[i - 1] * (1 - getFee(vol))  # amount being long or short

    # Trader value
    if act == 1:  # BUY
        df['port val'].ix[i] = position * (1 + df['return'].ix[i])
        vol += position
        pos = 1
    elif act == -1:  # SELL
        df['port val'].ix[i] = position * (1 - df['return'].ix[i])
        vol += position
        pos = -1
    elif (act == 0) & (pos == 1):  # HOLD long
        df['port val'].ix[i] = df['port val'].ix[i - 1] * (1 + df['return'].ix[i])
    elif (act == 0) & (pos == -1):  # HOLD short
        df['port val'].ix[i] = df['port val'].ix[i - 1] * (1 - df['return'].ix[i])

    # Buy and hold Bitcoin value
    df['btc'].ix[i] = df['btc'].ix[i - 1] * (1 + df['return'].ix[i])

    # update volume
    df['vol'].ix[i] = vol


# Plot Results
def ticks(x, pos):  # formatter function takes tick label and tick position
   s = '{:0,d}'.format(int(x))
   return s
y_format = tkr.FuncFormatter(ticks)  # make formatter

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
ax.yaxis.set_major_formatter(y_format)
plt.plot(df.index, df['port val'], c='steelblue', label='Algorthmic Trader')
plt.plot(df.index, df['btc'], c='darkorange', label='Buy and Hold XBT')
plt.plot(spy.index, spy['val'], c='salmon', label='Buy and Hold S&P 500')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
plt.legend()
plt.savefig('Figures/market_result.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

# Multi color
fig = plt.figure(figsize=(8, 6))

# Make 2 color line
df['port ret'] = df['port val'] / investment
leg = ['Positive Return', 'Negative Return', '_nolegend_']
p, n = 0, 1
for i in range(1, len(df.index)):
    if df['port ret'].ix[i] - df['port ret'].ix[i-1] > 0:
        plt.plot(df.index[i - 1: i + 1], df['port ret'].ix[i - 1: i + 1], c='darkgreen', label=leg[p])
        p = 2
    else:
        plt.plot(df.index[i - 1: i + 1], df['port ret'].ix[i - 1: i + 1], c='firebrick', label=leg[n])
        n = 2

# Make 2 color fill
plt.fill_between(df.index, 1, df['port ret'], where=df['port ret'] > 1,  color='darkgreen', alpha=0.4, label=leg[2])
plt.fill_between(df.index, df['port ret'], 1, where=df['port ret'] < 1,  color='firebrick', alpha=0.4, label=leg[2])

# plot data
plt.xlabel('Date')
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
plt.ylabel('Portfolio Return')
plt.legend(['Positive Return', 'Negative Return'])
plt.savefig('Figures/port_val.png', bbox_inches='tight', format='png', dpi=300)
plt.show()






# Plot Volume
vol_pairs = [(5e4, 16), (1e5, 14), (25e4, 12), (5e5, 10)]
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
ax.yaxis.set_major_formatter(y_format)
plt.plot(df.index, df['vol'], c='steelblue')
plt.fill_between(df.index, 0 , df['vol'], color='steelblue', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Cumulative Volume Traded (USD)')
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
# add fee lines
ax.text(pd.Timestamp("2017-09-14 8:00"), 1e4, '18 bp', color='lightgrey')
for vp in vol_pairs:
    plt.axhline(y=vp[0], linestyle='--', color='lightgrey')
    ax.text(pd.Timestamp("2017-09-14 8:00"), vp[0] + 1e4, '{} bp'.format(vp[1]), color='lightgrey')

plt.savefig('Figures/cum_vol.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

rfr = 2.13 * 1e-2  # (%) risk free interest rate of T-bill
real_rfr = (1 + rfr) ** (len(spy.index) / 365) - 1  # real risk free rate for given analyzed time period


# Sharpe ratio
def sharpe_ratio(returns, rf=real_rfr):
    return (np.mean(returns) - 1 - rf) / np.std(returns)


# Least partial moment
def lpm(returns, threshold, order):
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    diff = threshold_array - returns
    diff = diff.clip()
    return np.sum(diff ** order) / len(returns)


# Sortino ratio
def sortino_ratio(returns, rf=real_rfr, target=0):
    return (np.mean(returns) - 1 - rf) / math.sqrt(lpm(returns, target, 2))


print('Algorithm\nSharpe Ratio:\t{:.3f}\nSortino Ratio:\t{:.3f}'.format(sharpe_ratio(df['port val'] / investment),
                                                                        sortino_ratio(df['port val'] / investment)))
print('Bitcoin\nSharpe Ratio:\t{:.3f}\nSortino Ratio:\t{:.3f}'.format(sharpe_ratio(df['btc'] / investment),
                                                                      sortino_ratio(df['btc'] / investment)))
print('SPY\nSharpe Ratio:\t{:.3f}\nSortino Ratio:\t{:.3f}'.format(sharpe_ratio(spy['val'] / investment),
                                                                  sortino_ratio(spy['val'] / investment)))
