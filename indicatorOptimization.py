import numpy as np
import utilities as ut
import matplotlib.pyplot as plt
import pandas as pd
import re
import technicalAnalysis as ta
import seaborn as sns

# Read data
df = pd.read_excel('Historical_Data/bitcoin price history.xlsx', index_col=0)
df.index = pd.to_datetime(df.index, errors='coerce')

# Calculate daily returns

df['return'] = df['close'] / df['close'].shift(1) - 1
ta_df = df[['open', 'high', 'low', 'close', 'volume', 'return']].ix[1:]
ta_df['next return'] = ta_df['return'].shift(-1)
ta_df = ta_df.ix[:-1]

# Technical analysis
for i in range(5, 81):
    ta_df['rsi{}'.format(i)] = ta.rsi(ta_df['return'], i)
    ta_df['cci{}'.format(i)] = ta.cci(ta_df, i)
    ta_df['bb{}'.format(i)] = ta.bb(ta_df['close'], i)
    ta_df['wr{}'.format(i)] = ta.wr(ta_df, i)
    ta_df['atr{}'.format(i)] = ta.atr(ta_df, i)


# Optomize obv indicator with two time periods
def obv_opt(ohlc_df):
    short_period = range(5, 16)
    long_period = range(25, 81)
    obv_data = np.zeros((short_period[-1] + 1, long_period[-1] + 1))
    for i in short_period:
        for j in long_period:
            ta_df['obv{}_{}'.format(i, j)] = ta.obv(ohlc_df, i, j)

    obv_cols = [f for f in list(ta_df) if 'obv' in f]
    obv_corrs = ut.correlation(ta_df[obv_cols + ['next return']], col='next return', ret_col=True)
    for i in short_period:
        for j in long_period:
            obv_data[i, j] = obv_corrs['obv{}_{}'.format(i, j)]

    maxloc = np.unravel_index(obv_data.argmax(), obv_data.shape)
    print(maxloc)
    plt.figure(figsize=(20, 16/3))
    plt.plot([maxloc[1] + 0.5, 74.5], [9.5, 10.5], '*', color='gold', ms=18)
    vmin = np.min(obv_data[np.nonzero(obv_data)])
    ax = sns.heatmap(obv_data, vmin=vmin)
    plt.xlabel('Long Time Period (Hours)', fontsize=18)
    plt.ylabel('Short Time Period (Hours)', fontsize=18)
    plt.xlim([long_period[-1] - 1, long_period[0]])
    ax.set_ylim(top=11)
    plt.savefig('Figures/obv_opt.png', bbox_inches='tight', format='png', dpi=300)
    plt.show()

obv_opt(ta_df)

# View Results
features = ['rsi', 'cci', 'bb', 'wr', 'atr']
stars = {'rsi': [5, 10, 23, 33, 62], 'cci': [6, 11, 45], 'bb': [8, 11, 74], 'wr': [6, 13, 48, 76], 'atr': [5, 14, 69]}
i = 1

fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 8))
for feature, ax, in zip(features, axes.flatten()):

    selected = [f for f in list(ta_df) if feature in f]
    corrs = ut.correlation(ta_df[selected + ['next return']], col='next return', ret_col=True)

    # Sort correlation by period
    nums = []
    for name in corrs.index:
        try:
            num = int(re.findall(r'\d+', name)[0])
        except:
            pass
        nums.append(num)
    temp = pd.DataFrame({'x': np.asarray(nums), 'y': corrs})
    temp.sort_values(['x'], inplace=True)

    # Find y value of selected local maxima
    star_y = []
    for x in stars[feature]:
        star_y.append(temp.loc[temp['x'] == x, 'y'].iloc[0])

    # plot correlation as a function of period
    ax.plot(temp.x, temp.y, '-o', color='steelblue', ms=6)
    ax.plot(stars[feature], star_y, '*', color='gold', ms=15)
    ax.set_title(feature.upper() + ' Period Optimization')


fig.text(0.5, 0.04, 'Time Period (Hours)', ha='center', fontsize=15)
fig.text(0.06, 0.5, 'Correlation with Next Hour Return', va='center', rotation='vertical', fontsize=15)
# plt.savefig('Figures/indicator_period_opt.png', bbox_inches='tight', format='png', dpi=300)
plt.show()
