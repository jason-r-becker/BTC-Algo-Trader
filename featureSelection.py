import matplotlib.pyplot as plt
import numpy as np
import technicalAnalysis as ta
import pandas as pd
import utilities as ut
from matplotlib import cm
from matplotlib import style

# Settings
# -------------------------------- #
coin = 'bitcoin'
null_vals = 'all'
min_bound = 50
# -------------------------------- #

style.use('ggplot')

# Read data
df = pd.read_csv('Training_Data/{}_{}.csv'.format(coin, null_vals), index_col=0)
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[df['minutes'] > min_bound]


# Find whether to use raw or clean data
def raw_clean_compare(sentiment):
    comp_df = df[['raw ' + sentiment, 'clean ' + sentiment, 'return']]
    comp_corr = comp_df.corr().abs()
    rank = comp_corr.sort_values(['return'], ascending=False)['return']
    return rank.index[1]


sentiments = ['pol', 'sub', 'pos', 'neg', 'neu']
best_sents = ['count', 'normed count']
for s in sentiments:
    best_sents.append(raw_clean_compare(s))
best_sents.append('return')

# Determine correlation to daily returns
corr_df = df.drop(
    ['move', 'close', 'minutes', 'hour', 'weekday', 'clean pos', 'clean neg', 'raw neu', 'count', 'clean pol',
     'raw sub'],
    axis=1)
save_df = df.drop(['minutes', 'close', 'hour', 'weekday', 'clean pos', 'clean neg', 'raw neu', 'count', 'clean pol',
                   'raw sub'], axis=1)

corr = corr_df.corr().abs()
results = corr.sort_values(['return'], ascending=False)['return'][1:]
print(results)

# Plot results
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(len(corr.columns), len(corr.columns)))
ax.matshow(corr)
cmap = cm.get_cmap('coolwarm', 300)
cax = ax.imshow(corr, interpolation="nearest", cmap=cmap)
plt.xticks(range(len(corr.columns)), corr.columns, fontsize=15)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=15)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
cax.set_clim([0, 1])
fig.colorbar(cax, ticks=[0.00, .25, .5, .75, 1.00])
plt.savefig('Figures/correlation_matrix.png', bbox_inches='tight', format='png', dpi=300)
plt.show()
quit()
plt.show()


# Transform features
hist_df = save_df.copy()
hist_df['arcsin(rsi 5)'] = np.arcsin(np.sqrt((hist_df['rsi 5'] / 100)))
hist_df['arcsin(wr 6)'] = np.arcsin(np.sqrt((hist_df['wr 6'] / -100)))
hist_df['arcsin(wr 13)'] = np.arcsin(np.sqrt((hist_df['wr 13'] / -100)))
hist_df['arcsin(wr 48)'] = np.arcsin(np.sqrt((hist_df['wr 48'] / -100)))
hist_df['arcsin(wr 76)'] = np.arcsin(np.sqrt((hist_df['wr 76'] / -100)))
hist_df['ln(atr 5)'] = np.log(hist_df['atr 5'])
hist_df['ln(atr 14)'] = np.log(hist_df['atr 14'])
hist_df['ln(atr 69)'] = np.log(hist_df['atr 69'])
hist_df['ln(raw pos)'] = np.log(hist_df['raw pos'])
hist_df['ln(raw neg)'] = np.log(hist_df['raw neg'])

# Sentiment histograms
ut.hists(hist_df[['raw pos', 'raw neg', 'ln(raw neg)', 'ln(raw pos)']], 50, c='skyblue', ctop=True, fs=25)
# plt.savefig('Figures/sent_transformed_hists.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

ut.hists(hist_df[['clean neu', 'raw pol', 'clean sub', 'sbi']], 50, c='skyblue', fs=25)
# plt.savefig('Figures/sent_raw_hists.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

# Transformed features histrogram
save_df = hist_df.copy()
hist_df = hist_df.drop(['return', 'move'], axis=1)
ut.hists(hist_df[['atr 5', 'atr 14', 'atr 69', 'rsi 5', 'wr 6', 'wr 13', 'wr 48', 'wr 76', 'ln(atr 5)', 'ln(atr 14)',
                  'ln(atr 69)', 'arcsin(rsi 5)', 'arcsin(wr 6)', 'arcsin(wr 13)', 'arcsin(wr 48)', 'arcsin(wr 76)']],
         bins=50, size=(4, 4), ctop=True, fs=18)
# plt.savefig('Figures/transform_hist.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

# Non transformed historgrams
ta_df = hist_df.drop(
    ['raw pos', 'clean neu', 'raw neg', 'raw pol', 'clean sub', 'sbi', 'atr 5', 'atr 14', 'atr 69', 'rsi 5', 'wr 6',
     'wr 13', 'wr 48', 'wr 76', 'ln(atr 5)', 'ln(atr 14)', 'ln(atr 69)', 'arcsin(rsi 5)', 'arcsin(wr 6)',
     'arcsin(wr 13)', 'arcsin(wr 48)', 'arcsin(wr 76)', 'ln(raw neg)', 'ln(raw pos)'], axis=1)
ut.hists(ta_df, 50, fs=18)
# plt.savefig('Figures/non_transformed_hists.png', bbox_inches='tight', format='png', dpi=300)
plt.show()

# Save Transformed features for training
save_df = save_df.drop(['atr 5', 'atr 14', 'atr 69', 'rsi 5', 'wr 6', 'wr 13', 'wr 48', 'wr 76', 'raw pos', 'raw neg'],
                       axis=1)
print(list(save_df))
save_df.to_csv('Training_Data/btc final.csv')

corr = save_df.corr().abs()
results = corr.sort_values(['return'], ascending=False)['return'][1:]
print(results)