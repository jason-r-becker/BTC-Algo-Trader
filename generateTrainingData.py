import csv
import numpy as np
import os
import pandas as pd
import re
import settings
import time
import technicalAnalysis as ta

from nltk.corpus import words
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

t0 = time.clock()


# Read Twitter data file
def combineRawDataFiles(verbose=False):
    path = os.path.expanduser(r"~/odrive/Dropbox/Raw_Data/")
    df = pd.DataFrame()
    datafiles = [f for f in os.listdir(path) if '.csv' in f]

    for fid in datafiles:
        if verbose:
            print(fid)
        try:
            data = pd.read_csv(path + fid, names=['uniqueID', 'username', 'tweet'],
                               engine='c', quoting=csv.QUOTE_NONE)
        except:
            data = pd.read_csv(path + fid, names=['uniqueID', 'username', 'tweet'],
                               engine='python', quoting=csv.QUOTE_NONE)
        if verbose:
            print('read succesfully')
        df = pd.concat([df, data])
    df.to_csv(r'/Users/jasonbecker/PycharmProjects/CryptoTrading/RawData.csv')


# Get market data
def market(currency):
    dfy = pd.read_excel('Historical_Data/{} price history.xlsx'.format(currency), index_col=0)
    dfy.index = pd.to_datetime(dfy.index, errors='coerce')

    # Calculate daily returns
    dfy['return'] = dfy['close'].shift(-1) / dfy['close'] - 1

    def y_move(val):
        if float(val) > 0:
            return 1
        else:
            return 0

    # Find if price moved up or down
    dfy['move'] = dfy['return'].apply(lambda x: y_move(x))

    # Calculate technical indicators
    dfy['rsi 5'] = ta.rsi(dfy['return'], 5)
    dfy['rsi 10'] = ta.rsi(dfy['return'], 10)
    dfy['rsi 23'] = ta.rsi(dfy['return'], 23)
    dfy['rsi 33'] = ta.rsi(dfy['return'], 33)
    dfy['rsi 62'] = ta.rsi(dfy['return'], 62)
    dfy['cci 6'] = ta.cci(dfy, 6)
    dfy['cci 11'] = ta.cci(dfy, 11)
    dfy['cci 45'] = ta.cci(dfy, 45)
    dfy['bb 8'] = ta.bb(dfy['close'], 8)
    dfy['bb 11'] = ta.bb(dfy['close'], 11)
    dfy['bb 74'] = ta.bb(dfy['close'], 74)
    dfy['macd 5-35-5'] = ta.macd(dfy['close'], (5, 35, 5))
    dfy['macd 12-26-9'] = ta.macd(dfy['close'], (12, 26, 9))
    dfy['wr 6'] = ta.wr(dfy, 6)
    dfy['wr 13'] = ta.wr(dfy, 13)
    dfy['wr 48'] = ta.wr(dfy, 48)
    dfy['wr 76'] = ta.wr(dfy, 76)
    dfy['atr 5'] = ta.atr(dfy, 5)
    dfy['atr 14'] = ta.atr(dfy, 14)
    dfy['atr 69'] = ta.atr(dfy, 69)
    dfy['obv 6-40'] = ta.obv(dfy, 6, 40)
    dfy['obv 5-74'] = ta.obv(dfy, 5, 74)

    # Shift price data by 1 hour
    dfy['return'] = dfy['return'].shift(-1)
    dfy['move'] = dfy['move'].shift(-1)

    return dfy


combineRawDataFiles()
data = pd.read_csv('RawData.csv', index_col=0, engine='python', dtype=str)
data.index = pd.to_datetime(data.index, errors='coerce')

t1 = time.clock()
print('#---------------------------#')
print('Read Data:\t\t\t{:.1f} sec'.format(t1 - t0))

# Drop duplicates
data.drop_duplicates(subset='tweet', inplace=True)
data.drop_duplicates(subset='uniqueID', inplace=True)

# Add count column
data['count'] = 1

# Process Text
stop_words = stopwords.words('english')
en_words = words.words('en')


def binarySearch(word, wordList):
    first = 0
    last = len(wordList) - 1
    found = False
    while first <= last and not found:
        middle = (first + last) // 2
        if wordList[middle] == word:
            found = True
        else:
            if word < wordList[middle]:
                last = middle - 1
            else:
                first = middle + 1
    return found


def cleanTweet(text, full=False):
    # Remove non-alphanumerics and make lower-case
    cleanText = re.sub(r'[^a-zA-Z ]+', '', str(text)).lower()
    if full:
        noStopWords = [t for t in cleanText.lower().split(' ') if not binarySearch(t, stop_words)]
        cleanText = [n for n in noStopWords if binarySearch(n, en_words)]

    return cleanText


data['text'] = data['tweet'].apply(lambda x: cleanTweet(x))
# data['full clean'] = data['tweet'].apply(lambda x: cleanTweet(x, full=True))


t2 = time.clock()
print('Clean Data:\t\t\t{:.1f} sec'.format(t2 - t1))


# Perform Sentiment Analysis
# Subjectivity / Polarity analysis
def blob(x):
    return x.polarity, x.subjectivity


# Positivity / Neutrality / Negativity analysis
def vad(x):
    return x['pos'], x['neu'], x['neg']


vader = SentimentIntensityAnalyzer()

# Raw tweets
data['blob raw'] = data['tweet'].apply(lambda x: TextBlob(str(x)).sentiment)
data['raw pol'], data['raw sub'] = zip(*data['blob raw'].map(blob))
data['vader raw'] = data['tweet'].apply(lambda x: vader.polarity_scores(str(x)))
data['raw pos'], data['raw neu'], data['raw neg'] = zip(*data['vader raw'].map(vad))
data['raw diff'] = data['raw pos'] - data['raw neg']
data['raw pos n'] = np.where(data['raw diff'] > 0, data['count'], np.NAN)
data['raw neg n'] = np.where(data['raw diff'] < 0, data['count'], np.NAN)

# Cleaned tweets
data['blob clean'] = data['text'].apply(lambda x: TextBlob(''.join(str(x))).sentiment)
data['clean pol'], data['clean sub'] = zip(*data['blob clean'].map(blob))
data['vader clean'] = data['text'].apply(lambda x: vader.polarity_scores(''.join(str(x))))
data['clean pos'], data['clean neu'], data['clean neg'] = zip(*data['vader clean'].map(vad))
data['clean diff'] = data['clean pos'] - data['clean neg']
data['clean pos n'] = np.where(data['clean diff'] > 0, data['count'], np.NAN)
data['clean neg n'] = np.where(data['clean diff'] < 0, data['count'], np.NAN)

# No stop words 'scrubbed' tweets
# data['blob scrubbed'] = data['full clean'].apply(lambda x: TextBlob(''.join(str(x))).sentiment)
# data['scrubbed pol'], data['scrubbed sub'] = zip(*data['blob scrubbed'].map(blob))
# data['vader scrubbed'] = data['full clean'].apply(lambda x: vader.polarity_scores(''.join(str(x))))
# data['scrubbed pos'], data['scrubbed neu'], data['scrubbed neg'] = zip(*data['vader scrubbed'].map(vad))

t3 = time.clock()
print('Sentiment Analysis:\t{:.1f} min'.format((t3 - t2) / 60))


# Split dataframe into cryptocurrency dataframes
def createDFs(oldDF, currencies):
    # Assign indicators for specified currencies
    for c in currencies:
        oldDF[c] = oldDF['tweet'].apply(lambda x: 1 if c in str(x).lower() else 0)  # factor of cryptocurrency

        # Make new dataframe discarding tweets without any information
        oldDF_nonzero = oldDF[abs(oldDF['raw pol']) + abs(oldDF['raw sub'] + abs(oldDF['raw neu'] - 1)) != 0]
        df_dict = {'all': oldDF, 'non zero': oldDF_nonzero}

        # Count total number of tweets in each hour
        sum_odf = ((df_dict['all'][df_dict['all'][c] == 1]).resample('H').sum())

        # See how many minutes had tweets recorded
        min_odf = ((df_dict['all'][df_dict['all'][c] == 1]).resample('T').mean())
        min_odf = min_odf.resample('H').sum()

        for odf in ['all', 'non zero']:
            # Resample into 1 hour periods
            newDF = ((df_dict[odf][df_dict[odf][c] == 1]).resample('H').mean())
            newDF['minutes'] = min_odf['count']
            newDF['weekday'] = newDF.index.weekday  # Record weekday of each hour
            newDF['hour'] = newDF.index.hour  # Record which hour each tweet takes place

            # keep only periods with at least 50 mintues of data collection
            newDF = newDF[newDF['minutes'] >= 50]

            # Normalize counts by weekday and hour
            newDF['count'] = sum_odf['count']
            newDF['count'] *= 60 / newDF['minutes']
            newDF['normed count'] = newDF['count'] - newDF.groupby(['hour', 'weekday'])['count'].transform('mean')
            newDF['normed count'] /= newDF.groupby(['hour', 'weekday'])['count'].transform('std')

            # Calculate sentiment bullishness index (sbi)
            newDF['sbi'] = np.log((1 + sum_odf['raw pos n']) / (1 + sum_odf['raw neg n']))

            # Add market data
            newDF = newDF.join(market(c))
            newDF.dropna(inplace=True)

            # Print each dataframe into a csv file
            saveDF = newDF[
                ['minutes', 'weekday', 'hour', 'count', 'normed count', 'raw pol', 'raw sub', 'raw pos', 'raw neu',
                 'raw neg', 'sbi', 'clean pol', 'clean sub', 'clean pos', 'clean neu', 'clean neg', 'rsi 5', 'rsi 10',
                 'rsi 23', 'rsi 33', 'rsi 62', 'cci 6', 'cci 11', 'cci 45', 'bb 8', 'bb 11', 'bb 74', 'macd 5-35-5',
                 'macd 12-26-9', 'wr 6', 'wr 13', 'wr 48', 'wr 76', 'atr 5', 'atr 14', 'atr 69', 'obv 6-40', 'obv 5-74',
                 'close', 'return', 'move']]
            saveDF.to_csv(r'/Users/jasonbecker/PycharmProjects/CryptoTrading/Training_Data/{}_{}.csv'.format(c, odf))
    return


createDFs(data, settings.searchQuery)
t4 = time.clock()
print('Saving Data:\t\t{:.1f} sec'.format(t4 - t3))
print('#---------------------------#')
print('\nTotal Runtime:\t\t{:.2f} min'.format((t4 - t0) / 60))
print('#---------------------------#')
