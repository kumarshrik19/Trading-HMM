# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:29:01 2022
@author: oskarfransson
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_download_script import intraday_data, load_intraday_data

import quantstats as qs
import hmmlearn.hmm as hmm
import random

random.seed(1)

ticker = 'QQQ'

data = intraday_data(ticker, '1 min')
data = data.between_time('9:30', '16:00')
data['UnboundedDV2'] = (data.close / ((data.high.rolling(20).max() 
                                          + data.low.rolling(20).min())/2)).rolling(5).mean() - 1
n = np.floor(data.shape[0] * 0.75).astype(int)

def vwap(df):
    n = 78*5
    i = n
    large_vec = np.zeros(df.shape[0])
    while i < df.shape[0]:
        vec = df.iloc[(i-n):i]
        q = vec.volume
        p = vec.close
        large_vec[(i-n):i] = (p * q).cumsum() / q.cumsum()
        
        i = i + n
        
    return pd.Series(large_vec, index = df.index, name = 'vwap')

s1 = vwap(data)
s1 = data.close / s1 - 1
s2 = np.log(data.close/data.close.shift(1)).dropna()
s3 = data.UnboundedDV2.rolling(252).rank(pct = True).dropna()

df = pd.concat([s1, s2, s3], axis = 1).dropna()
df.columns = ['vwap', 'y', 'dv2']
df = df[df.vwap != np.inf]

model = hmm.GaussianHMM(n_components=3, n_iter=1000)
model = model.fit(df.iloc[:n,])

model.transmat_
model.means_

postProb = model.predict_proba(df.iloc[n:,])

d = pd.concat([pd.DataFrame(postProb, index = df.iloc[n:,].index), df.iloc[n:,]], axis = 1)
strat = pd.DataFrame(np.where(d[2] > 0.9, d.y.shift(-1), 
                              np.where(d[0]> 0.9, -d.y.shift(-1), 0)), index = d.index)
strat['chng'] = ((d[2].shift(1) < 0.9) & (d[2] > 0.9)) | ((d[0].shift(1) < 0.9) & (d[0] > 0.9))
strat[0] = strat[0] + np.where(strat.chng, -0.0041/100, 0)

qs.plots.returns(strat[0], ticker)
round(qs.stats.sharpe(strat[0], periods = 78*5*252),3)

#################
### -- OOS -- ###
#################

data = load_intraday_data('SPY', '1m')
data = data.between_time('9:30', '16:00')
data['UnboundedDV2'] = (data.close / ((data.high.rolling(20).max() 
                                          + data.low.rolling(20).min())/2)).rolling(5).mean() - 1

s1 = vwap(data)
s1 = data.close / s1 - 1
s2 = np.log(data.close/data.close.shift(1)).dropna()
s3 = data.UnboundedDV2.rolling(252).rank(pct = True).dropna()

df = pd.concat([s1, s2, s3], axis = 1).dropna()
df.columns = ['vwap', 'y', 'dv2']
df = df[df.vwap != np.inf]

postProb = model.predict_proba(df)
d = pd.concat([pd.DataFrame(postProb, index = df.index), df], axis = 1)
strat = pd.DataFrame(np.where(d[2] > 0.9, d.y.shift(-1), 
                              np.where(d[0]> 0.9, -d.y.shift(-1), 0)), index = df.index).dropna()

plt.plot(range(strat.shape[0]), np.cumprod(1 + strat))
plt.plot(range(strat.shape[0]), np.cumprod(1 + d.iloc[:strat.shape[0]].y))

qs.stats.sharpe(strat, periods = 78*5*252)