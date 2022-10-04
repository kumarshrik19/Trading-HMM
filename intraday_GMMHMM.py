# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:29:01 2022
@author: oskarfransson
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_download_script import intraday_data

import quantstats as qs
import hmmlearn.hmm as hmm

data = intraday_data('SPY', '1 min')
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
s1 = data.close.head(s1.shape[0] - 331) / s1.head(s1.shape[0] - 331) - 1
s2 = np.log(data.close/data.close.shift(1)).dropna()
s3 = data.UnboundedDV2.rolling(252).rank(pct = True).dropna()

df = pd.concat([s1, s2, s3], axis = 1).dropna()
df.columns = ['vwap', 'y', 'dv2']

model = hmm.GaussianHMM(n_components=3, n_iter=1000)
model = model.fit(df.iloc[:n,])
postProb = model.predict_proba(df.iloc[n:,])

d = pd.concat([pd.DataFrame(postProb, index = df.iloc[n:,].index), df.iloc[n:,]], axis = 1)
strat = pd.DataFrame(np.where(d[2] > 0.9, d.y.shift(-1), 
                              np.where(d[0]> 0.9, -d.y.shift(-1), 0)), index = d.index)

qs.plots.returns(strat, 'SPY')
qs.stats.sharpe(strat, periods = 78*5*252)

