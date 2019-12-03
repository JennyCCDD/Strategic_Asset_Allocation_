# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from tqdm import tqdm

#--* benchmark models*--
def benchmark(datas, period, rollingtime, method):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas / datas.iloc[0,] * 1000
    result = data_norm.copy()
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x: x.month)
    weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    position = 0
    if method == 'EW':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1]) and i > int(rollingtime) and result.m[i] % int(period) == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            else:
                N.iloc[i, :] = N.iloc[i - 1, :]
                w = N.iloc[i, :] * datas.loc[datas.index[i], :]
                weights.iloc[i, :] = w / w.sum()
    elif method == 'EV':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1]) and i > int(rollingtime) and result.m[i] % int(period) == 0:
                vol = ret.iloc[position:i].std()
                position = i
                weights.iloc[i, :] = (1 / vol) / ((1 / vol).sum())
                price = datas.loc[datas.index[i], :]
                V = (weights.iloc[i, :] * price).sum()
                n = V * weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n

            else:
                N.iloc[i, :] = N.iloc[i - 1, :]
                w = N.iloc[i, :] * datas.loc[datas.index[i], :]

                weights.iloc[i, :] = w / w.sum()
    else:
        pass

    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in tqdm(range(result.shape[0])):
        result.loc[result.index[i], 'mv'] = (datas.iloc[i, :] * N.iloc[i, :]).sum()
        if all(N.iloc[i, :] == 0):
            pass
        elif all(N.iloc[i, :] == N.iloc[i - 1, :]):
            result.loc[result.index[i], 'mv_adj_last_day'] = result.loc[result.index[i - 1], 'mv']
            result.loc[result.index[i], 'nav'] = result.nav[i - 1] * result.mv[i] / result.mv_adj_last_day[i]
        else:
            result.loc[result.index[i], 'mv_adj_last_day'] = (datas.iloc[i - 1, :] * N.iloc[i, :]).sum()
            result.loc[result.index[i], 'nav'] = result.nav[i - 1] * result.mv[i] / result.mv_adj_last_day[i]

    result = result.iloc[int(rollingtime):, :]
    weights = weights.iloc[int(rollingtime):, :]
    result['nav'] = result.nav / result.nav[0] * 1000
    return weights, result