# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.covariance import  ledoit_wolf

#--* define objective functions*--
def funsRP(weight,sigma):
    weight = np.array([weight]).T
    X = np.multiply(weight,np.dot(sigma.values,weight))
    result = np.square(np.dot(X,np.ones([1,X.shape[0]])) - X.T).sum()
    return(result)
def downward(RET,rollingtime=126):
    i=130
    data_cov = RET.iloc[i-int(rollingtime):i]
    data_cov[data_cov>0] = 0
    sigma = data_cov.cov()
    return sigma
def getSigma(datas, method='Simple'):
    asset = datas.columns
    datas['n'] = np.arange(datas.shape[0])
    datas['group'] = pd.qcut(datas.n, 4, labels=False)
    weights = np.arange(1, datas.shape[1]) / 10

    if method == 'Simple':
        sigma_1 = datas.loc[datas.group == 0, asset].cov()
        sigma_2 = datas.loc[datas.group == 1, asset].cov()
        sigma_3 = datas.loc[datas.group == 2, asset].cov()
        sigma_4 = datas.loc[datas.group == 3, asset].cov()
        sigma = 0.1 * sigma_1 + sigma_2 * 0.2 + sigma_3 * 0.3 + sigma_4 * 0.4
    elif method == 'Ledoit':
        sigma_1, a = ledoit_wolf(datas.loc[datas.group == 0, asset])
        sigma_2, a = ledoit_wolf(datas.loc[datas.group == 1, asset])
        sigma_3, a = ledoit_wolf(datas.loc[datas.group == 2, asset])
        sigma_4, a = ledoit_wolf(datas.loc[datas.group == 3, asset])
        sigma = 0.1 * sigma_1 + sigma_2 * 0.2 + sigma_3 * 0.3 + sigma_4 * 0.4
        sigma = pd.DataFrame(sigma)
    elif method == 'DW':
        datas[datas > 0] = 0
        datas['n'] = np.arange(datas.shape[0])
        datas['group'] = pd.qcut(datas.n, 4, labels=False)
        sigma_1 = datas.loc[datas.group == 0, asset].cov()
        sigma_2 = datas.loc[datas.group == 1, asset].cov()
        sigma_3 = datas.loc[datas.group == 2, asset].cov()
        sigma_4 = datas.loc[datas.group == 3, asset].cov()
        sigma = 0.1 * sigma_1 + sigma_2 * 0.2 + sigma_3 * 0.3 + sigma_4 * 0.4
    else:
        pass
    return sigma

#--*Rick Plan models*--
def RP(datas, period, rollingtime, method, wmin=0,wmax=1):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas / datas.iloc[0,] * 1000
    result = data_norm.copy()
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x: x.month)
    weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = tuple((wmin, wmax) for i in range(datas.shape[1]))
    for i in tqdm(range(result.shape[0])):
        if i == 0:
            weights.iloc[i, :] = 1 / datas.shape[1]
            price = datas.loc[datas.index[i], :]
            n = weights.iloc[i, :].values / price.values
            N.loc[result.index[i], :] = n
        elif (result.m[i] != result.m[i - 1]) and i > int(rollingtime) and result.m[i] % int(period) == 0:
            if method == 'variance':
                sigma = ret.iloc[i - int(rollingtime):i].cov()
            elif method == 'downward':
                sigma = downward(ret.iloc[i - int(rollingtime):i])
            elif method =='ledoit_wolf':
                sigma , a = ledoit_wolf(ret.iloc[i - int(rollingtime):i])
                sigma = pd.DataFrame(sigma)
            elif method == 'shrinkage_simple':
                sigma = getSigma(ret.iloc[i - int(rollingtime):i], method='Simple')
            elif method == 'shrinkage_DW':
                sigma = getSigma(ret.iloc[i - int(rollingtime):i], method='DW')
            elif method == 'shrinkage_Ledoit':
                sigma = getSigma(ret.iloc[i - int(rollingtime):i], method='Ledoit')
            else:
                pass
            weight = [0 for i in range(datas.shape[1])]
            res = minimize(funsRP, weight, method='SLSQP', args=(sigma,),
                           bounds=bnds, constraints=cons, tol=1e-20)
            weights.iloc[i, :] = res.x
            price = datas.loc[datas.index[i], :]
            V = (weights.iloc[i, :] * price).sum()
            n = V * weights.iloc[i, :].values / price.values
            N.loc[result.index[i], :] = n
        else:
            N.iloc[i, :] = N.iloc[i - 1, :]
            w = N.iloc[i, :] * datas.loc[datas.index[i], :]
            weights.iloc[i, :] = w / w.sum()

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