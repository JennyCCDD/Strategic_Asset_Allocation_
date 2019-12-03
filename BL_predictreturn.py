# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy.optimize import minimize

#--* Black Litterman Model*--
def blacklitterman(returns, Tau, P, Q):
    mu = returns.mean()
    sigma = returns.cov()
    pi1 = mu
    ts = Tau * sigma
    Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
    middle = inv(np.dot(np.dot(P, ts), P.T) + Omega)
    er = np.expand_dims(pi1, axis=0).T + np.dot(np.dot(np.dot(ts, P.T), middle),
                                                (Q - np.expand_dims(np.dot(P, pi1.T), axis=1)))
    newList = []
    for item in er:
        if type(item) == list:
            tmp = ''
            for i in item:
                tmp += float(i) + ' '
                newList.append(tmp)
        else:
            newList.append(item)
    New = []
    for j in newList:
        k = float(j)
        New.append(k)
    posteriorSigma = sigma + ts - np.dot(ts.dot(P.T).dot(middle).dot(P), ts)
    return [New, posteriorSigma]

def funs(weight, sigma):
    weight = np.array([weight]).T
    result = np.dot(np.dot(weight.T, np.mat(sigma)), weight)[0, 0]
    return (result)

def BL_predictreturn(datas, expected_return, period, rollingtime, tau=0.01, wmin=0,wmax=1):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas / datas.iloc[0,] * 1000
    result = data_norm.copy()
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x: x.month)
    weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    noa = datas.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = tuple((wmin, wmax) for i in range(datas.shape[1]))
    pick1 = np.array([1 for i in range(noa)])  # predicted returns for all assets
    pick21 = [0 for i in range(noa - 2)]
    pick22 = [0.5, -0.5]
    pick2 = np.array(pick22 + pick21)  # premiums between the frist asset and the second asset
    P = np.array([pick1, pick2])
    for i in tqdm(range(result.shape[0])):
        if i == 0:
            weights.iloc[i, :] = 1 / datas.shape[1]
            price = datas.loc[datas.index[i], :]
            n = weights.iloc[i, :].values / price.values
            N.loc[result.index[i], :] = n
            del price, n
        elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
            Rett = ret[i - int(rollingtime):i]
            expected_return['sum'] = expected_return.sum(axis=1)
            expected_return['premium'] = expected_return.iloc[:, 0] - expected_return.iloc[:, 1]
            Q = expected_return.iloc[i:i + 1, (expected_return.shape[1] - 2):expected_return.shape[1]].T.values
            #Returns = blacklitterman(Rett, tau, P, Q)[0]
            sigma = blacklitterman(Rett, tau, P, Q)[1]

            weight = [0 for i in range(datas.shape[1])]
            res = minimize(funs, weight, method='SLSQP', args=(sigma,),
                               bounds=bnds, constraints=cons, tol=1e-8)

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

