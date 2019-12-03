# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from statsmodels.tsa.filters.hp_filter import hpfilter
import scipy.stats as stats
from sklearn.decomposition import PCA
from numpy.linalg import inv
from scipy.optimize import minimize

#--* define objective functions*--
def funs2(weight, mu, sigma, delta):
    result = 0.5*delta*np.dot(np.dot(weight.T,sigma),weight)-np.dot(weight.T,mu)
    return [result]

#--*calculate PCA*--
def pca(used_macro_data, theta):
    components = 6  # 取多少个主成分
    index = used_macro_data.index
    pca = PCA(n_components=components,svd_solver='auto')
    explained_variance = pca.fit(used_macro_data).explained_variance_ratio_
    weight = np.array(explained_variance) / sum(explained_variance)
    pca_data = pca.fit_transform(used_macro_data)
    pca_data = pd.DataFrame(pca_data, columns=['fac%d' % i for i in range(1, components + 1)], index=index)
    pca_data = pca_data.diff() / pca_data.shift()  # 计算变化率
    pca_data.dropna(inplace=True)
    # pca_data = pca_data.apply(lambda x: winsorize(x,limits = (0.02,0.02)))
    pca_data[pca_data > 20] = 20
    pca_data[pca_data < -20] = -20
    pca_data = pca_data.apply(lambda x: hpfilter(x, theta)[1])
    return (pca_data, weight)

#--*calculate similarity*--
def similarity(df, n, weight):
    data_now = df.loc[df.index[-1] - (n - 1) * MonthEnd():,:]
    # 将全部样本中最后n个月的数据作为需比较相似性的数据
    first_date = df.index[0]  # get the date of the first input
    correlation = pd.Series(name='correlation')
    for i in range(len(df) - 2*n):
        temp_corr = []
        start = first_date + i * MonthEnd()
        end = start + (n - 1) * MonthEnd()
        temp = df.loc[start:end, :].copy()
        for j in range(len(df.columns)):
            corr = stats.pearsonr(temp.iloc[:,j].values,data_now.iloc[:,j].values)
            temp_corr.append(corr[0])
        mean_corr = sum(np.array(temp_corr)*weight)
        index = start.strftime('%Y-%m-%d') + ':' + end.strftime('%Y-%m-%d')
        correlation[index] = mean_corr
    correlation.sort_values(ascending=False, inplace=True)
    return correlation

#--*calculate parameters*--
def compute_parameter(m, n, t, theta, rf, used_equity_data, used_macro_data, t_month_return):
    equities = used_equity_data.columns
    k = len(equities)
    mkt_return = used_equity_data.apply(lambda x: x.mean(), axis=1)
    delta = abs((mkt_return.mean()-rf)/(mkt_return.std())**2)  # params for risk aversion
    prior_cov = used_equity_data.cov()  # prior covirance matrix
    P = np.identity(k)  # subjective matrix
    weight = np.array([1/k for i in range(k)])
    pi = delta*np.dot(prior_cov,weight)
    # pi = used_equity_data.mean()
    #used_macro_data[rollingtime]
    pca_data, weight = pca(used_macro_data,theta)
    rank = similarity(pca_data, n, weight)
    head_m = [datetime.strptime(x[1], '%Y-%m-%d') + MonthEnd(t) for x in rank.head(m).index.str.split(':')]
    # get m periods of t months which is most similar to the input time
    similar_data = t_month_return.loc[head_m, :]
    # get the situation of the assets in this m periods of t months
    mean_return = similar_data.mean()
    Q = mean_return.values.T  # k*1 of subjective matrix
    point_variance = np.power(similar_data.std(),2)
    Omega = np.diag(point_variance.values)  # k*k of subjective error matrix
    return (delta, prior_cov, pi, P, Q, Omega)

#--* Black Litterman Model*--
def BL_macro(datas, indicator_data, period, rollingtime,m=8,theta=24,tau=0.01, wmin=0,wmax=1,rf=0.025):
    ret = datas.pct_change(1)
    ret_rolling = datas.pct_change(rollingtime)
    ret_M=ret.resample('M').last()
    ret_M.dropna(inplace=True)
    ret_rolling_M=ret_rolling.resample('M').last()
    ret_rolling_M.dropna(inplace=True)
    data_norm = datas / datas.iloc[0,] * 1000
    result = data_norm.copy()
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x: x.month)
    weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    for i in tqdm(range(result.shape[0]-rollingtime)):
        if i == 0:
            weights.iloc[i, :] = 1 / datas.shape[1]
            price = datas.loc[datas.index[i], :]
            n = weights.iloc[i, :].values / price.values
            N.loc[result.index[i], :] = n
            del price, n
        elif (result.m[i] != result.m[i - 1]) and \
                (i<30*(len(indicator_data)-period))and\
                (i>30*m) and \
                (i > int(rollingtime)) and result.m[i] % int(period) == 0:
            Rett = ret[i - int(rollingtime):i]
            Rett_M = Rett.resample('M').last()
            Rett_M.dropna(inplace=True)
            Indicator_data = indicator_data[:i]
            Indicator_data_M = Indicator_data.resample('M').last()
            Indicator_data_M = Indicator_data_M.apply(lambda x: (x - x.mean()) / x.std())
            Indicator_data_M.dropna(inplace=True)
            n = int(rollingtime/120)
            delta, prior_cov, pi, P, Q, Omega = compute_parameter(m, n , period, theta, rf, Rett_M,Indicator_data_M,ret_rolling_M)
            post_pi_left = inv(inv(tau * prior_cov) + np.dot(np.dot(P.T, inv(Omega)), P))
            post_pi_right = np.dot(inv(tau * prior_cov), pi) + np.dot(np.dot(P.T, inv(Omega)), Q)
            post_pi = np.dot(post_pi_left, post_pi_right)
            post_cov = prior_cov + inv(inv(tau * prior_cov) + np.dot(np.dot(P.T, inv(Omega)), P))
            cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
            bnds = tuple((wmin, wmax) for i in range(datas.shape[1]))
            weight = [0 for i in range(datas.shape[1])]
            res = minimize(funs2, weight, method='SLSQP', args=(post_pi, post_cov, delta,),
                               bounds=bnds, constraints=cons, tol=1e-8)
            weights.iloc[i, :] = res.x
            #print(res.x)
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