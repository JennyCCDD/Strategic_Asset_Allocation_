# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy import optimize
from scipy.special import gammaln
import ffn
from scipy.stats import norm,t

"""
def SharpRatio(ret):
    ret.dropna()
    meanreturn = np.average(ret)
    SR = meanreturn*252 /(np.std(ret) * np.sqrt(252))
    return (SR)
"""
#--* define objective functions*--
def max_sharpe(weights,Returns):
    weights = np.array(weights)
    RET = np.sum(Returns * weights, axis=1)
    SR = ffn.calc_sharpe(RET)#SR = SharpRatio(RET)
    return -SR
def max_sortino(weights,Returns):
    weights = np.array(weights)
    RET = np.sum(Returns * weights, axis=1)
    sortino =ffn.calc_sortino_ratio(RET)
    return -sortino
def maxR2MDD(weights, Returns):
    weights = np.array(weights)
    RET = np.sum(Returns * weights, axis=1)
    value = (RET+1).cumprod()
    MDD = ffn.calc_max_drawdown(value)
    R2MDDadj = (np.average(RET) * 252)/(MDD)
    return -R2MDDadj
def minMDD(weights, Returns):
    weights = np.array(weights)
    RET = np.sum(Returns * weights, axis=1)
    value = (RET+1).cumprod()
    MDD = ffn.calc_max_drawdown(value)
    return (-MDD)
#define various VaR
def getNegativeLoglikelihood(d,r):
    LogLikeLihood = r.shape[0]*(gammaln((d+1)/2) - gammaln(d/2)
                                -np.log(np.pi)/2 - np.log(d-2)/2) \
                    - 1/2*(1+d)*np.log(r['z']**2/(d-2)+1).sum()
    return -LogLikeLihood
def Garch_VaR(RET):
    RET['return'] = np.log(1+RET)
    RET['sigma'] = RET['sigma2']**0.5
    RET['z'] = RET['return']/RET['sigma']
    RET = RET.dropna()
    RET = RET.reset_index(drop = True)
    d_best = optimize.fmin(getNegativeLoglikelihood,np.array([10]),
                           args=(RET['return'] ,),ftol = 0.000000001)
    T = RET.shape[0]
    Tu = 50
    u = RET.z.sort_values().values[Tu]
    xi = 1/50*np.log(RET.z.sort_values().values[:Tu]/u).sum()
    c = Tu/T*abs(u)**(1/xi)
    phi =  norm(0,1).ppf(p)
    VAR_EVT = -RET['sigma']*u*(p/(Tu/T))**(-xi)
    VAR_norm = - RET['sigma']*phi
    VAR_t= - RET['sigma']*t(d_best[0]).ppf(p)*((d_best[0] - 2)/d_best[0])**0.5
    VAR_CF= - RET['sigma']*(phi + RET.z.skew()/6*(phi**2-1) +
                            RET.z.kurt()/24*(phi**3-3*phi) - RET.z.skew()**2/36*(2*phi**3-5*phi))
    return VAR_EVT,VAR_norm,VAR_t,VAR_CF

def maxR2VaR(weights,Returns,method='historical',
             alpha=0.99,
             window = 250,eta=0.99):
    weights = np.array(weights)
    RET = np.sum(Returns *weights,axis=1)
    if method == 'historical':
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        var = abs(sorted_Returns[index])
    elif method =='weighted_historical':
        Weights = eta ** (np.arange(250,0,-1) - 1)*(1 - eta)/(1 - eta** window)
        var = np.sort(-RET)[np.min(np.where(Weights[np.argsort(-RET)].cumsum()> 0.99))]
    elif method =='RM':
        RET = pd.DataFrame(RET)
        sigma2 = pd.DataFrame(columns=RET.columns, index=RET.index)
        for i in range(RET.shape[0] - 2):
            sigma2.loc[i + 1,:] = sigma2.loc[i,:]*0.94 + 0.06* RET.loc[i,:]**2
        var= -sigma2**0.5 * norm(0,1).ppf(0.01)*np.sqrt(10)
    elif method =='EVT':
        var = Garch_VaR(RET)[0]
    elif method =='norm':
        var = Garch_VaR(RET)[1]
    elif method =='t':
        var = Garch_VaR(RET)[2]
    elif method =='CF':
        var = Garch_VaR(RET)[3]
    else:
        pass
    result = var/(np.average(RET)++10000000)
    return(result)
def maxR2CVaR(weights,Returns,alpha=0.99):
    weights = np.array(weights)
    RET = np.sum(Returns *weights,axis=1)
    sorted_Returns = np.sort(RET)
    index = int(alpha * len(sorted_Returns))
    sum_var = sorted_Returns[0]
    for i in range(1, index):
        sum_var += sorted_Returns[i]
    CVaR=abs(sum_var / index)
    result = CVaR/(np.average(RET)++10000000)
    return(result)

#--*Return Plan models*--
def ReturnPlan(datas, period, rollingtime, method, wmin=0,wmax=1):
    ret = datas.pct_change(1).dropna()#.fillna(0)
    datas = datas.dropna()
    data_norm = datas / datas.iloc[0,] * 1000
    result = data_norm.copy()
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x: x.month)
    weights = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    N = pd.DataFrame(columns=datas.columns, index=datas.index).fillna(0)  #
    datas_index = np.array(datas.index)
    noa = datas.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = tuple((wmin, wmax) for i in range(datas.shape[1]))
    if method == 'maxSharp':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
                Returns = ret.iloc[i - int(rollingtime):i]
                weight = [0 for i in range(datas.shape[1])]
                res = minimize(max_sharpe, weight, method='SLSQP', args=(Returns,),
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

    if method == 'max_sortino':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
                Returns = ret.iloc[i - int(rollingtime):i]
                weight = [0 for i in range(datas.shape[1])]
                res = minimize(max_sortino, weight, method='SLSQP', args=(Returns,),
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
    elif method == 'minMDD':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
                Returns = ret.iloc[i - int(rollingtime):i]
                weight = [0 for i in range(datas.shape[1])]
                res = minimize(minMDD, weight, method='SLSQP', args=(Returns,),
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
    elif method == 'maxR2VaR':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
                Returns = ret.iloc[i - int(rollingtime):i]
                weight = [0 for i in range(datas.shape[1])]
                res = minimize(maxR2VaR, weight, method='SLSQP', args=(Returns,),
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
    elif method == 'maxR2CVaR':
        for i in tqdm(range(result.shape[0])):
            if i == 0:
                weights.iloc[i, :] = 1 / datas.shape[1]
                price = datas.loc[datas.index[i], :]
                n = weights.iloc[i, :].values / price.values
                N.loc[result.index[i], :] = n
            elif (result.m[i] != result.m[i - 1]) and (i > int(rollingtime)) and result.m[i] % int(period) == 0:
                Returns = ret.iloc[i - int(rollingtime):i]
                weight = [0 for i in range(datas.shape[1])]
                res = minimize(maxR2CVaR, weight, method='SLSQP', args=(Returns,),
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
