# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
from scipy import stats
import ffn

#--* functions for calculate performance and permormance  annually*--
def performance(strategy):
    def MaxDrawdown(return_list):
        RET_ACC = []
        sum = 1
        for i in range(len(return_list)):
            sum = sum * (return_list[i] + 1)
            RET_ACC.append(sum)
        index_j = np.argmax(np.array((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC)))
        index_i = np.argmax(RET_ACC[:index_j])
        MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
        return sum, MDD, RET_ACC

    def sharp(return_list, std):
        returnew = pd.DataFrame(return_list, columns=['R'])
        m = pd.concat([returnew.R], axis=1)
        ret_adj = np.array(m)
        sharpratio = np.average(ret_adj) * 12 / std
        return sharpratio

    def Reward_to_VaR(strategy=strategy, alpha=0.99):
        RET = strategy.nav.pct_change(1).fillna(0)
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        var = abs(sorted_Returns[index])
        RtoVaR = np.average(RET) / var
        return -RtoVaR

    def Reward_to_CVaR(strategy=strategy, alpha=0.99):
        RET = strategy.nav.pct_change(1).fillna(0)
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        sum_var = sorted_Returns[0]
        for i in range(1, index):
            sum_var += sorted_Returns[i]
            CVaR = abs(sum_var / index)
        RtoCVaR = np.average(RET) / CVaR
        return -RtoCVaR

    ts = strategy.nav.pct_change(1).fillna(0)
    RET = (strategy.nav[strategy.shape[0] - 1] /strategy.nav[0])** (252 / strategy.shape[0]) - 1
    T = stats.ttest_1samp(ts, 0)[0]
    STD = np.std(ts) * np.sqrt(252)
    MDD = MaxDrawdown(ts)[1]
    ACC = MaxDrawdown(ts)[0]
    SHARP = (RET - 0.03) / STD
    R2VaR = Reward_to_VaR(strategy)
    R2CVaR = Reward_to_CVaR(strategy)
    print('annual-return', round(RET, 4))
    print('t-statistic', round(T, 4))
    print('volitility', round(STD, 4))
    print('MaxDrawdown', round(MDD, 4))
    print('Accumulated return', round(ACC, 4))
    print('sharp-ratio', round(SHARP, 4))
    print('Reward_to_VaR', round(R2VaR, 4))
    print('Reward_to_CVaR', round(R2CVaR, 4))
    return RET, T, STD, MDD, ACC, SHARP, R2VaR, R2CVaR


def performance_anl(strategy):
    def MaxDrawdown(return_list):
        RET_ACC = []
        sum = 1
        for i in range(len(return_list)):
            sum = sum * (return_list[i] + 1)
            RET_ACC.append(sum)
        index_j = np.argmax(np.array((np.maximum.accumulate(RET_ACC) - RET_ACC) / np.maximum.accumulate(RET_ACC)))
        index_i = np.argmax(RET_ACC[:index_j])
        MDD = (RET_ACC[index_i] - RET_ACC[index_j]) / RET_ACC[index_i]
        return sum, MDD, RET_ACC

    def MaxDrawdown2(return_list):
        value = (1 + return_list).cumprod()
        MDD = ffn.calc_max_drawdown(value)
        return -MDD

    def Reward_to_VaR(strategy=strategy, alpha=0.99):
        RET = strategy.nav.pct_change(1).fillna(0)
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        var = abs(sorted_Returns[index])
        RtoVaR = np.average(RET) / var
        return -RtoVaR

    def Reward_to_CVaR(strategy=strategy, alpha=0.99):
        RET = strategy.nav.pct_change(1).fillna(0)
        sorted_Returns = np.sort(RET)
        index = int(alpha * len(sorted_Returns))
        sum_var = sorted_Returns[0]
        for i in range(1, index):
            sum_var += sorted_Returns[i]
            CVaR = abs(sum_var / index)
        RtoCVaR = np.average(RET) / CVaR
        return -RtoCVaR

    strategy['Y'] = strategy.index
    strategy['Y'] = strategy.Y.apply(lambda x: x.year)
    n_year = strategy['Y'].value_counts()
    n_year_index = n_year.index.sort_values()
    RET_list = []
    T_list = []
    STD_list = []
    MDD_list = []
    ACC_list = []
    SHARP_list = []
    R2VaR_list = []
    R2CVaR_list = []
    for i in n_year_index:
        x = strategy.loc[strategy['Y'] == i]
        ts = x.nav.pct_change(1).fillna(0)
        # ts = ts.tolist()
        # print(ts)
        RET = (x.nav[x.shape[0] - 1] /x.nav[0])** (252 / x.shape[0]) - 1
        T = stats.ttest_1samp(ts, 0)[0]
        STD = np.std(ts) * np.sqrt(252)
        MDD = MaxDrawdown(ts)[1]
        # MDD = MaxDrawdown2(ts)
        ACC = MaxDrawdown(ts)[0]
        SHARP = (RET - 0.03) / STD
        R2VaR = Reward_to_VaR(x)
        R2CVaR = Reward_to_CVaR(x)
        RET_list.append(RET)
        T_list.append(T)
        STD_list.append(STD)
        MDD_list.append(MDD)
        ACC_list.append(ACC)
        SHARP_list.append(SHARP)
        R2VaR_list.append(R2VaR)
        R2CVaR_list.append(R2CVaR)
    RET_df = pd.DataFrame(RET_list)
    RET_df.columns={'RET'}
    T_df = pd.DataFrame(T_list)
    T_df.columns={'T'}
    STD_df = pd.DataFrame(STD_list)
    STD_df.columns={'STD'}
    MDD_df = pd.DataFrame(MDD_list)
    MDD_df.columns={'MDD'}
    ACC_df = pd.DataFrame(ACC_list)
    ACC_df.columns={'ACC'}
    SHARP_df = pd.DataFrame(SHARP_list)
    SHARP_df.columns={'SHARP'}
    R2VaR_df = pd.DataFrame(R2VaR_list)
    R2VaR_df.columns={'R2VaR'}
    R2CVaR_df = pd.DataFrame(R2CVaR_list)
    R2CVaR_df.columns={'R2CVaR'}
    P = pd.concat([RET_df, T_df, STD_df, MDD_df, ACC_df, SHARP_df, R2VaR_df, R2CVaR_df], axis=1, ignore_index=False)
    P.index = n_year_index
    print(P)
    return P