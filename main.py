# -*- coding: utf-8 -*-

__author__ = "Mengxuan Chen"
__email__  = "chenmx19@mails.tsinghua.edu.cn"
__date__   = "20191130"

#--* pakages*--
import pandas as pd
import numpy as np
import benchmark
import performance
import MAC_RP
import riskclass_strategy
import returnclass_strategy
import BL_predictreturn
import BL_macro
# please fix Wind python API if you possess a Wind account
from WindPy import *
w.start()
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
#font = FontProperties(fname=r'c:\windows\fonts\simkai.ttf',size = 15)
#plt.rcParams['font.sans-serif'] = 'SimHei'
#plt.rcParams['axes.unicode_minus'] = False

#--* parameters*--
class Para:
    performance_output = '.\\performance_output\\'
    weights_output = '.\\weights_output\\'
    results_output = '.\\results_output\\'
    period = 1  # the time period for trading; freq = month
    rollingtime = 126  #the time period for the data looking
    tau = 0.01
    wmin = 0
    wmax =1
    m = 8 # the number of periods to judge the trends of assets % freq =month
    theta = 24 #params for HP hpfilter
    rf = 0.025 #risk free rate
    RiPmethod = 'variance'
    #selectives：variance #downward #ledoit_wolf #shrinkage_simple #shrinkage_Ledoit #shrinkage_DW
    RePmethod = 'minMDD'
    #selectives：#maxSharp（error）#max_sortino（error）#maxR2MDD（error）#minMDD#maxR2VaR#maxR2CVaR
    VaRmethod='historical'
    #selectives：#weighted_historical#RM#EVT#norm#t#CF
para = Para()

class main:
    def __init__(self):

        start = input("please type the begining date(format: xxxx/xx/xx): ")
        end = input("please type the ending date(format: xxxx/xx/xx) ")

        self.startdate = datetime.strptime(start, '%Y/%m/%d')
        self.enddate = datetime.strptime(end, '%Y/%m/%d')

        self.codes = input("please type the codes of the asset，seperate them with a comma：")
        # 000985.CSI,H11001.CSI,CCFI.WI,AU9999.SGE,SPX.GI,HSI.HI  for all assets
        #  000985.CSI,H11001.CSI,CCFI.WI,AU9999.SGE   for China only
        return
    def data_get(self):
        datas = w.wsd(self.codes, "close", self.startdate, self.enddate, "")
        datas = pd.DataFrame(np.array(datas.Data).T, columns=datas.Codes, index=datas.Times)
        datas.to_excel('datas_final.xlsx')
        datas = pd.read_excel('datas_final.xlsx', index_col=0, parse_dates=True).dropna()
        return datas

if __name__ == "__main__":
    datas = main().data_get()

    print("------------------Equal weighted------------------")
    weights_EW, result_EW=benchmark.benchmark(datas,period=para.period, rollingtime=para.rollingtime,method='EW')
    pd.DataFrame(performance.performance(result_EW)).to_csv(para.performance_output+'result_EW_performance.csv')
    pd.DataFrame(performance.performance_anl(result_EW)).to_csv(para.performance_output+'result_EW_performance_anl.csv')
    weights_EW.to_excel(para.weights_output+'weights_EW.xlsx')
    result_EW.to_excel(para.results_output+'result_EW.xlsx')

    print("------------------Variance equal weighted------------------")
    weights_EV, result_EV=benchmark.benchmark(datas,period=para.period, rollingtime=para.rollingtime,method='EV')
    pd.DataFrame(performance.performance(result_EV)).to_csv(para.performance_output+'result_EV_performance.csv')
    pd.DataFrame(performance.performance_anl(result_EV)).to_csv(para.performance_output+'result_EV_performance_anl.csv')
    weights_EV.to_excel(para.weights_output+'weights_EV.xlsx')
    result_EV.to_excel(para.results_output+'result_EV.xlsx')

    print("------------------Mac weighted------------------")
    weights_nob1,result_nob1 = MAC_RP.withoutboundary(datas,period=para.period, rollingtime=para.rollingtime,
                                                      method='MAC', wmin=para.wmin,wmax=para.wmax)
    pd.DataFrame(performance.performance(result_nob1)).to_csv(para.performance_output+'result_MAC_performance.csv')
    pd.DataFrame(performance.performance_anl(result_nob1)).to_csv(para.performance_output+'result_MAC_performance_anl.csv')
    weights_nob1.to_excel(para.weights_output+'weights_MAC.xlsx')
    result_nob1.to_excel(para.results_output+'result_MAC.xlsx')

    print("------------------RP weighted------------------")
    weights_nob2,result_nob2 = MAC_RP.withoutboundary(datas,period=para.period, rollingtime=para.rollingtime,
                                                      method='RP',wmin=para.wmin,wmax=para.wmax)
    pd.DataFrame(performance.performance(result_nob2)).to_csv(para.performance_output+'result_RP_performance.csv')
    pd.DataFrame(performance.performance_anl(result_nob2)).to_csv(para.performance_output+'result_RP_performance_anl.csv')
    weights_nob2.to_excel(para.weights_output+'weights_RP.xlsx')
    result_nob2.to_excel(para.results_output+'result_RP.xlsx')

    print("------------------Risk Plan------------------")
    weights_RP1,result_RP1 = riskclass_strategy.RP(datas,period=para.period, rollingtime=para.rollingtime,
                                                   method=para.RiPmethod,wmin=para.wmin,wmax=para.wmax)
    pd.DataFrame(performance.performance(result_RP1)).to_csv(para.performance_output+
                                                             'result__'+para.RiPmethod + '_performance.csv')
    pd.DataFrame(performance.performance_anl(result_RP1)).to_csv(para.performance_output+
                                                                 'result__'+para.RiPmethod + '_performance_anl.csv')
    weights_RP1.to_excel(para.weights_output+'weights_riskclass_'+para.RiPmethod + '.xlsx')
    result_RP1.to_excel(para.results_output+'result_riskclass_'+para.RiPmethod + '.xlsx')

    print("------------------Return Plan------------------")
    weights_RP2,result_RP2 = returnclass_strategy.ReturnPlan(datas,period=para.period, rollingtime=para.rollingtime
                                                             ,method=para.RePmethod,wmin=para.wmin,wmax=para.wmax)
    pd.DataFrame(performance.performance(result_RP2)).to_csv(para.performance_output+
                                                             'result__'+ para.RePmethod + '_performance.csv')
    pd.DataFrame(performance.performance_anl(result_RP2)).to_csv(para.performance_output+
                                                                 'result__'+ para.RePmethod + '_performance_anl.csv')
    weights_RP2.to_excel(para.weights_output+'weights_returnclass_'+ para.RePmethod + '.xlsx')
    result_RP2.to_excel(para.results_output+'result_returnclass_'+ para.RePmethod + '.xlsx')

    print("------------------Black Litterman------------------")
    expected_return1=((datas-datas.shift(126))/datas.shift(126))/126
    expected_return1=expected_return1.fillna(method='ffill')
    expected_return1=expected_return1.fillna(0.003)
    weights_BL1,result_BL1 = BL_predictreturn.BL_predictreturn(datas,expected_return1,
                                                               period=para.period, rollingtime=para.rollingtime,
                                                               tau=para.tau,wmin=para.wmin,wmax=para.wmax)
    pd.DataFrame(performance.performance(result_BL1)).to_csv(para.performance_output+
                                                             'result__BL1_performance.csv')
    pd.DataFrame(performance.performance_anl(result_BL1)).to_csv(para.performance_output+
                                                                 'result__BL1_performance_anl.csv')
    weights_BL1.to_excel(para.weights_output+'weights_BL1.xlsx')
    result_BL1.to_excel(para.results_output+'result_BL1.xlsx')

    print("------------------Black Litterman with macro prediction------------------")
    indicator_data = pd.read_excel('indicator_index2.xlsx', index_col=0, parse_dates=True)
    if datas.index[-1]>indicator_data.index[-1]:
        end = datas.index[-1]
    else:
        end = indicator_data.index[-1]
    indicator_data = indicator_data[datas.index[1]:end]
    weights_BL2,result_BL2 = BL_macro.BL_macro(datas,indicator_data, period=para.period, rollingtime=para.rollingtime,
                                               tau=para.tau, wmin=para.wmin,wmax=para.wmax,m=para.m, theta=para.theta,rf=para.rf)
    pd.DataFrame(performance.performance(result_BL2)).to_csv(para.performance_output+
                                                             'result__BL2_performance.csv')
    pd.DataFrame(performance.performance_anl(result_BL2)).to_csv(para.performance_output+
                                                                'result__BL2_performance_anl.csv')
    weights_BL2.to_excel(para.weights_output+'weights_BL2.xlsx')
    result_BL2.to_excel(para.results_output+'result_BL2.xlsx')
