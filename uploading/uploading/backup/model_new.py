# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:08:37 2018

@author: 18810
"""

import os
import pymysql
import numpy as np
import pandas as pd
import Functions_new as fc
from logistic_data import loadData
from pandas import DataFrame as df
from itertools import combinations
import time
import pickle
import statsmodels.api as sm
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score,accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.model_selection import train_test_split

# path = os.getcwd()
# con = pymysql.connect('10.81.88.253', 'spider', 'spider1234', 'spark_text', charset='utf8')
# sql = 'select * from test_data_360'
# data = pd.read_sql(sql, con)
# con.close()
# data.columns = [i.replace('.', '_') for i in data.columns]
#
# ziduan_num = pd.read_excel("变量.xlsx", sheetname='数值')  ##添加年龄、性别、省份、申请金额、申请时间、人脸识别分、同盾相关内容字段
# ziduan_str = pd.read_excel("变量.xlsx", sheetname='字符')
# ziduan_zeng = pd.read_excel("变量.xlsx", sheetname='增长率')  ##增长率字段
# ziduan_plus = {'神州融': 'szr_', '银联智策': 'yl_', '聚信立（mg蜜罐）': 'mg_', '索伦': 'sl_', '新颜': 'xy_', '数据宝': 'sjb_',
#                '融360': 'r360_'}
# colsNum = list(df(ziduan_num['三方名称'].replace(ziduan_plus) + ziduan_num['字段名'])[0])
# colsStr = list(df(ziduan_str['三方名称'].replace(ziduan_plus) + ziduan_str['字段名'])[0])
# colsZeng = list(df(ziduan_zeng['三方名称'].replace(ziduan_plus) + ziduan_zeng['字段名'])[0])  ##字符字段

# colsSpec=['yl_CSRL001','yl_CSSS001']

###'yl.CSRL001'：风险得分;'yl.CSSS001'：消费自由度得分
# for col in colsSpec:
#    data[col].replace({9990:-1,9991:10000,9992:-3,9993:-2,np.nan:-4},inplace=True)

###'yl.CSRL003'：套现模型得分
# data['yl_CSRL003'].replace({9990:-1,9991:10000,np.nan:20000},inplace=True)
#

def training(data, colsNum, colsStr, colsTarget, result_folder_path):
    # === hyper parameters
    missThreNum = 0.6;
    missThreStr = 0.6;
    maxBinThre = 0.6;
    maxBins = 20;
    pValue = 0.1
    # target = 'y'
    target = colsTarget[0]
    idCol = 'idno'
    ivMinThre = 0.3;
    roh_thresould = 0.8
    monotonic_threshold = 0.95

    ## ============  数值 drop column mostly nan
    colsMoveNum = []
    missPcntNum = {}
    for col in colsNum:
        # print(col)
        missPcntPer = fc.MissingPcnt(data, col)  # get num of nan in a column
        missPcntNum[col] = missPcntPer
        # if missPcntPer <= 0:
        #     pass
        # elif missPcntPer < missThreNum:
        #     data[col] = fc.RandomSample(data, col)  # use random sample to fill nan????
        # else:
        #     colsMoveNum.extend([col])
        if missPcntPer >= missThreNum:
            colsMoveNum.extend([col])  # drop feature if too much nan
    colsNum = list(set(colsNum) - set(colsMoveNum))  ##200
    print('drop mostly nan num columns:')
    print(colsMoveNum)

    ## ============== 字符 drop column mostly nan
    colsMoveStr = []
    missPcntStr = {}
    for col in colsStr:
        # print(col)
        missPcntPer = fc.MissingPcnt(data, col)
        missPcntStr[col] = missPcntPer
        if missPcntPer <= 0:
            data.loc[:, col] = data[col].replace({'"null"': 'null', '': 'null', 'nan': 'null'})
        elif missPcntPer < missThreStr:
            data.loc[data[col].isnull(), col] = 'null' # fill nan with 'null'
            data.loc[:, col] = data[col].replace({'"null"': 'null', '': 'null', 'nan': 'null'})
        else:
            colsMoveStr.extend([col])
    colsStr = list(set(colsStr) - set(colsMoveStr))  ##33
    print('drop mostly nan str columns:')
    print(colsMoveStr)
    #

    # df({'missingPcnt':missPcntNum}).to_excel('missPcntNum.xlsx')
    # df({'missingPcnt':missPcntStr}).to_excel('missPcntStr.xlsx')


    ## =============== 数值 drop column with few different values
    maxBinMoveNum = []
    for col in colsNum:
        # print(col)
        maxBinPcnt = fc.MaximumBinPcnt(data, col)
        if maxBinPcnt >= maxBinThre:  #  most data with the same value
            maxBinMoveNum.extend([col])
    colsNum = list(set(colsNum) - set(maxBinMoveNum))  ##194
    print('drop num columns with too few bins:')
    print(maxBinMoveNum)

    ## ================== 字符 drop column with few different values
    maxBinMoveStr = []
    for col in colsStr:
        # print(col)
        maxBinPcnt = fc.MaximumBinPcnt(data, col)
        if maxBinPcnt >= maxBinThre:
            maxBinMoveStr.extend([col])
    colsStr = list(set(colsStr) - set(maxBinMoveStr))  ##22
    print('drop str columns with too few bins:')
    print(maxBinMoveStr)
    # data[maxBinMoveNum].to_excel('maxBinMoveNum.xlsx')
    # data[maxBinMoveStr].to_excel('maxBinMoveStr.xlsx')


    # colsAll=colsNum+colsStr+['y']


    ## ================  数值 features to bins
    startM = time.time()
    maxBinDelNum = []
    cutPointsDict = {}
    badRateNum = df()

    real_col_num = []
    cols_num_cut_off_points = {}
    for col in colsNum:
        print(col)
        cutOffPoints = fc.ChiMerge_MaxInterval_Original_Num(data, col, target,
                                max_interval=maxBins, pvalue=pValue)  # find the cutting points of each featurue, according to Chi-square of each value, combining them into an interval

        # if len(cutOffPoints) > maxBins:
        #     tmp_pvalue = 0.05  # lower pvale, try to reduce bins num
        #     cutOffPoints = fc.ChiMerge_MaxInterval_Original_Num(data, col, target,
        #                                                         max_interval=maxBins, pvalue=tmp_pvalue)
        #
        # if len(cutOffPoints) > maxBins:
        #     tmp_pvalue = 0.01  # lower pvale, try to reduce bins num
        #     cutOffPoints = fc.ChiMerge_MaxInterval_Original_Num(data, col, target,
        #                                                         max_interval=maxBins, pvalue=tmp_pvalue)


        print('bin num:', len(cutOffPoints))
        # if cutOffPoints == []:
        if len(cutOffPoints) == 1:  # only one bin, drop this feature
            continue
        else:
            data[col + '_Bin'] = data[col].map(lambda x: fc.AssignBinNum(x, cutOffPoints))  # assign each value to an interval

            badRate = df(
                fc.BadRateEncoding(data, col + '_Bin', target)['br_rate'])  # calculate bad rate for each interval

            tmp_rates = badRate.iloc[0]
            tmp_monotonic = np.corrcoef(tmp_rates.index, tmp_rates.values)[0, 1]  # correlation of bin num and bin bad rates
            print('absolute cof:', abs(tmp_monotonic))
            if abs(tmp_monotonic) < monotonic_threshold:  # drop features with non-monotonic
                continue

            badRate.index = [col]
            badRateNum = pd.concat([badRateNum, badRate], axis=0, join='outer')

            # exist0 = fc.Exist0(data, col + '_Bin', target)  # check if any bin of this variable is all 0 or all 1

            # if len(exist0) == 2:  # exist all 0 or all 1 ('bad', True), or ('good', True), or (False)
            #     result = fc.MergeNum0(data, col + '_Bin', target)  # combine interval that is all 0 or all 1
            #     data[col + '_Bin'] = result['colBin']
            #     cutOffPoints = result['cutOffPoints']
            # else:
            #     pass

            maxBinPcnt = fc.MaximumBinPcnt(data, col + '_Bin')  # interval with largest portion of data
            print('largest bin pct:', maxBinPcnt)

            real_col_num.append(col)  # retore the column name into the buffer
            cols_num_cut_off_points[col] = cutOffPoints  # record cut points for seperating bins

    endM = time.time()
    timeM = (endM - startM) / 60
    print('M:', timeM)

    # badRateDfM.index = colsNum
    badRateNum.to_excel('%s\\badRateNumM.xlsx' % result_folder_path)
    endM = time.time()
    timeM = (endM - startM) / 60
    print('M:', timeM)                   # finish dividing bins for numeric type features

    # badRateDfM.index=colsNum
    # badRateDfM.to_excel('badRateNumM0627.xlsx')


    # U型
    # startU=time.time()
    # colsU=
    # badRateDfU=df()
    #
    # for col in colsU:
    #     print(col)

    #    cutOffPoints=fc.ChiMerge_MaxInterval_Original(data, col, target, max_interval = maxBins)
    #    print(111111,col)
    #    data[col+'_Bin']=data[col].map(lambda x: fc.AssignBin(x,cutOffPoints))
    #    exist0=fc.Exist0(data,col+'_Bin',target)

    #    if len(exist0)==2:
    #        result=fc.MergeNum0(data,col+'_Bin',target)
    #        data[col+'_Bin']=result['colBin']
    #        cutOffPoints=result['cutOffPoints']
    #    else:
    #        pass
    #
    #    parabola=fc.Parabola(data,col+'_Bin',target)
    #
    #    len_cut=len(cutOffPoints)
    #    while parabola==False:
    #        cutOffPointsBin=fc.ChiMerge_MaxInterval_Original(data,col+'_Bin', target, max_interval = len_cut)
    #        data[col+'_Bin']=data[col+'_Bin'].map(lambda x: fc.AssignBin(x,cutOffPointsBin))
    #
    #    else:
    #        cutOffPoints=[]
    #        for i in np.unique(data[col+'_Bin'])[:-1]:
    #            cutOffPoints.append(np.max(data[col][data[col+'_Bin']==i]))
    #        cutPointsDict[col]=cutOffPoints
    #     print(2222,cutPointsDict)
    #
    #    badRate=df(fc.BadRateEncoding(data, col+'_Bin', target)['br_rate'])
    #
    #    badRateDfU=pd.concat([badRateDfU,badRate])
    #
    # endU=time.time()
    # timeU=(endU-startU)/60
    # print('U:',timeU)
    #
    # badRateDfU.index=colsU
    # badRateDfU.to_excel('badRateNumU.xlsx')
    #
    # colsNum=list(cutPointsDict.keys())     ##138


    ###字符
    # maxBinDelStr=[]
    # binDictStr={}
    # for col in colsStr:
    #    print(col)
    #    uniqueStr=len(data[col].unique())
    #    if uniqueStr<=maxBins:
    #        data[col+'_Bin']=data[col]

    #    else:
    #        data[col+'_Encoding']=fc.BadRateEncoding(data, col, target)['encoding']

    #        cutOffPoints=fc.ChiMerge_MaxInterval_Original(data, col+'_Encoding', target, max_interval = maxBins)
    #
    #        data[col+'_Bin']=data[col+'_Encoding'].map(lambda x: fc.AssignBin(x,cutOffPoints))
    #    maxBinPcnt=fc.MaximumBinPcnt(data,col+'_Bin')
    #    print(maxBinPcnt)
    #    groupDict=data[[col,col+'_Bin']].set_index(col).to_dict()[col+'_Bin']
    #    binDictStr[col]=groupDict

    # colsStr=list(binDictStr.keys())   ##22


    ##字符 features to bins
    binDictStr = {}
    badRateStr = df([])
    cols_str_category_sets = {}
    giveup_cols = ['xy.desc_detail']   # features that are not available in online-data
    for col in colsStr:
        if col in giveup_cols:  # drop that feature
            continue

        print(col)
        uniqueStr = len(data[col].unique())

        # if uniqueStr <= maxBins:
        #     data[col + '_Bin'] = data[col]
        # else:
            # data[col + '_Encoding'] = fc.BadRateEncoding(data, col, target)['encoding']
            # cutOffPoints = fc.ChiMerge_MaxInterval_Original(data, col + '_Encoding', target, max_interval=maxBins)
        if uniqueStr > 1:
            categorySet = fc.ChiMerge_MaxInterval_Original_Str(data, col, target, max_interval=maxBins, pvalue=pValue)

            # if len(categorySet) > maxBins:
            #     tmp_pvalue = 0.05  # try to reduce bins
            #     categorySet = fc.ChiMerge_MaxInterval_Original_Str(data, col, target, max_interval=maxBins,
            #                                                        pvalue=tmp_pvalue)
            #
            # if len(categorySet) > maxBins:
            #     tmp_pvalue = 0.01  # try to reduce bins
            #     categorySet = fc.ChiMerge_MaxInterval_Original_Str(data, col, target, max_interval=maxBins,
            #                                                        pvalue=tmp_pvalue)

            print('bin num:', len(categorySet))

            if len(categorySet) == 1:
                continue
            else:
                data[col + '_Bin'] = data[col].map(lambda x: fc.AssignBinStr(x, categorySet))
                cols_str_category_sets[col] = categorySet

                badRate = df(fc.BadRateEncoding(data, col + '_Bin', target)['br_rate'])  # calculate bad rate for each bin

                badRate.index = [col]
                badRateStr = pd.concat([badRateStr, badRate], axis=0, join='outer')  # record bad rates
        # exist0 = fc.Exist0(data, col + '_Bin', target)

        # if len(exist0) == 2:
        #     result = fc.MergeStr0(data, col + '_Bin', target)
        #     groupDict = result['groupDict']
        #     data[col + '_Bin'] = result['colBin']
        #     maxBinPcnt = fc.MaximumBinPcnt(data, col + '_Bin')
        #     print(maxBinPcnt)
        #
        # else:
        #     groupDict = data[[col, col + '_Bin']].set_index(col).to_dict()[col + '_Bin']
                groupDict = data[[col, col + '_Bin']].set_index(col).to_dict()[col + '_Bin']
                print('bin dictionary:', groupDict)
                binDictStr[col] = groupDict

    colsStr = list(binDictStr.keys())  # finish dividing bins for string type features

    # == save bin separation points
    with open('%s\\colsNumericBins.pickle' % result_folder_path, 'wb') as tmp_fo:  # save cols numeric seperation points
        pickle.dump(cols_num_cut_off_points, tmp_fo)

    with open('%s\\colsStringBins.pickle' % result_folder_path, 'wb') as tmp_fo:  # save cols string category sets
        pickle.dump(cols_str_category_sets, tmp_fo)

    # =====  drop columns with low IV
    colsAll = real_col_num + colsStr
    woeDict = {}
    ivDict = {}

    for col in colsAll:
        print(col)
        result = fc.CalcWOE(data, col + '_Bin', target)
        woe = result['WOE'];
        iv = result['IV']
        print(col, iv, 55555)

        woeDict[col] = woe;
        ivDict[col] = iv

    colsSmallIv = [col for col in ivDict.keys() if ivDict[col] < ivMinThre]  ##由于IV过低删除的cols

    colsAll = list(set(colsAll) - set(colsSmallIv))  ##140

    # remove the row in the bad row statistics
    badRateNum = badRateNum.loc[badRateNum.index.isin(colsAll)]
    badRateStr = badRateStr.loc[badRateStr.index.isin(colsAll)]

    # ==== save final feature names
    tmp_num_cols = [x for x in colsAll if x in real_col_num]
    with open('%s\\finalColsNum.pickle' % result_folder_path, 'wb') as tmp_fo:  # save numeric feature names
        pickle.dump(tmp_num_cols, tmp_fo)

    tmp_str_cols = [x for x in colsAll if x in colsStr]
    with open('%s\\finalColsStr.pickle' % result_folder_path, 'wb') as tmp_fo:  # save string feature names
        pickle.dump(tmp_str_cols, tmp_fo)

    #
    # WOE_encoding = []
    # for col in colsAll:
    #     print(col)
    #
    #     if col + '_Bin' in data.columns:
    #         data[col + '_WOE'] = data[col + '_Bin'].map(woeDict[col])
    #         WOE_encoding.append(col + '_WOE')
    #     else:
    #         print("{} cannot be found in trainData".format(col))

    # one hot encoding
    bin_cols = [x + '_Bin' for x in colsAll]
    data_bins = fc.oneHotEncoding(data, target, bin_cols)

    # compare = list(combinations(colsAll, 2))
    #
    # removed_var = []
    # for pair in compare:
    #     print(pair)
    #     (x1, x2) = pair
    #     roh = np.corrcoef([data[x1 + "_WOE"], data[x2 + "_WOE"]])[0, 1]
    #
    #     if abs(roh) >= roh_thresould:  # if correlation is too high between two boxes, keep the one with higher IV
    #         if ivDict[x1] > ivDict[x2]:
    #             removed_var.append(x2)
    #         else:
    #             removed_var.append(x1)
    #
    # colsAll = list(set(colsAll) - set(removed_var))

    # === drop highly correlated bins
    colsAllBins = data_bins.columns.tolist()
    colsAllBins.remove('y')
    all_pairs = list(combinations(colsAllBins, 2))
    removed_bins = []
    for tmp_pair in all_pairs:
        (tmp_bin1, tmp_bin2) = tmp_pair
        roh = np.corrcoef([data_bins[tmp_bin1], data_bins[tmp_bin2]])[0, 1]

        if abs(roh) >= roh_thresould:  # if correlation is too high between two boxes, keep the one with higher IV
            tmp_pattern = '_Bin\.\d+'
            tmp_bin1_col = tmp_bin1[:(re.search(tmp_pattern, tmp_bin1).span()[0])]  # find the corresponding column of this bin
            tmp_bin2_col = tmp_bin2[:(re.search(tmp_pattern, tmp_bin2).span()[0])]  # find the corresponding column of this bin

            if tmp_bin1_col == tmp_bin2_col: # the same feature with highly correlated bins, not drop
                print('same columns:', tmp_bin1_col)
                continue
            if ivDict[tmp_bin1_col] > ivDict[tmp_bin2_col]:
                removed_bins.append(tmp_bin2)
                print('drop bin:', tmp_bin2)

                tmp_bin2_no = float(tmp_bin2[ (len(tmp_bin2_col)+ 5) :])  # 5: '_Bin.'
                if tmp_bin2_col in badRateNum.index:
                    badRateNum.loc[tmp_bin2_col, tmp_bin2_no] = np.nan  # erase the bin record in bad rate statistics
                else:
                    badRateStr.loc[tmp_bin2_col, tmp_bin2_no] = np.nan  # erase the bin record in bad rate statistics
            else:
                removed_bins.append(tmp_bin1)
                print('drop bin', tmp_bin1)

                tmp_bin1_no = float(tmp_bin1[ (len(tmp_bin1_col)+ 5) :])  # 5: '_Bin.'
                if tmp_bin1_col in badRateNum.index:
                    badRateNum.loc[tmp_bin1_col, tmp_bin1_no] = np.nan  # erase the bin record in bad rate statistics
                else:
                    badRateStr.loc[tmp_bin1_col, tmp_bin1_no] = np.nan  # erase the bin record in bad rate statistics


    colsAllBins = list(set(colsAllBins) - set(removed_bins))

    # record final bins' bad rates
    badRateNum.to_excel('%s\\badRateNumFinal.xlsx' % result_folder_path)
    badRateStr.to_excel('%s\\badRateStrFinal.xlsx' % result_folder_path)

    # === logistic regression
    # var_WOE_list = [i + '_WOE' for i in colsAll]
    # y = data[target]
    # X = data[var_WOE_list]
    # X['intercept'] = [1] * X.shape[0]
    # LR = sm.Logit(y, X).fit()
    # summary = LR.summary()

    y = data_bins[target]
    X = data_bins[colsAllBins]
    # X.loc[:, 'intercept'] = 1
    LR = LogisticRegression(fit_intercept=True)
    LR.fit(X, y)

    with open('%s\\LR.pickle' % result_folder_path, 'wb') as tmp_fo:  # save model
        pickle.dump(LR, tmp_fo)

    with open('%s\\finalBins.pickle' % result_folder_path, 'wb') as tmp_fo: # save final bins used for regression
        pickle.dump(colsAllBins, tmp_fo)

    # in-sample evaluation
    y_prd = LR.predict(X)
    y_test = y.values

    model_precision = precision_score(y_test, y_prd, pos_label=1, average='binary')
    model_recall = recall_score(y_test, y_prd, pos_label=1, average='binary')
    fpr, tpr, tmp_thrs = roc_curve(y_test, y_prd, pos_label=1)
    ks = np.max(np.abs(tpr - fpr))
    aucscore = auc(fpr, tpr)
    confs_matrix = confusion_matrix(y_test, y_prd, labels=[0, 1])
    confs_matrix = df(confs_matrix, columns=['predict_good', 'predict_bad'], index=['actual_good', 'actual_bad'])

    print('in-sampe statistics:')
    print('precision:', model_precision)
    print('recall:', model_recall)
    print('auc:', aucscore)
    print('KS:', ks)
    print(confs_matrix)

    # LR = sm.Logit(y, X).fit()
    # summary = LR.summary()
    # print(summary)

def testing(data, colsTarget, result_folder_path):
    # == convert data into bins
    with open('%s\\finalColsNum.pickle' % result_folder_path, 'rb') as tmp_fo:  # load numeric feature names
        cols_num = pickle.load(tmp_fo)

    with open('%s\\finalColsStr.pickle' % result_folder_path, 'rb') as tmp_fo:  # load string feature names
        cols_str = pickle.load(tmp_fo)

    with open('%s\\colsNumericBins.pickle' % result_folder_path, 'rb') as tmp_fo:  # load numeric bins
        cols_num_cut_points = pickle.load(tmp_fo)

    with open('%s\\colsStringBins.pickle' % result_folder_path, 'rb') as tmp_fo:  # load string bins
        cols_str_category_sets = pickle.load(tmp_fo)

    bin_cols = []
    for tmp_col in cols_num:
        data.loc[:, tmp_col + '_Bin'] = data[tmp_col].map(lambda x: fc.AssignBinNum(x, cols_num_cut_points[tmp_col]))
        bin_cols.append(tmp_col + '_Bin')
    for tmp_col in cols_str:
        print(tmp_col)
        data.loc[data[tmp_col].isnull(), tmp_col] = 'null'
        data.loc[:, tmp_col] = data[tmp_col].replace({'"null"': 'null', '': 'null', 'nan': 'null'})
        data.loc[:, tmp_col + '_Bin'] = data[tmp_col].map(lambda x: fc.AssignBinStr(x, cols_str_category_sets[tmp_col]))
        bin_cols.append(tmp_col + '_Bin')

    # === one hot encoding
    target = colsTarget[0]
    data_bins = fc.oneHotEncoding(data, target, bin_cols)

    # ==== predict
    with open('%s\\LR.pickle' % result_folder_path, 'rb') as tmp_fo:  # load model
        LR = pickle.load(tmp_fo)

    with open('%s\\finalBins.pickle' % result_folder_path, 'rb') as tmp_fo:  # load final bin name
        final_bins = pickle.load(tmp_fo)

    # maybe bins missed in testing data
    miss_bins = [x for x in final_bins if x not in data_bins.columns.tolist()]
    for tmp_bin_col in miss_bins:
        data_bins.loc[:, tmp_bin_col] = 0

    X_test = data_bins[final_bins]
    y_test = data_bins[target]

    y_prd = LR.predict(X_test)
    # y_prd = y_prd.apply(lambda x: 1 if x > 0.5 else 0)
    # y_prd = y_prd.values
    y_test = y_test.values

    model_precision = precision_score(y_test, y_prd, pos_label=1, average='binary')
    model_recall = recall_score(y_test, y_prd, pos_label=1, average='binary')
    fpr,tpr,tmp_thrs = roc_curve(y_test,y_prd,pos_label = 1)
    ks = np.max(np.abs(tpr-fpr))
    aucscore = auc(fpr,tpr)
    confs_matrix = confusion_matrix(y_test,y_prd,labels=[0,1])
    confs_matrix = df(confs_matrix, columns=['predict_good', 'predict_bad'], index=['actual_good', 'actual_bad'])

    print('out-of-sampe statistics:')
    print('precision:', model_precision)
    print('recall:', model_recall)
    print('auc:', aucscore)
    print('KS:', ks)
    print(confs_matrix)

    return {'precision': model_precision, 'recall': model_recall, 'auc':aucscore, 'KS':ks, 'confusion_matrix': confs_matrix}



if __name__ == '__main__':
    result_folder_path = r'D:\Loan\PuhuiModel\training'

    # === load data
    data_folder_path = r'D:\Loan\PuhuiModel\data'
    data, colsNum, colsStr, colsTarget = loadData(data_folder_path)

    # data_train, data_test = train_test_split(data, test_size=0.3, train_size=0.7, random_state=int(time.clock()))
    #
    # training(data_train, colsNum, colsStr, colsTarget, result_folder_path)
    #
    # testing(data_test, colsTarget, result_folder_path)

    training(data, colsNum, colsStr, colsTarget, result_folder_path)

    testing(data, colsTarget, result_folder_path)


##
# binDictStr={}
# for col in colsStr:
#    print(col)
#    uniqueStr=len(data[col].unique())
#    data[col+'_Encoding']=fc.BadRateEncoding(data, col, target)['encoding']
#    data[col+'_Bin']=data[col+'_Encoding'].map(lambda x: fc.AssignBin(x,cutOffPoints))
#
#    exist0=fc.Exist0(data,col+'_Bin',target)
#
#    if len(exist0)==2:
#        result=fc.MergeStr0(data,col+'_Bin',target)
#        groupDict=result['groupDict']
#        data[col+'_Bin']=result['colBin']
#        print(groupDict)
#
#    else:
#        groupDict=data[[col,col+'_Bin']].set_index(col).to_dict()[col+'_Bin']
#    print(2333222,groupDict)
#
#    binDictStr[col]=groupDict
# colsStr=list(binDictStr.keys())   ##22


# startM=time.time()
# maxBinDelNum=[]
# cutPointsDict={}
#
#
# for col in colsNum:
#
#    print(col)
#    cutOffPoints=fc.ChiMerge_MaxInterval_Original(data, col, target, max_interval = maxBins)
#    data[col+'_Bin']=data[col].map(lambda x: fc.AssignBin(x,cutOffPoints))
#
#    exist0=fc.Exist0(data,col+'_Bin',target)
#
#    if len(exist0)==2:
#        result=fc.MergeNum0(data,col+'_Bin',target)
#        data[col+'_Bin']=result['colBin']
#        cutOffPoints=result['cutOffPoints']
#    else:
#        pass
#
#    maxBinPcnt=fc.MaximumBinPcnt(data,col+'_Bin')
#    print(maxBinPcnt)
#
#
#    badRate=df(fc.BadRateEncoding(data, col+'_Bin', target)['br_rate'])
#    badRateDfM=pd.concat([badRateDfM,badRate])
#
# badRateDfM.index=colsNum
# badRateDfM.to_excel('badRateNumM.xlsx')
# endM=time.time()
# timeM=(endM-startM)/60
# print('M:',timeM)
#
# badRateDfM.index=colsNum
# badRateDfM.to_excel('badRateNumM_0627.xlsx')


# con=connect(host='10.27.93.5',port=21050,database='fdm')
# cur=con.cursor()
# cur.execute('select * from bdm.bdm_cs_sbiao_persist \
# left join fdm.fdm_nmd_zq on bdm.bdm_cs_sbiao_persist.appno=fdm.fdm_nmd_zq.reqinfo_applyno \
# where bdm_cs_sbiao_persist.dt>='+'"'+str(startTime)+'"'+' and bdm_cs_sbiao_persist.dt<'+'"'+str(endTime)+'"')
# zq_info=df(cur.fetchall())
# cur.close()
#
# zq_info.columns=pd.read_csv(path+'\\zq_info.csv').columns
# zq_info1=zq_info.copy();zq_info1=zq_info1.drop_duplicates('appno')
#
# con=connect(host='10.27.93.5',port=21050,database='fdm')
# cur=con.cursor()
# cur.execute('select * from fdm.fdm_nmd_zq ')
# a=df(cur.fetchall())
# cur.close()
#
# a.columns=pd.read_csv(path+'\\a.csv').columns
# a1=a.copy();a1=a1.drop_duplicates('reqinfo_applyno')

# ['biz__province',
# 'jxl__region_call_out_cnt_region',
# 'ylzc__CDTB221',
# 'ylzc__CSSS002',
# 'jxl__region_call_in_cnt_pct_region',
# 'biz__apply_hour',
# 'jxl__region_call_out_cnt_pct_region',
# 'jxl__region_call_in_cnt_region',
# 'jxl__region_loc',
# 'jxl__region_avg_call_in_time_region'

###测试样本占比
# y_predict_0=Se(LR_fit.predict_proba(X_test)[:,0])
# score_cut=np.arange(0,1,step=0.1)
# score_bin=df(y_predict_0.map(lambda x: fc.AssignBin(x,score_cut)))
# score_bin['count']=[1]*1000
# score_bin_group=score_bin.groupby([0]).sum()
# score_bin_group['percent']=score_bin_group['count']/score_bin_group['count'].sum()

#   #####################将箱子号转化成分数
#    for col in colsAll:
#        if type(dataSe[col][0])!=str:
#            if 'null' in ScoreDict[col].keys():
#                ScoreDict[col].pop('null')
#            if 'max' in ScoreDict[col].keys():
#                ScoreDict[col].pop('max')
#        dataSe[col].replace(ScoreDict[col],inplace=True)
