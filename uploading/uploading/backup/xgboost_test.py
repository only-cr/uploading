#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:31:11 2018

@author: lipchiz
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.decomposition import PCA
from hyperopt import hp, tpe, fmin
import xgboost as xgb
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
global train_data, val_data

data = pd.read_excel(r"E:\umf\model2_data.xlsx")
print('the data has been loaded.')
'''------------------------------------------------------------------------------------------------------------------'''
city_consistence = []
for i in zip(data['CDTB150'], data['cell_loc']):
    if i[0]==i[1]:
        city_consistence.append(1)
    else:
        city_consistence.append(0)

city_consistence = pd.Series(city_consistence)
data['city_consistence'] = city_consistence
data.drop(['lbsInfo_city', 'CDTB150', 'CDTB151', 'cell_loc', 'query_id', 'lbsInfo_province'], axis=1, inplace=True)

data.dropna(axis=1, thresh=int(data.shape[0]*0.6), inplace=True)
data.dropna(axis=0, thresh=int(data.shape[1]*0.6), inplace=True)

str_list = []
float_list = []

for i in data.columns:
   try:
       data[i].astype(np.float)
       float_list.append(i)
   except:
       str_list.append(i)
del i, city_consistence

data[float_list] = data[float_list].fillna(data[float_list].min() * 1.5)
data[str_list] = data[str_list].fillna('unknown')
data_dummy = pd.get_dummies(data)
datay = data_dummy.y.copy()
datax = data_dummy.drop(['y'], axis=1).copy()
# datax_norm = `(datax)
# datax_norm = pd.DataFrame(datax_norm, columns=datax.columns)
del data, float_list, str_list
x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size=0.33, random_state=88)

train_data = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns)
val_data = xgb.DMatrix(x_test, y_test, feature_names=x_test.columns)



def objective(args):
#    global train_data,val_datas
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 2,
        'silent':1,
        'learning_rate': args['learning_rate'],
        'colsample_bytree': args['colsample_bytree'],
        'max_depth': args['max_depth'] + 4,
        'subsample': args['subsample']
    }
    xgb1 = xgb.train(params,args['train_data'],evals_result = {'eval_metric':'auc'},
            num_boost_round=100000, evals=[(args['train_data'],'train'),(args['val_data'],'val')],
            verbose_eval=5 ,early_stopping_rounds=20)
    y_score = xgb1.predict(val_data)
    fpr,tpr,threshods = roc_curve(val_data.get_label(),y_score,pos_label = 1)
    aucscore = auc(fpr,tpr)
    print(aucscore)
    return -aucscore


params_space = {
    'learning_rate': hp.uniform("learning_rate", 0.05, 0.15),
    'max_depth': hp.randint('max_depth',10),
    'subsample': hp.uniform("subsample", 0.5, 0.9),
    'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    'train_data':train_data,
    'val_data':val_data,
    
}

best_sln = fmin(objective, space=params_space, algo=tpe.suggest, max_evals=20)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 2,
    'silent':1,
    'learning_rate': best_sln['learning_rate'],
    'colsample_bytree': best_sln['colsample_bytree'],
    'max_depth': best_sln['max_depth'] + 4,
    'subsample': best_sln['subsample']
}

xgb1 = xgb.train(params,train_data, evals_result = {'eval_metric':'auc'},
            num_boost_round=100000, evals=[(train_data, 'train'), (val_data, 'val')],
            verbose_eval=5, early_stopping_rounds=20)

y_score = xgb1.predict(val_data)
y_predict = np.int64(y_score>0.5)
accuracyscore = accuracy_score(val_data.get_label(), y_predict)
fpr,tpr,threshods = roc_curve(val_data.get_label(), y_score, pos_label=1)
ks_test = np.max(np.abs(tpr-fpr))
print(ks_test)

y_score_train = xgb1.predict(train_data)
y_predict_train = np.int64(y_score_train>0.5)
accuracy_train = accuracy_score(train_data.get_label(), y_predict_train)
fpr_train, tpr_train, threshods_train = roc_curve(train_data.get_label(), y_score_train, pos_label=1)
ks_train = np.max(np.abs(tpr_train-fpr_train))
print(ks_train)

del y_score, y_score_train, accuracyscore, accuracy_train, fpr, fpr_train, tpr, tpr_train, threshods, threshods_train,
ks_test, ks_train, x_train, x_test, y_train, y_test, xgb1, train_data, val_data


feature_importance = xgb1.get_fscore()
feature_importance = pd.DataFrame(feature_importance, index=[0])
feature_sorted = pd.DataFrame(columns=['feature', 'importance'])
feature_sorted['feature'] = feature_importance.columns.tolist()
feature_sorted['importance'] = feature_importance.loc[0].tolist()
feature_sorted.sort_values(by='importance', axis=0, ascending=False, inplace=True)
feature_sorted.reset_index(inplace=True,drop=True)


useful_columns = feature_sorted.loc[0:200, 'feature']
del feature_importance, feature_sorted

datax = data_dummy[useful_columns]
x_train, x_test, y_train, y_test = train_test_split(datax, data_dummy.y, test_size=0.33, random_state=88 )

x_train_norm =  preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
pca = PCA(10, svd_solver='auto', random_state=88)
pca.fit(x_train_norm)
variance = pca.explained_variance_
var_ratio = pca.explained_variance_ratio_

print('the explained variance ratio is: ')
print(var_ratio.sum()*100)

x_train_pca = pca.transform(x_train_norm)
x_test_pca = pca.transform(x_test_norm)
lg = LogisticRegression(random_state=88, verbose=1, solver='lbfgs', n_jobs=3)
lg.fit(x_train_pca, y_train)

print('the result of logistic regression: ')

y_predict_train = lg.predict(x_train_pca)
y_train_score_tmp = lg.predict_proba(x_train_pca)
y_train_score = [x[1] for x in y_train_score_tmp]
accuracy_train = accuracy_score(y_train, y_predict_train)
fpr_train, tpr_train, threshods_train = roc_curve(y_train, y_train_score, pos_label=1)
ks_train = np.max(np.abs(tpr_train-fpr_train))
report_train = classification_report(y_train, y_predict_train)
print('1. result of train data: ')
print('report: ')
print(report_train)
print('accuracy: ')
print(accuracy_train)
print('K-S: ')
print(ks_train)






print('今年下半年，中美合拍的西游记即将正式开机，我继续扮演美猴王孙悟空，我会用美猴王艺术形象努力创造一个正能量的形象，文体两开花，弘扬中华文化，希望大家能多多关注。')