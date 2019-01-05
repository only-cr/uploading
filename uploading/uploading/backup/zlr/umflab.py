# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from hyperopt import hp, tpe, fmin
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
global train_data, val_data

#laod the data and make them merged
data = pd.read_csv(r"D:\umpvisitor120\bsbank\umf\data\20181227\data\original_data.csv")

feature = pd.read_excel(r"D:\umpvisitor120\bsbank\umf\data\20181227\data\for_model.xlsx")
feature = feature[0].tolist()
feature.append('y')
data = data[feature]

data.replace({-98:np.nan, -99:np.nan, -100: np.nan}, inplace=True)
data.dropna(axis=1, thresh=np.floor(0.3 * len(data)), inplace=True)
str_list = []
float_list = []
for i in data.columns:
   try:
       data[i] = data[i].astype(np.float)
       data[i] = data[i].fillna(data[i].min() * 1.5)
       float_list.append(i)
       
   except:
       data[i] = data[i].astype(np.str)
       tmp = data[i].copy()
       tmp = tmp.astype(np.str)
       tmp = [x.replace('.0','') for x in tmp]
       data[i] = pd.Series(tmp)
       str_list.append(i)
       del tmp
del i
#
#data[float_list] = data[float_list].fillna(data[float_list].min() * 1.5)
#data[str_list] = data[str_list].fillna('unknown')




data_dummy = pd.get_dummies(data)
datay = data_dummy.y.copy()
datax = data_dummy.drop(['y'], axis=1).copy()
tmp_label = datax.columns
datax = preprocessing.scale(datax)
datax = pd.DataFrame(datax, columns=tmp_label)

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

feature_importance = xgb1.get_fscore()
feature_importance = pd.DataFrame(feature_importance, index=[0])
feature_sorted = pd.DataFrame(columns=['feature', 'importance'])
feature_sorted['feature'] = feature_importance.columns.tolist()
feature_sorted['importance'] = feature_importance.loc[0].tolist()
feature_sorted.sort_values(by='importance', axis=0, ascending=False, inplace=True)
feature_sorted.reset_index(inplace=True,drop=True)

#useful_columns = feature_sorted.loc[0:800, 'feature']
##del feature_importance, feature_sorted
#
#def result_print(x, y, clf, type, thre=0.5):
#    # y_predict = clf.predict(x)
#    y_score_tmp = clf.predict_proba(x)
#    y_score = [i[1] for i in y_score_tmp]
#    y_predict = [1 if i > thre else 0 for i in y_score]
#    accuracy = accuracy_score(y, y_predict)
#    fpr, tpr, threshods = roc_curve(y, y_score, pos_label=1)
#    ks = np.max(np.abs(tpr - fpr))
#    report = classification_report(y, y_predict)
#    print('result of %s data: '%type)
#    print('report: ')
#    print(report)
#    print('accuracy: ')
#    print(accuracy)
#    print('K-S: ')
#    print(ks)
#    return ks
##
##x_train = x_train[useful_columns]
##x_test = x_test[useful_columns]
##
#clf = LDA()
## clf.fit(x_train, y_train)
#rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(4), scoring='accuracy')
#rfecv.fit(x_train, y_train)
#x_train = x_train.loc[:, rfecv.get_support().tolist()]
#x_test = x_test.loc[:, rfecv.get_support().tolist()]
##x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.33, random_state=88)
#clf = LDA()
#clf.fit(x_train, y_train)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
#
#result_print(x_train, y_train, clf, 'train', 0.5)
#result_print(x_test, y_test, clf, 'test', 0.5)

#
print("今年下半年，中美合拍的西游记即将正式开机，我继续扮演美猴王孙悟空，我会用美猴王艺术形象努力创造一个正能量的形象，文体两开花，弘扬中华文化，希望大家能多多关注。")

