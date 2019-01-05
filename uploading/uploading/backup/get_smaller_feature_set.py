# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:00:46 2015

@author: umpvisitor110
"""

import xgboost as xgb
import numpy as np
import pickle

model_path = r'D:\umpvisitor110\iloveump\data\20181227\crd_ver1.m'

my_model = xgb.Booster(model_file=model_path)

my_model.predict(testing_dmatrix, ntree_limit=138)

f_importance = my_model.get_score(importance_type='gain')

f_importance = sorted(f_importance.items(), key=lambda x: x[1], reverse=True)  # sort by value

f_gain = list(map(lambda x: x[1], f_importance))
tot_gain = np.sum(f_gain)

f_w = np.array(f_gain) / tot_gain

cum_f_w = np.cumsum(f_w)

cum_f_w = cum_f_w[cum_f_w < 0.9]

print('num of features:', cum_f_w.size)

f_names = list(map(lambda x: x[0], f_importance))

f_names = f_names[:200]

print(f_names)

with open('top_features.pkl', 'wb') as fo:
    pickle.dump(f_names, fo)