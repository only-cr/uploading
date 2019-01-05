# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:15:01 2015

@author: umpvisitor110
"""

import pickle
import pandas as pd

with open('subset_most_important_feature.pkl', 'rb') as fi:
    top_feature_names = pickle.load(fi)
    
feature_name_prefix = ['G517', 'G402', 'G403', 'G438']

feature_realname_files = list(map(lambda x: x + '.csv', feature_name_prefix))

for tmp_prefix, tmp_path in zip(feature_name_prefix, feature_realname_files):
    all_f_name = pd.read_csv(tmp_path, encoding='gbk')
    
    name_this_file = [x for x in top_feature_names if x.find(tmp_prefix) == 0]  # find feature name belong to this file
    
    print(tmp_prefix,':', len(name_this_file))
    
    if tmp_prefix == 'G517':
        tmp_index = [int(x[5:]) - 1 for x in name_this_file]
        tmp_df = pd.DataFrame({'input': tmp_index, 'output': name_this_file})
        tmp_df.to_csv(tmp_prefix + '_final.csv', index=False)
#        with open(tmp_prefix + '_final.txt', 'w') as fo:
#            for num in tmp_index:
#                fo.write('%d\n' % num)
    
    if tmp_prefix == 'G402':
        tmp_index = [int(x[5:]) + 1 for x in name_this_file]
        selected_f_name = all_f_name.loc[all_f_name[u'变量编号'].isin(tmp_index)]
        selected_f_name.to_csv(tmp_prefix + '_final.csv', index=False, encoding='gbk')
        
    if tmp_prefix == 'G403':
        tmp_index = [int(x[5:]) + 1 for x in name_this_file]
        tmp_index = list(map(lambda x: '第%d列' % x, tmp_index))
        selected_f_name = all_f_name.loc[all_f_name[u'标签数据文件位置\n（第一列0为序号+第二列1为月份）'].isin(tmp_index)]
        selected_f_name.to_csv(tmp_prefix + '_final.csv', index=False, encoding='gbk')
    
    if tmp_prefix == 'G438':
        tmp_index = [int(x[5:]) + 1 for x in name_this_file]
        selected_f_name = all_f_name.loc[all_f_name[u'变量编号'].isin(tmp_index)]
        selected_f_name.to_csv(tmp_prefix + '_final.csv', index=False, encoding='gbk')