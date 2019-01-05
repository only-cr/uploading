import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, roc_curve, auc
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
import json
import numpy as np
import time
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

class myClassifier:
    def __init__(self):
        self.all_feature_filepaths = [r'D:\umpvisitor110\iloveump\data\20181227\mxf20181219_lab0120343_Gup1006517_lab0120343_match.txt', 
                                          r'D:\umpvisitor110\iloveump\data\20181227\mxf20181219_lab0120344_Gup1006402_lab0120344_match.txt',
                                          r'D:\umpvisitor110\iloveump\data\20181227\mxf20181219_lab0120344_Gup1006403_lab0120344_match.txt',
                                          r'D:\umpvisitor110\iloveump\data\20181227\mxf20181219_lab0120344_Gup1006438_lab0120344_match.txt']
        self.feature_name_prefix = ['G517', 'G402', 'G403', 'G438']
        self.label_filepath = r'D:\umpvisitor110\iloveump\data\mxf20181219(tag).csv'
        self.subset_filepath = r'D:\umpvisitor110\iloveump\trunk\new_sample.txt'
        self.training_features = []
        self.training_labels = []
        self.testing_features = []
        self.testing_labels = []
        self.testing_index = []
        self.trees_details = ''
        self.feature_bins = {}
        self.feature_nan_score = {}
        self.feature_bin_scores = {}
        self.current_training_features = []
        self.current_training_labels = []
        self.validation_features = []
        self.validation_labels = []
        self.k_fold = []
        self.cv_num = 3
        self.nan_pro_theshold = 0.7
        self.opt_counter = 0

    def load_data(self):
        '''
        # load data from file
        tmp_features = []
        tmp_features.append(pd.read_csv(self.all_feature_filepaths[0], index_col=None, header=None))
        for i, tmp_path in enumerate(self.all_feature_filepaths):
            if i == 0:
                continue
            tmp_features.append(pd.read_table(tmp_path, sep='\t', index_col=None, header=None))
        
        # process data
        tmp_features[0] = tmp_features[0].replace([-100, -99, -98], np.nan) # replace values back to nan
        tmp_datetime = tmp_features[0][1].astype('str').apply(lambda x: datetime.strptime(x, '%Y%m%d'))
        execution_date = tmp_features[0][[0]].join(tmp_datetime)
        execution_date = execution_date.rename({1: 'exe_date'}, axis=1)  # loan release dates
        
        tmp_features[1] = self.feature_processing(tmp_features[1])  # only take the nearest data
        for tmp_col in [157, 158, 159, 160, 164, 165]:  # column of dates
            tmp_datetime = tmp_features[1][tmp_col].apply(lambda x: datetime.strptime(str(int(x)), '%Y%m') if not np.isnan(x) else x)
            tmp_date = tmp_features[1][[0]].join(tmp_datetime)
            tmp_date = tmp_date.merge(execution_date, on=0, how='left')
            
            tmp_features[1].loc[:, tmp_col] = (tmp_date['exe_date'] - tmp_date[tmp_col]).apply(lambda x: x.days if x ==x else np.nan)  # get diff day num
            #tmp_features[1] = tmp_features[1].drop(tmp_col, axis=1)
        
        tmp_features[2] = self.feature_processing(tmp_features[2])
        tmp_features[2] = tmp_features[2].replace({'unknown': np.nan})
        tmp_features[2].loc[:, 4] = tmp_features[2][4] + tmp_features[2][5] # combine province and city
        tmp_features[2] = tmp_features[2].drop(5, axis=1)
        
        tmp_features[3] = self.feature_processing(tmp_features[3])  # only take the nearest data
        
        for i, tmp_single_feature in enumerate(tmp_features):  # combine features from all files into one
            new_col_names = list(map(lambda x: '%s_%d' % (self.feature_name_prefix[i], x), tmp_features[i].columns.tolist()))
            tmp_features[i].columns = new_col_names  # rename columns
            tmp_features[i] = tmp_features[i].rename({'%s_0' % self.feature_name_prefix[i]: 'index'}, axis=1)
            tmp_features[i] = tmp_features[i].drop('%s_1' % self.feature_name_prefix[i], axis=1) # drop execution date/month
            
            if i == 0:
                raw_features = tmp_features[i].copy()
            else:
                raw_features = raw_features.merge(tmp_features[i], on='index', how='outer')
        
        raw_features.to_csv('final_features.csv', index=False)
        '''       
        
        raw_features = pd.read_csv('final_features.csv', index_col=None)
        
        # combine labels
        raw_label = pd.read_csv(self.label_filepath, index_col=None, header=0)
        
        raw_data = raw_features.merge(raw_label[['index', 'tag']], on='index', how='inner') # combine features and labels
        
#        # extract sub dataset
#        tmp_1_raw_data = raw_data.loc[raw_data['tag'] == 1]
##        tmp_0_raw_data = raw_data.loc[raw_data['tag'] == 0]
#        with open(self.subset_filepath, 'r') as fi: # read subset index
#            subset_index = []
#            for tmp_l in fi.readlines():
#                subset_index.append(int(tmp_l.strip()))
##        raw_data = raw_data.loc[raw_data['index'].isin(subset_index)]
#        tmp_1_raw_data = tmp_1_raw_data.loc[tmp_1_raw_data['index'].isin(subset_index)]
##        tmp_0_raw_data = tmp_0_raw_data.loc[tmp_0_raw_data['index'].isin(subset_index)]
#        raw_data = tmp_1_raw_data.copy()
        
        # seperate feature and label
        raw_labels = raw_data['tag']
        self.testing_index = raw_data['index']
        raw_data = raw_data.drop(['index', 'tag'], axis=1)
        
        # separate numeric and categorical features
        numeric_feature_names = []
        categorical_feature_names = []
        total_feature_names = raw_data.columns.tolist()
        for tmp_col_nm in total_feature_names:
            try:
                raw_data.loc[:, tmp_col_nm] = raw_data[tmp_col_nm].astype('float')
                numeric_feature_names.append(tmp_col_nm)
            except ValueError:
                categorical_feature_names.append(tmp_col_nm)
        
        # seperate numeric and categorical
        raw_categorical_data = raw_data[categorical_feature_names]
        raw_data = raw_data.drop(categorical_feature_names, axis=1)

        # convert categorical data to dummy variables
        dummy_variables = self.one_hot_encoding(raw_categorical_data)
        raw_data = raw_data.join(dummy_variables)
        
        with open('subset_most_important_feature.pkl', 'rb') as fi:
            top_feature_names = pickle.load(fi)
        
        self.testing_features = raw_data[top_feature_names]
        self.testing_labels = raw_labels
        
    def data_distribution(self, model, testing_features, testing_labels, tree_num):
        testing_dmatrix = xgb.DMatrix(testing_features, label=testing_labels, feature_names=testing_features.columns)
        pro_pred = model.predict(testing_dmatrix, ntree_limit=tree_num).tolist()
        
        scores = list(map(lambda x: self.reverse_sigmoid(x), pro_pred))
        
        mean_scores = np.mean(scores)
        std_scores = np.std(scores)
        normalized_scores = list(map(lambda x: (x - mean_scores) / std_scores * 1000, scores))
        
        normalized_scores = list(map(lambda x: (x + 3000) / 6, normalized_scores))
        
        return normalized_scores
#        return scores
    
    def reverse_sigmoid(self, y):
        return np.log(y) - np.log(1- y)
    
    def feature_processing(self, data):
        tmp_new_features = data.groupby(0)[1].max()
        tmp_new_features = tmp_new_features.reset_index()
        tmp_new_features = tmp_new_features.merge(data, on=[0, 1], how='left')
        
        return tmp_new_features
    
    def one_hot_encoding(self, categorical_data):
        dummy_variables = pd.DataFrame([], index=categorical_data.index)
        for col in categorical_data.columns:
            uni_vals = categorical_data.loc[~categorical_data[col].isnull(), col].unique()

            for tmp_val in uni_vals:
                tmp_new_col = col + '_' + tmp_val
                dummy_variables.loc[:, tmp_new_col] = np.nan
                dummy_variables.loc[categorical_data[col] == tmp_val, tmp_new_col] = 1

        return dummy_variables

    def parse_trees(self, trees_details):
        for tmp_tree in trees_details:
            tmp_tree_json = json.loads(tmp_tree) # split -> feature, split_condition -> split point, yes -> left child, no -> right child, missing -> child that missing data go, children -> list containing json

            self.find_leaf(tmp_tree_json, [-np.inf, np.inf], tmp_tree_json['split'], True)

        # add bin scores together
        for tmp_feature, tmp_bin_scores in self.feature_bins:
            tmp_ori_bins = [x[0] for x in tmp_bin_scores]
            tmp_ori_scores = [x[1] for x in tmp_bin_scores]
            tmp_split_points = set()
            for tmp_single_bin in tmp_ori_bins:
                tmp_split_points = tmp_split_points | set(tmp_single_bin) # find unique split points
            tmp_split_points = list(tmp_split_points)
            tmp_split_points.sort(reverse=False)

            tmp_real_bins = []
            for i in range(len(tmp_split_points) - 1):
                tmp_real_bins.append([tmp_split_points[i], tmp_split_points[i+1]]) # unique bins

            tmp_real_scores = np.zeros(len(tmp_real_bins))
            for tmp_single_bin, tmp_single_score in zip(tmp_ori_bins, tmp_ori_scores): # convert scores on fake bins to real bins
                if tmp_single_bin[0] + 1 != tmp_single_bin[0]: # left limit is not -inf:  >=
                    tmp_index = tmp_split_points.index(tmp_single_bin[0]) # find covered real bins
                    tmp_real_scores[tmp_index:] += tmp_single_score

                elif tmp_single_bin[1] + 1 != tmp_single_bin[1]: # right limit is not inf: <=
                    tmp_index = tmp_split_points.index(tmp_single_bin[1]) 
                    tmp_real_scores[:tmp_index] += tmp_single_score
            
            self.feature_bin_scores[tmp_feature] = [tmp_real_bins, tmp_real_scores]
            
        with open('feature_bin_score.pkl', 'wb') as fo:
            pickle.dump(self.feature_bin_scores, fo)
    
    def get_score_n_probabiliy(self, features):
        final_scores = pd.DataFrame([], index=features.index)
        for tmp_col in features.columns:
            tmp_bins = list(map(lambda x: x[0], self.feature_bin_scores[tmp_col]))
            tmp_scores = list(map(lambda x: x[1], self.feature_bin_scores[tmp_col]))
            
            def tmp_func(x, bins):
                for i, b in enumerate(bins):
                    if (b[0] <= x) and (x < b[1]):
                        return i
                print('cannot find bin for %f' % x)
                raise ValueError
                return np.nan
            
            tmp_index = features[tmp_col].apply(lambda x: tmp_func(x, tmp_bins))
            final_scores.loc[:, tmp_col] = tmp_index.apply(lambda x: tmp_scores[x])
            


    def find_leaf(self, node, current_interval, feature_name, is_nan_included):
        if 'leaf' in node.keys():
            if feature_name in self.feature_bins.keys():
                self.feature_bins[feature_name].append([current_interval, node['leaf']])  # record interval & score
            else:  # new feature
                self.feature_bins[feature_name] = [[current_interval, node['leaf']]]

            if is_nan_included: # record score for nan
                if feature_name in self.feature_nan_score:
                    self.feature_nan_score[feature_name] += node['leaf']
                else:
                    self.feature_nan_score[feature_name] = node['leaf']
        else: # not a leaf, recursive
            if node['split'] != feature_name:
                print('feature name inconsistent!')  # make sure only one feature in a tree
                raise ValueError

            if node['missing'] == node['yes']:
                left_child_include_nan = True
                right_child_include_nan = False
            elif node['missing'] == node['no']:
                left_child_include_nan = False
                right_child_include_nan = True

            left_child = node['children'][0]
            left_interval = current_interval.copy()
            left_interval[1] = node['split_condition'] # update right limit for left child
            self.find_leaf(left_child, left_interval, node['split'], left_child_include_nan)

            right_child = node['children'][1]
            right_interval = current_interval.copy()
            right_interval[0] = node['split_condition'] # update left limit for right child
            self.find_leaf(right_child, right_interval, node['split'], right_child_include_nan)


if __name__ == '__main__':
    # testing_params = {
    #     'booster': 'dart',
    #     'nthread': 3,
    #     'learning_rate': 0.1,
    #     'min_child_weight': 0.1,
    #     'max_depth': 1,
    #     'gamma': 0.1,  # min loss reduce
    #     'subsample': 0.9,  # select samples
    #     'colsample_bytree': 0.9,  # select features
    #     'max_delta_step': 1,
    # # non-zero when sample imbalance, (when too few samples in a node, G is close to zero, weight would be very large)
    #     'reg_lambda': 1,  # L2 regularization
    #     'reg_alpha': 0.5,  # L1 regularization
    #     'scale_pos_weight': 1,  # weight for sample imbalance
    #     'objective': 'binary:logistic',
    #     # 'num_class':3, # multi-class classification
    #     'seed': 6,  # random state
    #     'eval_metric': 'auc',
    #
    #     # dart params
    #     'sample_type': 'uniform',
    #     'normalize_type': 'tree',
    #     'rate_drop': 0.3,
    # }
    t_start = time.clock()
    
    mc = myClassifier()
    mc.load_data()
    
    model = xgb.Booster(model_file='crd_ss_sd_ver1.m')
    
    with open('bst_param_set_ss.pkl', 'rb') as fi:
        bst_param_set = pickle.load(fi)
    
    scores = mc.data_distribution(model, mc.testing_features, mc.testing_labels, bst_param_set['best_boost_num'])
    
    plt.hist(scores, bins=50)
    
    out_f = mc.testing_features(10)
    out_i = mc.testing_index(10)
    out_l = mc.testing_labels(10)
    out_s = mc.data_distribution(model, out_f, out_l, bst_param_set['best_boost_num'])
    out_s = pd.Series(out_s)
    out_s.name = 'scores'
    
    out_f.to_csv('testing_sample_features.csv', index=False, encoding='gbk')
    out_s.to_csv('testing_sample_scores.csv', index=False, encoding='gbk')
    out_i.to_csv('testing_sample_index.csv', index=False, encoding='gbk')
    
    print(time.clock() - t_start)

    # mc.parse_trees(trees_details)
