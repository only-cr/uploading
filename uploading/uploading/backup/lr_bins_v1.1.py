import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, roc_curve, auc
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
import json
import numpy as np

class myClassifier:
    def __init__(self):
        self.data_filepath = 'F:/ModelTest/data/model2_data.csv'
        self.training_features = []
        self.training_labels = []
        self.testing_features = []
        self.testing_labels = []
        self.trees_details = ''
        self.feature_bins = {}
        self.feature_nan_score = {}
        self.current_training_features = []
        self.current_training_labels = []
        self.validation_features = []
        self.validation_labels = []
        self.k_fold = []
        self.cv_num = 3
        pass

    def load_data(self):
        # load data from file
        raw_data = pd.read_csv(self.data_filepath)
        total_feature_names = raw_data.columns.tolist()
        raw_data = raw_data.drop(total_feature_names[-1], axis=1)
        raw_labels = raw_data['y']
        raw_data = raw_data.drop('y', axis=1)
        total_feature_names = total_feature_names[:-2]

        # separate numeric and categorical features
        numeric_feature_names = []
        categorical_feature_names = []
        for tmp_col_nm in total_feature_names:
            try:
                raw_data.loc[:, tmp_col_nm] = raw_data[tmp_col_nm].astype('float')
                numeric_feature_names.append(tmp_col_nm)
            except ValueError:
                categorical_feature_names.append(tmp_col_nm)

        # convert categorical data to dummy variables
        raw_categorical_data = raw_data[categorical_feature_names]
        raw_data = raw_data.drop(categorical_feature_names, axis=1)

        #
        dummy_variables = self.one_hot_encoding(raw_categorical_data)
        raw_data = raw_data.join(dummy_variables)

        # split data into training and testing sets
        self.training_features, self.testing_features, self.training_labels, self.testing_labels = train_test_split(raw_data, raw_labels,test_size=0.1,
                                                                                                          random_state=6)
        pass

    def model_training(self, params, training_features, training_labels, validation_features, validation_labels):
        # params['colsample_bytree'] = 1. / training_features.shape[1]  # control only one feature each tree

        # use validation set to find the best tree num by early stopping
        pos_count = training_labels.sum()
        neg_count = training_labels.size - pos_count
        params['scale_pos_weight'] = neg_count / float(pos_count)

        training_dmatrix = xgb.DMatrix(training_features, label=training_labels, feature_names=training_features.columns)
        validation_dmatrix = xgb.DMatrix(training_features, label=training_labels)

        model = xgb.train(params, training_dmatrix, num_boost_round=50000, verbose_eval=100, early_stopping_rounds=15,
                          evals=[(validation_dmatrix, 'eval'), (training_dmatrix, 'train')])

        # combine original training data and validation data and train again
        training_features = training_features.append(validation_features)
        training_labels = training_labels.append(validation_labels)
        training_dmatrix = xgb.DMatrix(training_features, label=training_labels, feature_names=training_features.columns.tolist())

        best_tree_num = model.best_ntree_limit
        print('best tree number:', best_tree_num)

        pos_count = training_labels.sum()
        neg_count = training_labels.size - pos_count
        params['scale_pos_weight'] = neg_count / float(pos_count)
        model = xgb.train(params, training_dmatrix, num_boost_round=best_tree_num)

        model.save_model('crd_ver1.m')

        trees_details = model.get_dump(with_stats=True, dump_format='json')  # string in json format, containing splits, weights and so on

        return model, trees_details


    def model_accessing(self, model, testing_features, testing_labels):
        testing_dmatrix = xgb.DMatrix(testing_features, label=testing_labels, feature_names=testing_features.columns)
        pro_pred = model.predict(testing_dmatrix, ntree_limit=model.best_ntree_limit).tolist()
        label_pred = list(map(lambda x: 1 if x > 0.5 else 0, pro_pred))  # convert probability to label

        # accessment
        accessment_figures = {}
        accessment_figures['precision'] = precision_score(testing_labels, label_pred, average='micro')
        fpr, tpr, tmp_thrs = roc_curve(testing_labels, pro_pred, pos_label=1)
        accessment_figures['ks_score'] = np.max(tpr - fpr)
        accessment_figures['auc'] = auc(fpr, tpr)

        return accessment_figures

    def hyper_optimization(self):
        params_space = {
            'silent': 1,
            'booster': 'dart',
            'nthread': 3,
            'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
            'min_child_weight': hp.uniform('min_child_weight', 0, 2),
            'max_depth': 1,
            'gamma': hp.uniform('gamma', 0, 20),  # min loss reduce
            'subsample': hp.uniform('subsample', 0.5, 1),  # select samples
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),  # select features
            'max_delta_step': hp.uniform('max_delta_step', 1, 10),
            # non-zero when sample imbalance, (when too few samples in a node, G is close to zero, weight would be very large)
            'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # L2 regularization
            'reg_alpha': hp.uniform('reg_alpha', 0, 2),  # L1 regularization
            # 'scale_pos_weight': 1,  # weight for sample imbalance
            'objective': 'binary:logistic',
            # 'num_class':3, # multi-class classification
            'seed': 6,  # random state
            'eval_metric': 'auc',

            # dart params
            'sample_type': 'uniform',
            'normalize_type': 'tree',
            'rate_drop': hp.uniform('rate_drop', 0, 0.5),
        }

        self.current_training_features, self.validation_features, self.current_training_labels, self.validation_labels = train_test_split(self.training_features, self.training_labels,
            test_size=0.1, random_state=6)

        self.k_fold = KFold(n_splits=self.cv_num, shuffle=False, random_state=6)  # k-fold cross validation
        self.k_fold.get_n_splits(self.training_features)

        tmp_trial = Trials()
        best_sln = fmin(self.opt_objective, space=params_space, algo=tpe.suggest, max_evals=100, trials=tmp_trial)  # find the best hyper parameters

        print('best params:\n', best_sln)
        for tmp_k, tmp_v in params_space.items():  # fmin only return params not fixed before
            if tmp_k not in best_sln.keys():
                best_sln[tmp_k] = tmp_v

        tmp_idx = np.argmin(np.array(tmp_trial.losses()))
        best_boost_num = tmp_trial.results[tmp_idx]['tree_num']  # best number of trees from early stopping

        print('best num of trees:\n', best_boost_num)

        pos_count = self.training_labels.sum()
        neg_count = self.training_labels.size - pos_count
        best_sln['scale_pos_weight'] = neg_count / float(pos_count)
        training_dmatrix = xgb.DMatrix(self.training_features, label=self.training_labels, feature_names=self.training_features.columns)
        model = xgb.train(best_sln, training_dmatrix, num_boost_round=best_boost_num)   # train on the whole dataset with best parameter set

        model_details = model.get_dump(with_stats=True, dump_format='json')

        return model, model_details


    def opt_objective(self, args):
        params = args

        pro_pred = []
        for train_index, val_index in self.k_fold.split(self.training_features):
            training_features = self.training_features.iloc[train_index]
            training_labels = self.training_labels.iloc[train_index]
            validation_features = self.training_features.iloc[val_index]
            validation_labels = self.training_labels.iloc[val_index]

            pos_count = training_labels.sum()
            neg_count = training_labels.size - pos_count
            params['scale_pos_weight'] = neg_count / float(pos_count)

            training_dmatrix = xgb.DMatrix(training_features, label=training_labels, feature_names=training_features.columns)
            validation_dmatrix = xgb.DMatrix(validation_features, label=validation_labels, feature_names=validation_features.columns)

            model = xgb.train(params, training_dmatrix, num_boost_round=50000, verbose_eval=100, early_stopping_rounds=20,
                              evals=[(validation_dmatrix, 'eval'), (training_dmatrix, 'train')])

            # accessment on validation set
            tmp_pro_pred = model.predict(validation_dmatrix, ntree_limit=model.best_ntree_limit).tolist()
            pro_pred.extend(tmp_pro_pred)

        label_pred = list(map(lambda x: 1 if x > 0.5 else 0, pro_pred))  # convert probability to label

        # accessment
        accessment_figures = {}
        accessment_figures['precision'] = precision_score(self.training_labels, label_pred, average='micro')
        fpr, tpr, tmp_thrs = roc_curve(self.training_labels, pro_pred, pos_label=1)
        accessment_figures['ks_score'] = np.max(tpr - fpr)
        accessment_figures['auc'] = auc(fpr, tpr)

        acces_figures = self.model_accessing(model, validation_features, validation_labels)


        return {'loss': -acces_figures['auc'], 'status': STATUS_OK, 'tree_num': best_tree_num}

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

            tmp_real_scores = [0] * len(tmp_real_bins)
            for tmp_single_bin, tmp_single_score in zip(tmp_ori_bins, tmp_ori_scores): # convert scores on fake bins to real bins
                if tmp_single_bin[0] + 1 != tmp_single_bin[0]: # left limit is not -inf:  >=
                    tmp_index = tmp_split_points.index(tmp_single_bin[0]) # find covered real bins

                    for i in range(tmp_index, len(tmp_real_bins)):
                        tmp_real_scores[i] += tmp_single_score

                elif tmp_single_bin[1] + 1 != tmp_single_bin[1]:
                    tmp_index = tmp_split_points.index(tmp_single_bin[0])



        pass

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

    def feature_selection(self):
        pass


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

    mc = myClassifier()
    mc.load_data()

    # new_training_features, validation_features, new_training_labels, validation_labels = train_test_split(mc.training_features, mc.training_labels, test_size=0.3, random_state=6)

    model, model_details = mc.hyper_optimization()

    # model, trees_details = mc.model_training(testing_params, new_training_features, new_training_labels, validation_features, validation_labels)

    acces_figures = mc.model_accessing(model, mc.testing_features, mc.testing_labels)

    print(acces_figures)

    # mc.parse_trees(trees_details)
