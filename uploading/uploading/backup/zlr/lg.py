import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score,roc_curve,classification_report,auc,confusion_matrix
from sklearn.decomposition import PCA
from hyperopt import hp, tpe, fmin
import xgboost as xgb
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
global train_data, val_data

def result_print(x, y, clf, type, thre=0.5):
    # y_predict = clf.predict(x)
    y_score_tmp = clf.predict_proba(x)
    y_score = [i[1] for i in y_score_tmp]
    y_predict = [1 if i > thre else 0 for i in y_score]
    accuracy = accuracy_score(y, y_predict)
    fpr, tpr, threshods = roc_curve(y, y_score, pos_label=1)
    ks = np.max(np.abs(tpr - fpr))
    report = classification_report(y, y_predict)
    print('result of %s data: '%type)
    print('report: ')
    print(report)
    print('accuracy: ')
    print(accuracy)
    print('K-S: ')
    print(ks)
    return ks

data = pd.read_excel(r"E:\umf\data.xlsx")
print('the data has been loaded.')


X = data.drop(['y'], axis=1, )
Y = data['y']

# rlg = RandomizedLogisticRegression(random_state=88)
# rlg.fit(X, Y)

# print('the usful variable is: \n %s '%(X.columns[rlg.get_support()]))
# X = X[X.columns[rlg.get_support()].tolist()]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=88)

# lg = LogisticRegression(random_state=88, solver='lbfgs', n_jobs=3, max_iter=200, verbose=1)
# lg.fit(x_train, y_train, )
clf = LDA()
# clf.fit(x_train, y_train)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(4), scoring='accuracy')
rfecv.fit(x_train, y_train)
data = X.loc[:, rfecv.get_support().tolist()]
x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.33, random_state=88)
clf = LDA()
clf.fit(x_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

result_print(x_train, y_train, clf, 'train', 0.1)
result_print(x_test, y_test, clf, 'test', 0.1)


print("今年下半年，中美合拍的西游记即将正式开机，我继续扮演美猴王孙悟空，我会用美猴王艺术形象努力创造一个正能量的形象，文体两开花，弘扬中华文化，希望大家能多多关注。")