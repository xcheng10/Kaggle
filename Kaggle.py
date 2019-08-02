# # # This Python 3 environment comes with many helpful analytics libraries installed
# # # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # # For example, here's several helpful packages to load in 

# # import numpy as np # linear algebra
# # import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # # Input data files are available in the "../input/" directory.
# # # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# # import os
# # print(os.listdir("../input"))

# # # Load data to dataframe
# # raw_train = pd.read_csv("train.csv")
# # raw_test = pd.read_csv("test.csv")
# # X_train = raw_train.drop(columns=["ID_code","target"])
# # y_train = raw_train[["target"]] 
# # X_test = raw_test.drop(columns=["ID_code"])
# # y_test = raw_test[["ID_code"]]

# # import warnings

# # from imblearn.under_sampling import RandomUnderSampler
# # from imblearn.over_sampling import RandomOverSampler
# # from imblearn.pipeline import make_pipeline
# # from sklearn.feature_selection import SelectKBest
# # from sklearn.metrics import roc_auc_score
# # from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
# # from sklearn.preprocessing import StandardScaler

# # # #Model_01: logistic regression
# # # from sklearn.linear_model import LogisticRegression

# # # warnings.filterwarnings('ignore')
# # # param_grid = {'logisticregression__C': np.logspace(-1, 1, 7)}
# # # pipe = make_pipeline(RandomUnderSampler(), StandardScaler(), LogisticRegression(class_weight='balanced',
# # #                                                                                 random_state=0))
# # # model_01 = GridSearchCV(pipe, param_grid, cv=5)
# # # model_01.fit(X_train, y_train)

# # # score_01 = np.mean(cross_val_score(model_01, X_train, y_train, scoring='roc_auc', cv=7))
# # # print('Average ROC AUC Score: {:.3f}'.format(score_01))

# # # y_pred_01 = model_01.predict(X_test)
# # # #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))
# # # y_test['target'] = y_pred_01
# # # y_test_01 = y_test[['ID_code','target']]
# # # y_test_01.to_csv('submission_01.csv', index=False)
# # # # <odel_02: xgboost
# # # from xgboost import XGBClassifier

# # # model_02 = XGBClassifier(max_depth=2,
# # #                          learning_rate=1,
# # #                          min_child_weight = 1,
# # #                          subsample = 0.5,
# # #                          colsample_bytree = 0.1,
# # #                          scale_pos_weight = round(sum(y_train.target == 1)/len(y_train.target),2),
# # #                          #gamma=3,
# # #                          seed=0)
# # # model_02.fit(X_train, y_train.values)

# # # score_02 = np.mean(cross_val_score(model_02, X_train, y_train.values, scoring='roc_auc', cv=7))
# # # print('Average ROC AUC Score: {:.3f}'.format(score_02))

# # # y_pred_02 = model_02.predict(X_test)
# # # #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))

# # # y_test['target'] = y_pred_02
# # # y_test_02 = y_test[['ID_code','target']].copy()
# # # y_test_02.to_csv('submission.csv', encoding='utf-8', index=False)

# # # Model_03: extratrees
# # from sklearn.ensemble import ExtraTreesClassifier

# # model_03 = make_pipeline(RandomUnderSampler(), ExtraTreesClassifier(n_estimators=150,
# #                                                                     criterion='entropy',
# #                                                                     max_depth=8,
# #                                                                     min_samples_split=300,
# #                                                                     min_samples_leaf=15,
# #                                                                     random_state=0,
# #                                                                     class_weight='balanced_subsample'))
# # model_03.fit(X_train, y_train)

# # score_03 = np.mean(cross_val_score(model_03, X_train, y_train, scoring='roc_auc', cv=7))
# # print('Average ROC AUC Score: {:.3f}'.format(score_03))
# # y_pred_03 = model_03.predict(X_test)
# # y_test['target'] = y_pred_03
# # y_test_03 = y_test[['ID_code','target']].copy()
# # y_test_03.to_csv('submission_03.csv', encoding='utf-8', index=False)
# # # Any results you write to the current directory are saved as output.

# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# train_df = pd.read_csv('../input/train.csv')
# test_df = pd.read_csv('../input/test.csv')
# features = [c for c in train_df.columns if c not in ['ID_code', 'target']] #basic features
# target = train_df['target']
# # param = {
# #     'bagging_freq': 6,          
# #     'bagging_fraction': 0.9,   'boost_from_average':'false',   
# #     'boost': 'gbdt',             'feature_fraction': 0.04,     'learning_rate': 0.009,
# #     'max_depth': -1,             'metric':'auc',                'min_data_in_leaf': 90,     'min_sum_hessian_in_leaf': 10.0,
# #     'num_leaves': 13,            'num_threads': 8,              'tree_learner': 'serial',   'objective': 'binary',
# #     'reg_alpha': 0.14, 'reg_lambda': 0.37,'verbosity': 1
# # }
# param = {
#     'bagging_freq': 5,
#     'bagging_fraction': 0.335,
#     'boost_from_average':'false',
#     'boost': 'gbdt',
#     'feature_fraction': 0.041,
#     'learning_rate': 0.083,
#     'max_depth': -1,
#     'metric':'auc',
#     'min_data_in_leaf': 80,
#     'min_sum_hessian_in_leaf': 10.0,
#     'num_leaves': 13,
#     'num_threads': 8,
#     'tree_learner': 'serial',
#     'objective': 'binary', 
#     'verbosity': -1
# }
# folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=99999)
# oof = np.zeros(len(train_df))
# predictions = np.zeros(len(test_df))
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
#     print("Fold {}".format(fold_))
#     trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
#     val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
#     clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 2000)
#     oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
#     predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
# print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
# sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
# sub["target"] = predictions
# sub.to_csv("submission.csv", index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1)

random_state = 42
np.random.seed(random_state)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y
    
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()
features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = yp/N

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)