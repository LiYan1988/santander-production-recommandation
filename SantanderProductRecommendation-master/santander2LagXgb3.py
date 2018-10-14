# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:17:04 2016

@author: liyan
Test hyperparameter optimization

Best parameter:
 {'colsample_bytree': 0.5,
  'gamma': 2,
  'learning_rate': 0.1,
  'max_depth': 8,
  'min_child_weight': 1,
  'n_estimators': 3000,
  'objective': 'p',
  'reg_alpha': 0.5,
  'reg_lambda': 2,
  'seed': 0,
  'silent': 1,
  'subsample': 0.7}
"""

from santanderStart import *

#train, _ = read_data('draft_hebbe3_train.pkl')
#test_may, test_june = read_data('draft_hebbe3_test.pkl')
#
#target_cols = train.variable.unique().tolist()
#le = preprocessing.LabelEncoder()
#le.fit(target_cols)
#
#y_train = train.variable
#y_train = le.transform(y_train)
#y_train = pd.Series(y_train)
#
#x_train = train.copy()
#x_train.drop(['variable'], axis=1, inplace=True)
#x_train.fillna(-1, inplace=True)
#
#x_test = test_june.copy()
#
#cols_drop = [col for col in test_may.columns if col not in target_cols]
#cols_drop.remove('ncodpers')
#test_may.drop(cols_drop, axis=1, inplace=True)
#
#del train, test_june
#gc.collect()
#
#save_data('train_data_xgb.pkl', (x_train, y_train, x_test, test_may, le))

#%%
x_train, y_train, x_test, test_may, le = read_data('train_data_xgb.pkl')

param = {}
param['objective'] = ['multi:softprob']
param['learning_rate'] = [0.1]
param['max_depth'] = [8]
param['silent'] = [1]
param['min_child_weight'] = [1]
param['subsample'] = [0.3, 0.5, 0.7]
param['colsample_bytree'] = [0.3, 0.5, 0.7]
param['seed'] = [0]
param['reg_alpha'] = [0.5, 1]
param['reg_lambda'] = [1.5, 2]
param['gamma'] = [1, 2]
param['n_estimators'] = [3000]

param_list = list(model_selection.ParameterSampler(param, n_iter=8, 
                                                   random_state=0))

#%%
clf = xgb.XGBClassifier()
y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params = \
    xgb_gridcv(clf, param_list, x_train, y_train, x_test, 
               cv=3, random_state=0, esr=10)

#%%
#save_submission('sub1.csv', y_test_pred_mean, le, x_test, test_may)
