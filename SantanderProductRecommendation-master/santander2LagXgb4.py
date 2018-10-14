# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:17:37 2016

@author: liyan

Best:
     {'colsample_bytree': 0.6,
  'gamma': 3,
  'learning_rate': 0.1,
  'max_depth': 10,
  'min_child_weight': 1,
  'n_estimators': 3000,
  'objective': 'multi:softprob',
  'reg_alpha': 0,
  'reg_lambda': 3,
  'seed': 0,
  'silent': 1,
  'subsample': 0.6}
  0.99473818
Second:
     {'colsample_bytree': 0.6,
  'gamma': 3,
  'learning_rate': 0.1,
  'max_depth': 8,
  'min_child_weight': 1,
  'n_estimators': 3000,
  'objective': 'multi:softprob',
  'reg_alpha': 0,
  'reg_lambda': 3,
  'seed': 0,
  'silent': 1,
  'subsample': 0.5},
  0.99508731
"""


from santanderStart import *

#%%
x_train, y_train, x_test, test_may, le = read_data('train_data_xgb.pkl')

param = {}
param['objective'] = ['multi:softprob']
param['learning_rate'] = [0.1]
param['max_depth'] = [6, 8, 10]
param['silent'] = [1]
param['min_child_weight'] = [1]
param['subsample'] = [0.4, 0.5, 0.6]
param['colsample_bytree'] = [0.4, 0.5, 0.6]
param['seed'] = [0]
param['reg_alpha'] = [0]
param['reg_lambda'] = [2, 3]
param['gamma'] = [2, 3]
param['n_estimators'] = [3000]

param_list = list(model_selection.ParameterSampler(param, n_iter=8, 
                                                   random_state=0))

#%%
clf = xgb.XGBClassifier()
y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params = \
    xgb_gridcv(clf, param_list, x_train, y_train, x_test, 
               cv=3, random_state=0, esr=10)

#%%
save_data('HOxgb4.pkl', (y_test_pred_list,y_train_pred_list,
                         obj_list,ntree_list,params))
