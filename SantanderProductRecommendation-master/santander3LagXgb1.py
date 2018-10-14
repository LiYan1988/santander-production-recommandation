#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:26:04 2016

@author: lyaa
Mean:  0.996888842696  std:  0.00194681287598
LB: 0.0297225
"""

from santanderStart import *

#train, _ = read_data('draft_hebbe5_1_train.pkl')
#test_may, test_june = read_data('draft_hebbe5_1_test.pkl')
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
#save_data('train_data_xgb_hebbe5_1.pkl', (x_train, y_train, x_test, test_may, le))

#%%
x_train, y_train, x_test, test_may, le = read_data('train_data_xgb_hebbe5_1.pkl')

param = {}
param['objective'] = ['multi:softprob']
param['learning_rate'] = [0.1]
param['max_depth'] = [10]
param['silent'] = [1]
param['min_child_weight'] = [1]
param['subsample'] = [0.6]
param['colsample_bytree'] = [0.6]
param['seed'] = [0]
param['reg_alpha'] = [0]
param['reg_lambda'] = [3]
param['gamma'] = [3]
param['n_estimators'] = [3000]
#param['nthread'] = 10

param_list = list(model_selection.ParameterSampler(param, n_iter=1, 
                                                   random_state=0))

#%%
clf = xgb.XGBClassifier()
y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params = \
    xgb_gridcv(clf, param_list, x_train, y_train, x_test, 
               cv=3, random_state=0, esr=10)

#%%
save_submission('sub1.csv', y_test_pred_list[0], le, x_test, test_may)
