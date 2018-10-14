#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:54:23 2016

@author: lyaa
logloss CV->Mean:  1.03275552048  std:  0.00536395197864
LB->0.0296469
"""

from santanderStart import *

train, _ = read_data('draft_hebbe3_train.pkl')
test_may, test_june = read_data('draft_hebbe3_test.pkl')

target_cols = train.variable.unique().tolist()
le = preprocessing.LabelEncoder()
le.fit(target_cols)

y_train = train.variable
y_train = le.transform(y_train)
y_train = pd.Series(y_train)

x_train = train.copy()
x_train.drop(['variable'], axis=1, inplace=True)
x_train.fillna(-1, inplace=True)

x_test = test_june.copy()

cols_drop = [col for col in test_may.columns if col not in target_cols]
cols_drop.remove('ncodpers')
test_may.drop(cols_drop, axis=1, inplace=True)

del train, test_june
gc.collect()

#%%
clfxgb = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.05, max_depth=8,
                           min_child_weight=1, subsample=0.7, colsample_bytree=0.7, seed=0, 
                           n_estimators=200)

#%%
#y_test_pred, y_train_pred, mlogloss, ntree = \
#    cv_predict_xgb(clfxgb, x_train, y_train, x_test, cv=3, 
#                   random_state=0, esr=5)

y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred = \
    cv_predict_xgb_repeat(clfxgb, x_train, y_train, x_test, 
                          cv=3, random_state=0, rep=5, esr=10)

#%%
save_submission('sub1.csv', y_test_pred_mean, le, x_test, test_may)
