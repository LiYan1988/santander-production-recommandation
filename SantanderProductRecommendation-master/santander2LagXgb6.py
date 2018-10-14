# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:44:01 2016

@author: celin
Mean:  0.989765937905  std:  0.00605249602943
"""


from santanderStart import *

#%%
x_train, y_train, x_test, test_may, le = read_data('train_data_xgb.pkl')

param ={'colsample_bytree': 0.6,
  'gamma': 3,
  'learning_rate': 0.01,
  'max_depth': 10,
  'min_child_weight': 1,
  'n_estimators': 3000,
  'objective': 'multi:softprob',
  'reg_alpha': 0,
  'reg_lambda': 3,
  'seed': 0,
  'silent': 1,
  'subsample': 0.6}

#%%
clf = xgb.XGBClassifier()
clf.set_params(**param)
y_test_pred, y_train_pred, mlogloss, ntree = \
    cv_predict_xgb(clf, x_train, y_train, x_test, cv=3, 
                   random_state=0, esr=100)
    
#%%
save_submission('sub2.csv', y_test_pred, le, x_test, test_may)
