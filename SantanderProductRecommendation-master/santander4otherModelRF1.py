#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:42:17 2016

@author: lyaa
"""

from santanderStart import *
from sklearn.ensemble import RandomForestClassifier

x_train, y_train, x_test, test_may, le = read_data('train_data_xgb_hebbe5_1.pkl')

clf = RandomForestClassifier(n_jobs=15, verbose=10, n_estimators=150, 
                             max_depth=12, max_features=0.7, max_leaf_nodes=50)
y_test_pred, y_train_pred, obj = cv_predict(clf, x_train, y_train, x_test, cv=3, random_state=0)

save_submission('sub3.csv', y_test_pred, le, x_test, test_may)