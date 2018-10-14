#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:31:04 2016

@author: lyaa
"""

from santanderStart import *

x_train, y_train, x_test, test_may, le = read_data('train_data_xgb_hebbe5_1.pkl')
param_list = read_data('param_list.pkl')
param = param_list[0]

#%%
clf = xgb.XGBClassifier()
y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params = \
    xgb_gridcv(clf, param_list, x_train, y_train, x_test, 
               cv=3, random_state=0, esr=10)

#%%
save_data('HO0.pkl', (y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params))