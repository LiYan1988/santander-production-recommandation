# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:27:35 2016

@author: celin
"""

from santanderStart import *

param = {}
param['objective'] = ['multi:softprob']
param['learning_rate'] = [0.1]
param['max_depth'] = [9, 12, 15, 18]
param['silent'] = [1]
param['min_child_weight'] = [1, 2, 3]
param['subsample'] = [0.2, 0.4, 0.6, 0.8]
param['colsample_bytree'] = [0.2, 0.4, 0.6, 0.8]
param['seed'] = [0]
param['reg_alpha'] = [0, 1, 2]
param['reg_lambda'] = [1, 3]
param['gamma'] = [1, 3, 5]
param['n_estimators'] = [3000]
#param['nthread'] = 10

param_list = list(model_selection.ParameterSampler(param, n_iter=50, 
                                                   random_state=0))

save_data('param_list.pkl', param_list)