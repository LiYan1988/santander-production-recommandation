# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:28:41 2018

@author: lyaa
"""

# Train on 2015-02-28 to 2016-04-28, validate on 2016-05-28

from santander_helper import *

param = {'objective': 'multi:softprob',
         'eta': 0.1,
         'max_depth': 10,
         'silent': 1,
         'num_class': len(target_cols),
         'eval_metric': 'merror',
         'min_child_weight': 10,
         'min_split_loss': 1,
         'subsample': 0.7,
         'colsample_bytree': 0.7,
         'seed': 0}

n_repeats = 2
n_trees = 2
train, val = load_pickle('all_month_validate.pickle')

df, clfs, running_time = cv_all_month(param, train, val, n_features=350, num_boost_round=n_trees,
    n_repeats=n_repeats, random_state=0, verbose_eval=True)

save_pickle('cluster_1_1.pickle', (df, clfs, running_time))