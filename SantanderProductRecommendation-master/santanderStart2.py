"""
Code based on BreakfastPirate Forum post
__author__ : SRK
"""
#%%
import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import pickle

import xgboost as xgb
from sklearn import preprocessing, ensemble
			
def runXGB(train_X, train_y, seed_val=0):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 50

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)	
    return model

def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data
 
#%%
x_train, y_train = read_data('train_mj2.pkl')
x_test = read_data('test_mj2.pkl')
  
print("Building model..")
#model = runXGB(x_train, y_train, seed_val=0)
clf = xgb.XGBClassifier(max_depth=8, objective='multi:softprob', 
                              learning_rate=.05, silent=1, min_child_weight=1,
                              subsample=0.7, colsample_bytree=0.7, seed=0,
                              n_estimators=50)
clf.fit(x_train, y_train, eval_metric='mlogloss', verbose=True, 
        eval_set=[(x_train, y_train)])
print("Predicting..")
preds_train = clf.predict(x_train)
preds = clf.predict(x_test)
#xgtest = xgb.DMatrix(x_test)
#preds = model.predict(xgtest)

#%%
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]
target_cols = np.array(target_cols)

#%%
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv("test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('sub_xgb_new.csv', index=False)
