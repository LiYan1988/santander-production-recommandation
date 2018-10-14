# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:22:31 2016

@author: celin
"""

from santanderStart import *

x_train, y_train = read_data('train_mj2.pkl')
x_test = read_data('test_mj2.pkl')

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


 
model = runXGB(train_X, train_y, seed_val=0)
del train_X, train_y
print("Predicting..")
xgtest = xgb.DMatrix(test_X)
preds = model.predict(xgtest)
del test_X, xgtest
print(datetime.datetime.now()-start_time)

print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('sub_xgb_new.csv', index=False)
print(datetime.datetime.now()-start_time)