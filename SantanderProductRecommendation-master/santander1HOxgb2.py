# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:58:46 2016

@author: celin
"""
#%%
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
param['seed'] = 0
num_rounds = 500

#%%
kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=0)
y_test_pred = []
y_train_pred = np.zeros((y_train.shape[0],22))
train_index, test_index = next(kf.split(x_train))
x_train1 = x_train[train_index]
y_train1 = y_train[train_index]
x_train2 = x_train[test_index]
y_train2 = y_train[test_index]
xgtrain = xgb.DMatrix(x_train1, label=y_train1)
xgvalid = xgb.DMatrix(x_train2, label=y_train2)
model = xgb.train(list(param.items()), xgtrain, num_rounds, evals=[(xgvalid, 'valid')],
                  early_stopping_rounds=5)
y_pred2 = model.predict(xgvalid, ntree_limit=model.best_ntree_limit)
y_train_pred[test_index] = y_pred2

#%%
ntree.append(model.best_ntree_limit)
#        reg.n_estimators = ntree[-1]
#        reg.fit(x_train1, y_train1)
y_pred2 = model.predict(xgvalid)
y_train_pred[test_index] = y_pred2
mae.append(mae_invlogs(y_train2, y_pred2))
y_test_pred.append(reg.predict(x_test))

print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('sub_xgb_new.csv', index=False)
print(datetime.datetime.now()-start_time)