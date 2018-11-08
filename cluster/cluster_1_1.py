# Train on 2015-02-28 to 2016-04-28, validate on 2016-05-28

from santander_helper import *

create_monthly_data()

target = calculate_customer_product_pair()

mean_encoding_result = mean_encoding_month_product()

x_train = []
y_train = []
w_train = []
fixed_lag = 6
for i, m in enumerate(month_list):
    if m in ['2015-01-28', '2016-06-28']:
        continue
    x_tmp, y_tmp, w_tmp = create_train(m, max_lag=i, fixed_lag=fixed_lag, pattern_flag=True)
    x_train.append(x_tmp)
    y_train.append(y_tmp)
    w_train.append(w_tmp)
del x_tmp, y_tmp, w_tmp
gc.collect()

# Prepare for train and validation
x_val = x_train[-1]
y_val = y_train[-1]
w_val = w_train[-1]

x_train = pd.concat(x_train[:-1], axis=0, ignore_index=True, sort=False)
y_train = pd.concat(y_train[:-1], axis=0, ignore_index=True, sort=False)
w_train = pd.concat(w_train[:-1], axis=0, ignore_index=True, sort=False)

gc.collect()

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

# number of rows in train dataset, to simplify testing, always set to None
n_rows = None
n_repeats = 2
n_trees = 2
train = {'x': x_train.iloc[:n_rows, :], 'y': y_train.iloc[:n_rows], 'w': w_train.iloc[:n_rows]}
val = {'x': x_val.iloc[:n_rows, :], 'y': y_val.iloc[:n_rows], 'w': w_val.iloc[:n_rows]}
#df, clfs, running_time = cv_all_month(param, train, val, n_features=350, num_boost_round=n_trees,
#    n_repeats=n_repeats, random_state=0, verbose_eval=True)

#save_pickle('cluster_1_1.pickle', (df, clfs, running_time))