import os
if os.name=='nt':
    try:
        mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
        os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    except:
        pass
    
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import gc
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
import copy
import time
import collections
from numba import jit
import pickle
import itertools

tqdm.tqdm.pandas()


################################ Define constants ###############################
cat_cols = ['ncodpers',
 'canal_entrada',
 'conyuemp',
 'ind_actividad_cliente',
 'ind_empleado',
 'ind_nuevo',
 'indext',
 'indfall',
 'indrel',
 'indrel_1mes',
 'indresi',
 'pais_residencia',
 'segmento',
 'sexo',
 'tipodom',
 'tiprel_1mes',
 'age',
 'antiguedad',
 'renta']

target_cols = ['ind_cco_fin_ult1',
 'ind_cder_fin_ult1',
 'ind_cno_fin_ult1',
 'ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1',
 'ind_ctop_fin_ult1',
 'ind_ctpp_fin_ult1',
 #'ind_deco_fin_ult1',
 'ind_dela_fin_ult1',
 #'ind_deme_fin_ult1',
 'ind_ecue_fin_ult1',
 'ind_fond_fin_ult1',
 'ind_hip_fin_ult1',
 'ind_nom_pens_ult1',
 'ind_nomina_ult1',
 'ind_plan_fin_ult1',
 'ind_pres_fin_ult1',
 'ind_reca_fin_ult1',
 'ind_recibo_ult1',
 'ind_tjcr_fin_ult1',
 'ind_valo_fin_ult1']
 #'ind_viv_fin_ult1']
    
month_list = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28', 
              '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', 
              '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28', '2016-06-28']
			  

################################ Feature Engineering Functions ###############################
def encoding(x):
    '''
    Encoding the pattern in one product for one customer
    (previous, this):
    (0, 0): 0
    (0, 1): 2
    (1, 0): 1
    (1, 1): 3
    '''
    a, b = x.values[:-1, :], x.values[1:, :]
    c = a+b*2
    c = pd.DataFrame(c, index=x.index[0:-1], columns=x.columns)
    
    return c

def count_changes(dt):
    '''Process for the whole dataframe'''

    # group by customer
    group = dt.groupby('ncodpers')[target_cols]
    # encode patterns
    print('Encoding pattern...')
    dt_changes = group.progress_apply(encoding)
    
    # find appearance each patterns
    print('Finding pattern...')
    a3 = (dt_changes==3.0).astype(int)
    a3.columns = [k+'_p3' for k in a3.columns]
    a2 = (dt_changes==2.0).astype(int)
    a2.columns = [k+'_p2' for k in a2.columns]
    a1 = (dt_changes==1.0).astype(int)
    a1.columns = [k+'_p1' for k in a1.columns]
    a0 = (dt_changes==0.0).astype(int)
    a0.columns = [k+'_p0' for k in a0.columns]
    a = pd.concat((a3, a2, a1, a0), axis=1)
    
    # count number of patterns
    print('Counting pattern...')
    dt_count = a.groupby('ncodpers').progress_apply(np.sum, axis=0)
    
    del group, dt_changes, a3, a2, a1, a0, a
    
    return dt_count

def count_pattern(month1, max_lag):
    '''
    Encoding the pattern in one product for one customer
    (previous, this):
    (0, 0): 0
    (0, 1): 2
    (1, 0): 1
    (1, 1): 3
    '''
        
    if os.path.exists('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag)):
    
        # directly load data if it exists
        pattern_count = pd.read_hdf('../input/count_pattern_{}_{}.hdf', 'pattern_count')
        return pattern_count
        
    else:
        month_end = month_list.index(month1)
        month_start = month_end-max_lag+1
        
        # Create a DataFrame containing all the previous months up to the month_index month
        df = []
        for m in range(month_start, month_end+1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        ncodpers_list = df[-1].ncodpers.unique().tolist()

        df = pd.concat(df, ignore_index=True)
        
        # count patterns for customers with at least two months records
        dt = count_changes(df)
        
        # create patterns for all customers, fillna with 0.0 if less than two months records
        pattern_count = df.loc[df.fecha_dato==month_list[month_end], ['ncodpers']]
        pattern_count.set_index('ncodpers', drop=False, inplace=True)
        pattern_count = pattern_count.join(dt)
        pattern_count.drop('ncodpers', axis=1, inplace=True)
        pattern_count.fillna(0.0, inplace=True)
           
        del dt, df, ncodpers_list
        gc.collect()
        
        pattern_count.to_hdf('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag), 'pattern_count')
        return pattern_count

def create_train_test(month, max_lag=5, target_flag=True, pattern_flag=False):
    '''Create train and test data for month'''
    
    start_time = time.time()
    
    month2 = month # the second month
    month1 = month_list[month_list.index(month2)-1] # the first month
    
    # check if max_lag and month are compatible
    assert month_list.index(month2)>=max_lag, 'max_lag should be less than the index of {}, which is {}'.format(
        month2, month_list.index(month2))
    
    print('Loading {} data'.format(month1))
    # first/early month
    df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    print('Loading {} data'.format(month2))
    # second/later month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    
    print('Products in {}'.format(month2))
    # second month products
    df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
    df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
    # a dataframe containing the ncodpers only
    df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
    # drop ncodpers from df2_target
    df2_target.drop('ncodpers', axis=1, inplace=True)
    
    print('Products in {}'.format(month1))
    # first month products for all the customers in the second month
    df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
    df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
    # obtain the products purchased by all the customers in the second month
    # by joining df1_target to df2_ncodpers, NAN filled by 0.0
    df1_target = df2_ncodpers.join(df1_target, how='left')
    df1_target.fillna(0.0, inplace=True)
    df1_target.drop('ncodpers', axis=1, inplace=True)
    
    print('New products added in {}'.format(month2))
    # new products from the first to second month
    target = df2_target.subtract(df1_target)
    target[target<0] = 0
    target.fillna(0.0, inplace=True)
    
    print('Join customer features and previous month products for {}'.format(month2))
    # feature of the second month: 
    # 1. customer features in the second month
    # 2. products in the first month
    x_vars = df2[cat_cols].copy() # cat_cols already includes ncodpers
    x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
    x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
    x_vars_cols = x_vars.columns.tolist()
    x_vars_cols[0] = 'sample_order' # change the name of the new column
    x_vars.columns = x_vars_cols
    x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again
    x_vars = x_vars.join(df1_target) # direct join since df1_target contains all customers in month2
    
    print('Concatenate this and previous months ind_activadad_cliente')
    # concatenate this and previous month values of ind_activadad_cliente
    df2_ind_actividad_cliente = df2[['ncodpers', 'ind_actividad_cliente']].copy()
    df2_ind_actividad_cliente.set_index('ncodpers', inplace=True)
    df2_ind_actividad_cliente.sort_index(inplace=True)
    
    df1_ind_actividad_cliente = df1[['ncodpers', 'ind_actividad_cliente']].copy()
    df1_ind_actividad_cliente.set_index('ncodpers', inplace=True)
    df1_ind_actividad_cliente.sort_index(inplace=True)

    df2_ind_actividad_cliente = df2_ind_actividad_cliente.join(df1_ind_actividad_cliente, rsuffix='_p')
    df2_ind_actividad_cliente.fillna(2.0, inplace=True)
    df2_ind_actividad_cliente['ind_actividad_client_combine'] = 3*df2_ind_actividad_cliente.ind_actividad_cliente+df2_ind_actividad_cliente.ind_actividad_cliente_p
    df2_ind_actividad_cliente = pd.DataFrame(df2_ind_actividad_cliente.iloc[:, -1])

    x_vars = pd.merge(x_vars, df2_ind_actividad_cliente, left_index=True, right_index=True, how='left')
    
    print('Concatenate this and previous months tiprel_1mes')
    # concatenate this and previous month value of tiprel_1mes
    df2_tiprel_1mes = df2[['ncodpers', 'tiprel_1mes']].copy()
    df2_tiprel_1mes.set_index('ncodpers', inplace=True)
    df2_tiprel_1mes.sort_index(inplace=True)

    df1_tiprel_1mes = df1[['ncodpers', 'tiprel_1mes']].copy()
    df1_tiprel_1mes.set_index('ncodpers', inplace=True)
    df1_tiprel_1mes.sort_index(inplace=True)

    df2_tiprel_1mes = df2_tiprel_1mes.join(df1_tiprel_1mes, rsuffix='_p')
    df2_tiprel_1mes.fillna(0.0, inplace=True)
    df2_tiprel_1mes['tiprel_1mes_combine'] = 6*df2_tiprel_1mes.tiprel_1mes+df2_tiprel_1mes.tiprel_1mes_p
    df2_tiprel_1mes = pd.DataFrame(df2_tiprel_1mes.iloc[:, -1])

    x_vars = pd.merge(x_vars, df2_tiprel_1mes, left_index=True, right_index=True, how='left')
    
    print('Combine all products for each customer')
    # combination of target columns
    x_vars['target_combine'] = np.sum(x_vars[target_cols].values*
        np.float_power(2, np.arange(0, len(target_cols))), axis=1, dtype=np.float64)
    # Load mean encoding data and merge with x_vars
    target_mean_encoding = pd.read_hdf('../input/target_mean_encoding_2.hdf', 'target_mean_encoding')
    x_vars = x_vars.join(target_mean_encoding, on='target_combine')

    # number of purchased products in the previous month
    x_vars['n_products'] = x_vars[target_cols].sum(axis=1)
    
    del (df1_tiprel_1mes, df2_tiprel_1mes, df1_ind_actividad_cliente, 
        df2_ind_actividad_cliente, df2_target, df1_target, df2_ncodpers)
    gc.collect()

    if pattern_flag:
        print('\nStart counting patterns:')
        # count patterns of historical products
        dp = count_pattern(month1, max_lag)
        x_vars = x_vars.join(dp)
        x_vars.loc[:, dp.columns] = x_vars.loc[:, dp.columns].fillna(-1)
        
        del dp
        gc.collect()
        
    # return x_vars if target_flag is False
    if not target_flag:
        x_vars.drop('sample_order', axis=1, inplace=True) # drop sample_order
        x_vars.reset_index(inplace=True, drop=False) # add ncodpers
        
        end_time = time.time()
        print('Time used: {:.3f} min'.format((end_time-start_time)/60.0))
        
        return x_vars 
    
    if target_flag:    
        print('Prepare target')
        # prepare target/label for each added product from the first to second month
        # join target to x_vars
        x_vars_new = x_vars.join(target, rsuffix='_t')
        # set ncodpers as one column
        x_vars_new.reset_index(inplace=True, drop=False)
        x_vars.reset_index(inplace=True, drop=False)
        var_cols = x_vars.columns.tolist()
        
        del x_vars
        gc.collect()
		
        # melt
        return x_vars_new
        x_vars_new = x_vars_new.melt(id_vars=var_cols)
        # mapping from target_cols to index
        target_cols_mapping = {c+'_t': n for (n, c) in enumerate(target_cols)}
        # replace column name by index
        x_vars_new.variable.replace(target_cols_mapping, inplace=True)
        # reorder rows
        x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)
        # keep new products
        x_vars_new = x_vars_new[x_vars_new.value>0]
        # drop sample_order and value
        x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
        # keep the order of rows as in the original data set
        x_vars_new.reset_index(drop=True, inplace=True)

        
        var_cols.remove('sample_order')
        # variable
        x_vars = x_vars_new.loc[:, var_cols].copy()
        # target/label
        target = x_vars_new.loc[:, 'variable'].copy()

        end_time = time.time()
        print('Time used: {:.3f} min'.format((end_time-start_time)/60.0))
        
        return x_vars, target
		

def obtain_target(month):
    '''Create train and test data for month'''
    
    month2 = month # the second month
    month1 = month_list[month_list.index(month2)-1] # the first month
    
    # first/early month
    df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    # second/later month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    
    # second month products
    df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
    df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
    # a dataframe containing the ncodpers only
    df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
    # drop ncodpers from df2_target
    df2_target.drop('ncodpers', axis=1, inplace=True)
    
    # first month products for all the customers in the second month
    df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
    df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
    # obtain the products purchased by all the customers in the second month
    # by joining df1_target to df2_ncodpers, NAN filled by 0.0
    df1_target = df2_ncodpers.join(df1_target, how='left')
    df1_target.fillna(0.0, inplace=True)
    df1_target.drop('ncodpers', axis=1, inplace=True)
    
    # new products from the first to second month
    target = df2_target.subtract(df1_target)
    target[target<0] = 0
    target.fillna(0.0, inplace=True)
    
    # feature of the second month: 
    # 1. customer features in the second month
    # 2. products in the first month
    x_vars = df2[['ncodpers']].copy() # cat_cols already includes ncodpers
    x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
    x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
    x_vars_cols = x_vars.columns.tolist()
    x_vars_cols[0] = 'sample_order' # change the name of the new column
    x_vars.columns = x_vars_cols
    x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again

    # prepare target/label for each added product from the first to second month
    # join target to x_vars
    x_vars_new = x_vars.join(target)
    # set ncodpers as one column
    x_vars_new.reset_index(inplace=True, drop=False)
    x_vars.reset_index(inplace=True, drop=False)
    
    # melt
    x_vars_new = x_vars_new.melt(id_vars=x_vars.columns)
    # mapping from target_cols to index
    target_cols_mapping = {c: n for (n, c) in enumerate(target_cols)}
    # replace column name by index
    x_vars_new.variable.replace(target_cols_mapping, inplace=True)
    # reorder rows
    x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)

    # keep new products
    x_vars_new = x_vars_new[x_vars_new.value>0]
    # drop sample_order and value
    x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
    # keep the order of rows as in the original data set
    x_vars_new.reset_index(drop=True, inplace=True)
    
    x_vars_new.columns = ['ncodpers', 'target']

    return x_vars_new

def check_target(month1, month2, target_flag=True):
    '''Create train and test data between month1 and month2'''
    
    # first/early month
    df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    # second/later month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    
    # second month products
    df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
    df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
    # a dataframe containing the ncodpers only
    df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
    # drop ncodpers from df2_target
    df2_target.drop('ncodpers', axis=1, inplace=True)
    
    # first month products for all the customers in the second month
    df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
    df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
    # obtain the products purchased by all the customers in the second month
    # by joining df1_target to df2_ncodpers, NAN filled by 0.0
    df1_target = df2_ncodpers.join(df1_target, how='left')
    df1_target.fillna(0.0, inplace=True)
    df1_target.drop('ncodpers', axis=1, inplace=True)
    
    # new products from the first to second month
    target = df2_target.subtract(df1_target)
    target[target<0] = 0
    target.fillna(0.0, inplace=True)
    
    # feature of the second month: 
    # 1. customer features in the second month
    # 2. products in the first month
    x_vars = df2[cat_cols].copy() # cat_cols already includes ncodpers
    x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
    x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
    x_vars_cols = x_vars.columns.tolist()
    x_vars_cols[0] = 'sample_order' # change the name of the new column
    x_vars.columns = x_vars_cols
    x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again
    x_vars = x_vars.join(df1_target) # direct join since df1_target contains all customers in month2
    
    # return x_vars if target_flag is False
    if not target_flag:
        x_vars.drop('sample_order', axis=1, inplace=True) # drop sample_order
        x_vars.reset_index(inplace=True, drop=False) # add ncodpers
        return x_vars #, df2_ncodpers, df1, df2, df1_target, df2_target
    
    if target_flag:    
        # prepare target/label for each added product from the first to second month
        # join target to x_vars
        x_vars_new = x_vars.join(target, rsuffix='_t')
        # set ncodpers as one column
        x_vars_new.reset_index(inplace=True, drop=False)
        x_vars.reset_index(inplace=True, drop=False)

        # melt
        x_vars_new = x_vars_new.melt(id_vars=x_vars.columns)
        # mapping from target_cols to index
        target_cols_mapping = {c+'_t': n for (n, c) in enumerate(target_cols)}
        # replace column name by index
        x_vars_new.variable.replace(target_cols_mapping, inplace=True)
        # reorder rows
        x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)
        # keep new products
        x_vars_new = x_vars_new[x_vars_new.value>0]
        # drop sample_order and value
        x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
        # keep the order of rows as in the original data set
        x_vars_new.reset_index(drop=True, inplace=True)

        var_cols = x_vars.columns.tolist()
        var_cols.remove('sample_order')
        # variable
        x_vars = x_vars_new.loc[:, var_cols].copy()
        # target/label
        target = x_vars_new.loc[:, 'variable'].copy()

        return x_vars, target, x_vars_new
        
def count_zeros(month1, max_lag):
    if os.path.exists('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag)):
        df = pd.read_hdf('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag), 
            'count_zeros')
        
        return df
    else:
        month_new = month_list.index(month1)+1
        month_end = month_list.index(month1)
        month_start = month_end-max_lag+1
        
        # Check if month_new is the last month
        if month_new<len(month_list)-1:
            # Customers with new products in month_new
            customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
            ncodpers_list = customer_product_pair.loc[customer_product_pair.fecha_dato==month_list[month_new], 
                'ncodpers'].unique().tolist()

        # Load data for all the lag related months
        df = []
        for m in range(month_start, month_end+1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        # concatenate data
        df = pd.concat(df, ignore_index=True)
        df = df.loc[:, ['ncodpers', 'fecha_dato']+target_cols]
        if month_new<len(month_list)-1:
            # select customers if this is not test set
            df = df.loc[df.ncodpers.isin(ncodpers_list), :]
        # set ncodpers and fecha_dato as index
        df.set_index(['ncodpers', 'fecha_dato'], inplace=True)
        # unstack to make month as columns
        df = df.unstack(level=-1, fill_value=0)

        # count number of concatenating zeros before the second/current month
        df = df.groupby(level=0, axis=1).progress_apply(lambda x: (1-x).iloc[:, ::-1].cummin(axis=1).sum(axis=1))
        df.columns = [k+'_zc' for k in df.columns]
        
        gc.collect()
        
        df.to_hdf('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag), 'count_zeros')
        
        return df
    
############################# The following functions are more updated
        
def create_train(month, max_lag=5, pattern_flag=False):
    '''Another method to create train data sets'''
    
    # First check if the data is saved.
    try:
        if os.path.exists('../input/x_train_{}_{}.hdf'.format(month, max_lag)):
            x_train = pd.read_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'x_train')
            y_train = pd.read_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'y_train')
            weight = pd.read_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'weight')
            
            return x_train, y_train, weight
    except:
        pass
    
    month2 = month # the second month
    month1 = month_list[month_list.index(month2)-1] # the first month
    
    # Load customer product pair
    customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
    
    # Load second month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    df2 = df2.loc[:, cat_cols]
    df2 = df2.loc[df2.ncodpers.isin(customer_product_pair.loc[customer_product_pair.fecha_dato==month2].ncodpers.unique())]
    
    # Load first month
    df1_0 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    df1 = df1_0.loc[:, cat_cols+target_cols]
    df1_target = df1_0.loc[:, ['ncodpers']+target_cols]
    
    # Merge first month product with second month customer information
    df2 = df2.merge(df1_target, on='ncodpers', how='left')
    df2.fillna(0.0, inplace=True)
    
    # Combination of ind_activadad_cliente
    # second month ind_actividad_cliente
    df2_copy = df2.loc[:, ['ncodpers', 'ind_actividad_cliente']].copy()
    # first month ind_actividad_cliente
    df1_copy = df1.loc[:, ['ncodpers', 'ind_actividad_cliente']].copy()
    # merge two months
    df2_copy = pd.merge(df2_copy, df1_copy, on='ncodpers', suffixes=('', '_prev'), how='left')
    # fillna
    df2_copy.fillna(2.0, inplace=True)
    # combine 
    df2_copy['ind_actvidad_client_combine'] = df2_copy.ind_actividad_cliente.values*3+df2_copy.ind_actividad_cliente_prev.values
    # drop other columns
    df2_copy.drop(['ind_actividad_cliente', 'ind_actividad_cliente_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')
    
    # Combination of tiprel_1mes
    # second month tiprel_1mes
    df2_copy = df2.loc[:, ['ncodpers', 'tiprel_1mes']].copy()
    # first month tiprel_1mes
    df1_copy = df1.loc[:, ['ncodpers', 'tiprel_1mes']].copy()
    # merge two months
    df2_copy = pd.merge(df2_copy, df1_copy, on='ncodpers', suffixes=('', '_prev'), how='left')
    # fillna
    df2_copy.fillna(0.0, inplace=True)
    # combine 
    df2_copy['tiprel_1mes_combine'] = df2_copy.tiprel_1mes.values*6+df2_copy.tiprel_1mes_prev.values
    # drop other columns
    df2_copy.drop(['tiprel_1mes', 'tiprel_1mes_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')    
    
    # Combine target
    df2['target_combine'] = np.sum(df2[target_cols].values*
        np.float_power(2, np.arange(0, len(target_cols))), axis=1, 
        dtype=np.float64)
    # Load mean encoding data
    mean_encoding_result = pd.read_hdf('../input/mean_encoding_result_eda_4_21.hdf',
    'mean_encoding_result')
    # Merge with mean encoding result
    df2 = df2.merge(mean_encoding_result, on='target_combine', how='left')
    
    # number of products in the first month
    df2['n_products'] = df2[target_cols].sum(axis=1)
    
    # select (customer, product) pairs
    # not all the products goes into train set, because each customer only purchases 
    # a few products
    cpp = customer_product_pair.loc[customer_product_pair.fecha_dato==month2, 
        ['ncodpers', 'product']].copy()
    df2 = pd.merge(df2, cpp, on='ncodpers', how='right')
    
    # number of zero indexes
    zc = count_history(month1, max_lag)
    df2 = df2.join(zc, on='ncodpers')
    
    if pattern_flag:
        #print('\nStart counting patterns:')
        # count patterns of historical products
        dp = count_pattern_2(month1, max_lag)
        df2 = df2.join(dp, on='ncodpers')
        df2.loc[:, dp.columns] = df2.loc[:, dp.columns].fillna(0.0)
        
        del dp
        gc.collect()
    
    cols = df2.columns.tolist()
    cols.remove('product')
    x_train = df2.loc[:, cols].copy()
    y_train = df2.loc[:, 'product'].copy()
    weight = calculate_weight(x_train, y_train)
    
    # Save data if it does not exist
#    if not os.path.exists('../input/x_train_{}_{}.hdf'.format(month, max_lag)):
    x_train.to_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'x_train')
    y_train.to_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'y_train')
    weight.to_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'weight')

    
    return x_train, y_train, weight
    
def create_test(month='2016-06-28', max_lag=5, pattern_flag=False):
    '''Another method to create train data sets'''
    
    # First check if the data is saved.
    if os.path.exists('../input/x_train_{}_{}.hdf'.format(month, max_lag)):
        x_train = pd.read_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'x_train')
        
        return x_train
    
    month2 = month # the second month
    month1 = month_list[month_list.index(month2)-1] # the first month

    # Load second month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    df2 = df2.loc[:, cat_cols]

    # Load first month
    df1_0 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    df1 = df1_0.loc[:, cat_cols+target_cols] # keep cat_cols and target_cols
    df1_target = df1_0.loc[:, ['ncodpers']+target_cols] # keep targets

    # Merge first month product with second month customer information
    df2 = df2.merge(df1_target, on='ncodpers', how='left')
    df2.fillna(0.0, inplace=True)

    # Combination of ind_activadad_cliente
    # second month ind_actividad_cliente
    df2_copy = df2.loc[:, ['ncodpers', 'ind_actividad_cliente']].copy()
    # first month ind_actividad_cliente
    df1_copy = df1.loc[:, ['ncodpers', 'ind_actividad_cliente']].copy()
    # merge two months
    df2_copy = pd.merge(df2_copy, df1_copy, on='ncodpers', suffixes=('', '_prev'), how='left')
    # fillna
    df2_copy.fillna(2.0, inplace=True)
    # combine 
    df2_copy['ind_actvidad_client_combine'] = df2_copy.ind_actividad_cliente.values*3+df2_copy.ind_actividad_cliente_prev.values
    # drop other columns
    df2_copy.drop(['ind_actividad_cliente', 'ind_actividad_cliente_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')

    # Combination of tiprel_1mes
    # second month tiprel_1mes
    df2_copy = df2.loc[:, ['ncodpers', 'tiprel_1mes']].copy()
    # first month tiprel_1mes
    df1_copy = df1.loc[:, ['ncodpers', 'tiprel_1mes']].copy()
    # merge two months
    df2_copy = pd.merge(df2_copy, df1_copy, on='ncodpers', suffixes=('', '_prev'), how='left')
    # fillna
    df2_copy.fillna(0.0, inplace=True)
    # combine 
    df2_copy['tiprel_1mes_combine'] = df2_copy.tiprel_1mes.values*6+df2_copy.tiprel_1mes_prev.values
    # drop other columns
    df2_copy.drop(['tiprel_1mes', 'tiprel_1mes_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')    

    # Combine target
    df2['target_combine'] = np.sum(df2[target_cols].values*
        np.float_power(2, np.arange(0, len(target_cols))), axis=1, 
        dtype=np.float64)
    # Load mean encoding data
    mean_encoding_result = pd.read_hdf('../input/mean_encoding_result_eda_4_21.hdf',
    'mean_encoding_result')
    # Merge with mean encoding result
    df2 = df2.merge(mean_encoding_result, on='target_combine', how='left')

    # number of products in the first month
    df2['n_products'] = df2[target_cols].sum(axis=1)
    
    # number of history of data 
    zc = count_history(month1, max_lag)
    df2 = df2.join(zc, on='ncodpers')
    
    if pattern_flag:
        #print('\nStart counting patterns:')
        # count patterns of historical products
        dp = count_pattern_2(month1, max_lag)
        df2 = df2.join(dp, on='ncodpers')
        df2.loc[:, dp.columns] = df2.loc[:, dp.columns].fillna(0.0)
        
        del dp
        gc.collect()
    
    # Save data if it does not exist
    if not os.path.exists('../input/x_train_{}_{}.hdf'.format(month, max_lag)):
        df2.to_hdf('../input/x_train_{}_{}.hdf'.format(month, max_lag), 'x_train')
    
    return df2

################## calculate weights for train and validation sets ############
def calculate_weight(x_train, y_train):
    '''Calculate weights for xgboost'''
    x_train_ncodpers = pd.concat((x_train.loc[:, 'ncodpers'], y_train), axis=1, ignore_index=True)
    x_train_ncodpers.columns = ['ncodpers', 'n_target']
    x_train_ncodpers = pd.DataFrame(x_train_ncodpers.groupby('ncodpers')['n_target'].count())
    x_train_ncodpers['xgb_weight_1'] = 1.0/x_train_ncodpers['n_target']
    x_train_ncodpers['xgb_weight_2'] = np.exp(1.0/x_train_ncodpers['n_target']-1)
    x_train_ncodpers['xgb_weight_3'] = 1.0
    x_train_ncodpers['xgb_weight_4'] = x_train_ncodpers['n_target'].apply(lambda x: 1/sum(1/k for k in range(1, 1+x)))
    x_train_ncodpers['xgb_weight_5'] = np.exp(1.3/x_train_ncodpers['n_target']-1.3)
    x_train_ncodpers['xgb_weight_6'] = np.exp(1.2/x_train_ncodpers['n_target']-1.2)
    x_train_ncodpers['xgb_weight_7'] = np.exp(1.1/x_train_ncodpers['n_target']-1.1)
    x_train_ncodpers['xgb_weight_8'] = np.exp(0.9/x_train_ncodpers['n_target']-0.9)
    x_train_ncodpers['xgb_weight_9'] = np.exp(0.8/x_train_ncodpers['n_target']-0.8)
    x_train_ncodpers['xgb_weight_10'] = np.exp(0.7/x_train_ncodpers['n_target']-0.7)
    
    xgb_weight = pd.DataFrame(x_train.loc[:, 'ncodpers'].copy()).join(x_train_ncodpers, on='ncodpers')
    xgb_weight.drop('n_target', axis=1, inplace=True)
    xgb_weight = xgb_weight.iloc[:, 1:].copy()
    
    return xgb_weight

############################ count pattern 2 ##################################
    
def count_pattern_2(month1, max_lag):
    '''
    Encoding the pattern in one product for one customer
    (previous, this):
    (0, 0): 0
    (0, 1): 2
    (1, 0): 1
    (1, 1): 3
    '''
        
    if os.path.exists('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag)):
    
        # directly load data if it exists
        pattern_count = pd.read_hdf('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag), 'pattern_count')
        return pattern_count
        
    else:
        month_new = month_list.index(month1)+1 # the second month 
        month_end = month_list.index(month1) # the first month
        month_start = month_end-max_lag+1 # the first lagging month
                
        # Create a DataFrame containing all the previous months up to the month_index month
        df = []
        for m in range(month_start, month_end+1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        # If this is a train/val month, only keep customers with new products in the second month,
        # else, if this is a test month (2-16-06-28), we have to keep all the customers in the first month,
        # since the second month products are unknown
        if month_new<len(month_list)-1: # if this is not the last month in month_list
            # Load customer product pair
            customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
            ncodpers_list = list(set(customer_product_pair.loc[customer_product_pair.fecha_dato==month_list[month_new], 'ncodpers'].values))
        else:
            ncodpers_list = df[-1].ncodpers.unique().tolist()
            
        df = pd.concat(df, ignore_index=True)
        df = df.loc[df.ncodpers.isin(ncodpers_list), :]
        
        # count patterns for customers with at least two months records
        dt = count_changes(df)
        
        # create patterns for all customers, fillna with 0.0 if less than two months records
        pattern_count = df.loc[df.fecha_dato==month_list[month_end], ['ncodpers']]
        pattern_count.set_index('ncodpers', drop=False, inplace=True)
        pattern_count = pattern_count.join(dt)
        pattern_count.drop('ncodpers', axis=1, inplace=True)
        pattern_count.fillna(0.0, inplace=True)
           
        del dt, df, ncodpers_list
        gc.collect()
        
        # save data: pattern count that ends in month1 and count backward max_lag months
        pattern_count.to_hdf('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag), 'pattern_count')
        return pattern_count

        
###################### count history ############################
# Distance to the last positive flank
def dist_pos_flank(x):
    x = x.values[:, ::-1]
    x = np.hstack((x, np.ones((x.shape[0], 1)), np.zeros((x.shape[0], 1)) ))
    x = np.diff(x, axis=1)
    x = np.argmin(x, axis=1)
    return x

# Distance to the first positive flank
def dist_pos_flank_first(x):
    x = x.values[:, ::-1]
    x = np.hstack(( np.ones((x.shape[0], 1)), np.zeros((x.shape[0], 1)), x ))
    x = np.diff(x, axis=1)[:, ::-1]
    x = np.argmin(x, axis=1)
    return x

# Distance to the last negative flank
def dist_neg_flank(x):
    x = x.values[:, ::-1]
    x = np.hstack((x, np.zeros((x.shape[0], 1)), np.ones((x.shape[0], 1)) ))
    x = np.diff(x, axis=1)
    x = np.argmax(x, axis=1)
    return x

# Distance to the first negative flank
def dist_neg_flank_first(x):
    x = x.values[:, ::-1]
    x = np.hstack(( np.zeros((x.shape[0], 1)), np.ones((x.shape[0], 1)), x ))
    x = np.diff(x, axis=1)[:, ::-1]
    x = np.argmax(x, axis=1)
    return x

# Distance to the first 1
def dist_first_one(x):
    x = x.values
    x = np.hstack( (x, np.ones((x.shape[0], 1)) ) )
    x = x.shape[1]-2-np.argmax(x, axis=1)
    return x
    
# Distance to the last 1
def dist_last_one(x):
    x = 1-x
    return x.iloc[:, ::-1].cummin(axis=1).sum(axis=1).values

def valid_active_month(x, month1):
    '''Calculate the first valid month for each customer, x is the whole unstacked DataFrame'''
    # Valid: the records are not NAN
    # Active: buys any product
    
    # Use all the target features to determine if a customer is valid or active
    valid_active = x[target_cols].isnull().sum(axis=1, level=1)
    # First valid month: the first month that does not have any NAN, represented by index in month_list
    valid_active = pd.DataFrame(valid_active.apply(lambda u: month_list.index((u==0.0).idxmax()), axis=1))
    valid_active.columns = ['first_valid_month']
    valid_active['dist_first_valid_month'] = month_list.index(month1)-valid_active['first_valid_month']+1
    
    # First active month: the first month that buys products
    activity = x[target_cols].sum(axis=1, level=1)
    valid_active['active_percentage'] = (activity>=1.0).sum(axis=1)/activity.shape[1]
    
    return valid_active

def count_history(month1, max_lag):
    '''Statistics about historical data'''
    
    if os.path.exists('../input/history_count_{}_{}.hdf'.format(month1, max_lag)):
        df = pd.read_hdf('../input/history_count_{}_{}.hdf'.format(month1, max_lag), 
            'count_zeros')
        
        return df
    
    month_new = month_list.index(month1)+1
    month_end = month_list.index(month1)
    month_start = month_end-max_lag+1
    
    # Check if month_new is the last month
    if month_new<len(month_list)-1:
        # Customers with new products in month_new
        customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
        ncodpers_list = customer_product_pair.loc[customer_product_pair.fecha_dato==month_list[month_new], 
            'ncodpers'].unique().tolist()

    # Load data for all the lag related months
    df = []
    for m in range(month_start, month_end+1):
        df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

    # concatenate data
    df = pd.concat(df, ignore_index=True)
    
    # Drop irrelevant features, which never occur in models
    df = df.loc[:, ['fecha_dato']+cat_cols+target_cols]
    # If month_new is not the last month, keep only customers with new products 
    if month_new<len(month_list)-1:
        # select customers if this is not test set
        df = df.loc[df.ncodpers.isin(ncodpers_list), :]

    # set ncodpers and fecha_dato as index
    df.set_index(['ncodpers', 'fecha_dato'], inplace=True)

    # unstack to make month as columns, now we have MultiIndex with months and 
    # products in two levels
    df = df.unstack(level=-1, fill_value=np.nan)

    # Arithmetic /exponent weighted average of products for each (customer, product) pair 

    # Group data by features
    group0 = df.fillna(0.0).groupby(axis=1, level=0)

    # Average of products for each (customer, product) pair
    mean_product = pd.DataFrame()
    mean_product['ncodpers'] = df.index.tolist() # Note: orders of ncodpers in df and ncodpers_list are different! 
    for k in target_cols:
        mean_product[k+'_lag_mean'] = group0.get_group(k).mean(axis=1).values
    mean_product.set_index('ncodpers', inplace=True)

    # Exponent average of products for each (customer, product) pair with 
    # different decay factors 
    mean_exp_product = pd.DataFrame()
    mean_exp_product['ncodpers'] = df.index.tolist() # Note: orders of ncodpers in df and ncodpers_list are different! 
    mean_exp_alpha1 = 0.1
    mean_exp_weight1 = np.float_power(1-mean_exp_alpha1, np.arange(0, max_lag))
    mean_exp_weight1 = mean_exp_weight1[::-1]/np.sum(mean_exp_weight1)
    mean_exp_alpha2 = 0.5
    mean_exp_weight2 = np.float_power(1-mean_exp_alpha2, np.arange(0, max_lag))
    mean_exp_weight2 = mean_exp_weight2[::-1]/np.sum(mean_exp_weight2)
    for k in target_cols:
        mean_exp_product[k+'_lag_exp_mean1'] = np.average(group0.get_group(k).values, axis=1, weights=mean_exp_weight1) #group0.get_group(k).apply(np.average, axis=1, weights=mean_exp_weight1).values
        mean_exp_product[k+'_lag_exp_mean2'] = np.average(group0.get_group(k).values, axis=1, weights=mean_exp_weight2) # group0.get_group(k).apply(np.average, axis=1, weights=mean_exp_weight2).values

    mean_exp_product.set_index('ncodpers', inplace=True)

    # Distance to the last positive flank (01)
    distance_positive_flank = pd.DataFrame()
    distance_positive_flank['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_positive_flank[k+'_dist_pos_flank'] = dist_pos_flank(group0.get_group(k))
    distance_positive_flank.set_index('ncodpers', inplace=True)
    
    # Distance to the first positive flank (01)
    distance_positive_flank_first = pd.DataFrame()
    distance_positive_flank_first['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_positive_flank_first[k+'_dist_pos_flank_first'] = dist_pos_flank_first(group0.get_group(k))
    distance_positive_flank_first.set_index('ncodpers', inplace=True)

    # Distance to the last negative flank (10)
    distance_negative_flank = pd.DataFrame()
    distance_negative_flank['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_negative_flank[k+'_dist_neg_flank'] = dist_neg_flank(group0.get_group(k))
    distance_negative_flank.set_index('ncodpers', inplace=True)

    # Distance to the first negative flank (10)
    distance_negative_flank_first = pd.DataFrame()
    distance_negative_flank_first['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_negative_flank_first[k+'_dist_neg_flank_first'] = dist_neg_flank_first(group0.get_group(k))
    distance_negative_flank_first.set_index('ncodpers', inplace=True)

    # Distance to the first 1
    distance_first_one = pd.DataFrame()
    distance_first_one['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_first_one[k+'_dist_first_one'] = dist_first_one(group0.get_group(k))
    distance_first_one.set_index('ncodpers', inplace=True)

    # Count number of concatenating zeros before the second/current month
    # Or equivalently, distance to the last 1
    distance_last_one = pd.DataFrame()
    distance_last_one['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_last_one[k+'_dist_last_one'] = dist_last_one(group0.get_group(k))
    distance_last_one.set_index('ncodpers', inplace=True)
    
    # First valid month, distance to first valid month, active month percentage
    valid_active = valid_active_month(df, month1)
    
    

    history = distance_last_one.join((distance_first_one, distance_negative_flank, 
        distance_positive_flank, mean_exp_product, mean_product, 
        distance_positive_flank_first, distance_negative_flank_first, 
        valid_active))
    history.to_hdf('../input/history_count_{}_{}.hdf'.format(month1, max_lag), 'count_zeros')
    
    return history
    
############################## CV ######################################
def cv_xgb_skfrm(params, x_train, y_train, num_boost_round=3, n_splits=3, 
                           n_repeats=2, random_state=0, verbose_eval=False):
    '''
    CV of xgb using Stratified KFold Repeated Models (SKFRM)
    verbose_eval is the same as in xgb.train
    '''
    cv_results = {}
    clfs = {}
    running_time = {}
    
    eval_metric = params['eval_metric']
    
    np.random.seed(random_state)
    skf = StratifiedKFold(n_splits=n_splits, random_state=np.random.randint(10**6), shuffle=True)
    
    for m in range(n_repeats):
        for n, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
            
            start_time = time.time()
            
            # Construct DMatrix
            dtrain = xgb.DMatrix(x_train.iloc[train_index], label=y_train.iloc[train_index])
            dval = xgb.DMatrix(x_train.iloc[val_index], label=y_train.iloc[val_index])
            
            # Placeholder for evals_result
            cv_results[m, n] = {}
            params['seed'] = np.random.randint(10**6)
            clfs[m, n] = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                                   evals=[(dtrain, 'train'), (dval, 'val')], 
                                   maximize=True, early_stopping_rounds=None, 
                                   evals_result=cv_results[m, n], verbose_eval=verbose_eval)
        
            running_time[m, n] = time.time() - start_time
            
            print('Repeat {}, split {}, validate score = {:.3f}, running time = {:.3f} min'.format(m, n, 
                cv_results[m, n]['val'][eval_metric][-1], running_time[m, n]/60))
        
    # Post-process cv_results
    cv_results_final = {}
    for m in range(n_repeats):
        for n in range(n_splits):
            cv_results_final['train', m, n] = cv_results[m, n]['train'][eval_metric]
            cv_results_final['val', m, n] = cv_results[m, n]['val'][eval_metric]
    
    df = pd.DataFrame.from_dict(cv_results_final)
    df.index.name = 'iteration'
    df.columns.names = ['dataset', 'repeat', 'split']

    print('Score mean = {:.3f}, std = {:.3f}'.format(df['val'].iloc[-1].mean(), df['val'].iloc[-1].std()))
    
    return df, clfs, running_time

def plot_cv_results(df, font_size=22):
    '''plot CV results stored in df'''
    train_mean = df.mean(axis=1, level=0)['train']
    val_mean = df.mean(axis=1, level=0)['val']
    train_std = df.std(axis=1, level=0)['train']*10
    val_std = df.std(axis=1, level=0)['val']*10

    plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 22})
    plt.plot(df.index, train_mean, label='train')
    plt.plot(df.index, val_mean, label='validate')
    plt.fill_between(df.index, train_mean-train_std, train_mean+train_std, alpha=0.5)
    plt.fill_between(df.index, val_mean-val_std, val_mean+val_std, alpha=0.5)
    plt.grid()
    plt.legend()
    
    
def cv_month(param, num_rounds, month_train, month_val, n_repeat=2, random_seed=0,
                    lag_train=5, lag_val=5, weight_set=(1), verbose_eval=True):
    '''Train on one month and validate on another'''
    history = {}
    model_dict = {}

    x_train, y_train, weight_train = create_train(month_train, max_lag=lag_train, pattern_flag=True)
    x_val, y_val, weight_val = create_train(month_val, max_lag=lag_val, pattern_flag=True)

    gt_train = prep_map(x_train, y_train)
    gt_val = prep_map(x_val, y_val)

    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_val, y_val)

    ground_truth = {'train': gt_train, 'val': gt_val}
    data_hash = {'train': hash(dtrain.get_label().tostring()), 'val': hash(dval.get_label().tostring())}

    for weight_index in weight_set:
        history[weight_index] = {}
        model_dict[weight_index] = []

        dtrain.set_weight(weight_train.values[:, weight_index])
        dval.set_weight(weight_val.values[:, weight_index])
        
        print('Start weight index {}'.format(weight_index))
        print('#'*50)
        
        for n in range(n_repeat):
            history[weight_index][n] = {}
            
            param['seed'] = np.random.randint(10**6)
            
            time_start = time.time()
            print('Train with weight {}, repetition {} of {}'.format(weight_index, n, n_repeat))
            model = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, 'train'), (dval, 'val')], 
                verbose_eval=verbose_eval, feval=eval_map, evals_result=history[weight_index][n], 
                gt=ground_truth, ts=data_hash)
            model_dict[weight_index].append(model)
            time_end = time.time()
            print('Validate logloss = {:.5f}, MAP@7 = {:.5f}, time = {:.2f} min'.format(
                history[weight_index][n]['val']['mlogloss'][-1], 
                history[weight_index][n]['val']['MAP@7'][-1], (time_end-time_start)/60))
            print('-'*50)
            print('')
        print('')

    history = {(w, n, d, m): history[w][n][d][m] 
               for w in weight_set 
               for n in range(n_repeat)
               for d in ['train', 'val'] 
               for m in ['mlogloss', 'MAP@7']}
    history = pd.DataFrame(history)
    history.columns.names = ['weight_index', 'repetition', 'data_set', 'metrics']
        
    return history, model_dict
###########################################################################
    
############################## MAP #########################################
# This part implements a hacky way of evaluating MAP during xgboost training.
# This method is suitable when training on one month and validate on another 
# month, since ncodpers is the key in ground truth dictionaries.

# The procedure of MAP evaluation in train is:
# 1. create train and validation sets, their names must be x_train, y_train, 
#   and x_val, y_val
# 2. create ground truth value and index for both train and validation sets
# 3. run xgboost train
    
# Example code:
# =============================================================================
# #x_train, y_train, weight_train = create_train('2015-06-28', pattern_flag=True)
# #x_val, y_val, weight_val = create_train('2016-05-28', pattern_flag=True)
# #
# #dtrain = xgb.DMatrix(x_train, y_train, weight=weight_train)
# #dval = xgb.DMatrix(x_val, y_val, weight=weight_val)
# #
# #gt_train = prep_map(x_train, y_train)
# #gt_val = prep_map(x_val, y_val)
# #
# #param = {'objective': 'multi:softprob', 
# #         'eta': 0.05, 
# #         'max_depth': 4, 
# #         'silent': 1, 
# #         'num_class': len(target_cols),
# #         'eval_metric': 'mlogloss',
# #         'min_child_weight': 1,
# #         'subsample': 0.7,
# #         'colsample_bytree': 0.7,
# #         'seed': 0}
# #num_rounds = 200
# #
# #model = xgb.train(param, dtrain, num_rounds, 
# #   evals=[(dtrain, 'train'), (dval, 'dval')], 
# #   verbose_eval=True, feval=eval_map)
# =============================================================================

@jit
def apk(actual, predicted, k=7, default=0.0):
    if predicted.size>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if actual.size==0:
        return default

    return score / min(actual.size, k)

@jit
def eval_map(y_prob, dtrain, gt={}, ts={}):
    '''Evaluate MAP@7 for train and validation sets'''
    # Check which set is it?
    if hash(dtrain.get_label().tostring())==ts['train']:
        gti = gt['train']['index']
        gtv = gt['train']['value']
    elif hash(dtrain.get_label().tostring())==ts['val']:
        gti = gt['val']['index']
        gtv = gt['val']['value']
    
    n = len(gti)
    apks = np.zeros(n)
    y_pred = {}
    for i, (cust_id, idx) in enumerate(gti.items()):
        tmp = np.mean(y_prob[idx, :], axis=0)
        y_pred[cust_id] = np.argsort(tmp)[:-8:-1]
        apks[i] = apk(gtv[cust_id], y_pred[cust_id])
    score = np.mean(apks)

    return 'MAP@7', score
    
def prep_map(x_train, y_train):
    '''Prepare ground truth value and index for MAP evaluation, and save it.'''
    # Ground truth value: MAP needs to know the products bought by each customers
    gtv = pd.concat((pd.DataFrame(x_train.loc[:, 'ncodpers'].copy()), y_train), axis=1, ignore_index=True)
    gtv.columns = ['ncodpers', 'target']
    gtv = gtv.groupby('ncodpers')['target'].apply(lambda x: x.values).to_dict()
    # Ground truth index: MAP needs to know for each customer which rows are its corresponding data
    gti = pd.DataFrame(x_train.loc[:, 'ncodpers']).reset_index()
    gti = gti.groupby('ncodpers')['index'].apply(lambda x: x.values).to_dict()
    
    gt = {'value': gtv, 'index': gti}
    
    return gt
###########################################################################
    
############################### pickle ########################################
def save_pickle(file_name, obj ):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name ):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
###############################################################################