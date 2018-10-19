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
import copy
import time

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
        np.float_power(2, np.arange(-10, len(target_cols)-10)), axis=1, dtype=np.float64)
    # Load mean encoding data and merge with x_vars
    target_mean_encoding = pd.read_hdf('../input/target_mean_encoding.hdf', 'target_mean_encoding')
    x_vars = x_vars.join(target_mean_encoding, on='target_combine')

    # number of purchased products in the previous month
    x_vars['n_products'] = x_vars[target_cols].sum(axis=1)

    if pattern_flag:
        print('\nStart counting patterns:')
        # count patterns of historical products
        dp = count_pattern(month1, max_lag)
        x_vars = x_vars.join(dp)
        x_vars.loc[:, dp.columns] = x_vars.loc[:, dp.columns].fillna(-1)
    
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