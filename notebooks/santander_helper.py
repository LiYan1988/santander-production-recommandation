import os

if os.name == 'nt':
    try:
        mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
        os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    except:
        pass

import numpy as np
import pandas as pd
import tqdm
import gc
import xgboost as xgb
import time
from numba import jit
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
import copy
import collections
import itertools
import re
import timeit

tqdm.tqdm.pandas()

# Define constants
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
               # 'ind_deco_fin_ult1',
               'ind_dela_fin_ult1',
               # 'ind_deme_fin_ult1',
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
# 'ind_viv_fin_ult1']

month_list = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28',
              '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28',
              '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28', '2016-06-28']


def create_monthly_data():
    '''Preprocess and clean data in each month and save to HDF files'''

    flag_exist = True
    for m in month_list:
        if not os.path.exists('../input/data_month_{}.hdf'.format(m)):
            flag_exist = False

    if flag_exist:
        return None

    fecha_dato_train = pd.read_csv('../input/train_ver2.csv', usecols=cat_cols + ['fecha_dato'])
    fecha_dato_test = pd.read_csv('../input/test_ver2.csv', usecols=cat_cols + ['fecha_dato'])

    df2 = pd.concat((fecha_dato_train, fecha_dato_test), ignore_index=True)

    del fecha_dato_train, fecha_dato_test
    gc.collect()

    # Data conversion

    mapping_dict = {
        'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},
        'sexo': {'V': 0, 'H': 1, -99: 2},
        'ind_nuevo': {'0': 0, '1': 1, -99: 2},
        'indrel': {'1': 0, '99': 1, -99: 2},
        'indrel_1mes': {-99: 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5},
        'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},
        'indresi': {-99: 0, 'S': 1, 'N': 2},
        'indext': {-99: 0, 'S': 1, 'N': 2},
        'conyuemp': {-99: 0, 'S': 1, 'N': 2},
        'indfall': {-99: 0, 'S': 1, 'N': 2},
        'tipodom': {-99: 0, '1': 1},
        'ind_actividad_cliente': {'0': 0, '1': 1, -99: 2},
        'segmento': {'02 - PARTICULARES': 0, '03 - UNIVERSITARIO': 1, '01 - TOP': 2, -99: 2},
        'pais_residencia': {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17,
                            'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73,
                            'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67,
                            'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20,
                            'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90,
                            'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118,
                            'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7,
                            'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4,
                            'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95,
                            'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66,
                            'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81,
                            'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37,
                            'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5,
                            'QA': 58, 'MZ': 27},
        'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12,
                          'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159,
                          'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51,
                          'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70,
                          'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7,
                          'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131,
                          'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101,
                          'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85,
                          'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104,
                          'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106,
                          'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130,
                          'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126,
                          'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79,
                          'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96,
                          'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136,
                          'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93,
                          'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92,
                          'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148,
                          'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20,
                          'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44,
                          'KGX': 24, 'KGC': 18, 'KGN': 11}
    }

    df2.canal_entrada.replace(mapping_dict['canal_entrada'], inplace=True)
    df2.canal_entrada.fillna(mapping_dict['canal_entrada'][-99], inplace=True)

    mapping_dict['pais_residencia'][np.nan] = mapping_dict['pais_residencia'][-99]
    df2.pais_residencia.replace(mapping_dict['pais_residencia'], inplace=True)

    df2.segmento.replace({'02 - PARTICULARES': 0, '03 - UNIVERSITARIO': 1, '01 - TOP': 2, np.nan: 2}, inplace=True)

    df2.ind_actividad_cliente.replace({0: 0, 1: 1, np.nan: 2}, inplace=True)

    df2.tipodom.replace({np.nan: 0, 1: 1}, inplace=True)

    df2.indfall.replace({np.nan: 0, 'S': 1, 'N': 2}, inplace=True)

    df2.conyuemp.replace({np.nan: 0, 'S': 1, 'N': 2}, inplace=True)

    df2.indext.replace({np.nan: 0, 'S': 1, 'N': 2}, inplace=True)

    df2.indresi.replace({np.nan: 0, 'S': 1, 'N': 2}, inplace=True)

    df2.tiprel_1mes.replace(mapping_dict['tiprel_1mes'], inplace=True)
    df2.tiprel_1mes.fillna(0, inplace=True)

    df2.indrel_1mes.replace('P', 5, inplace=True)
    df2.indrel_1mes.fillna(0, inplace=True)
    df2.indrel_1mes = pd.to_numeric(df2.indrel_1mes, errors='coerce')

    df2.indrel.replace({np.nan: 2, 1: 0, 99: 1}, inplace=True)

    df2.ind_nuevo.fillna(2, inplace=True)

    df2.sexo.fillna(2, inplace=True)
    df2.sexo.replace(mapping_dict['sexo'], inplace=True)

    df2.ind_empleado.fillna(0, inplace=True)
    df2.ind_empleado.replace(mapping_dict['ind_empleado'], inplace=True)

    df2.age = pd.to_numeric(df2.age, errors='coerce')

    mean_age = 40.
    min_age = 20.
    max_age = 90.
    range_age = max_age - min_age
    df2.age.fillna(mean_age, inplace=True)
    df2.loc[df2.age > max_age, 'age'] = max_age
    df2.loc[df2.age < min_age, 'age'] = min_age
    df2.age = (df2.age - min_age) / range_age

    df2.antiguedad = pd.to_numeric(df2.antiguedad, errors='coerce')

    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    missing_value = 0.
    df2.antiguedad.fillna(0.0, inplace=True)
    df2.loc[df2.antiguedad < min_value, 'antiguedad'] = min_value
    df2.loc[df2.antiguedad > max_value, 'antiguedad'] = max_value
    df2.antiguedad = (df2.antiguedad - min_value) / range_value

    df2.renta = pd.to_numeric(df2.renta, errors='coerce')

    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    missing_value = 101850.
    df2.renta.fillna(missing_value, inplace=True)
    df2.loc[df2.renta < min_value, 'renta'] = min_value
    df2.loc[df2.renta > max_value, 'renta'] = max_value
    df2.renta = (df2.renta - min_value) / range_value

    # Merge `cat_cols` with `target_cols`

    target = pd.read_csv('../input/train_ver2.csv', usecols=['ncodpers', 'fecha_dato'] + target_cols)

    df2 = pd.merge(df2, target, on=['fecha_dato', 'ncodpers'], how='left')
    df2.fillna(0.0, inplace=True)

    for m in month_list:
        dt = df2.loc[df2.fecha_dato == m, :].copy()
        dt.to_hdf('../input/data_month_{}.hdf'.format(m), 'data_month', complib='blosc:lz4', complevel=9, format='t')

    # Compare with previous results

    # for m in tqdm.tqdm(month_list):
    #     dt1 = pd.read_hdf('../input/data_month_{}.hdf'.format(m), 'data_month')
    #     dt2 = pd.read_hdf('../input/data_month_2_{}.hdf'.format(m), 'data_month')
    #     dt1 = dt1[dt2.columns]
    #     print((dt1!=dt2).sum().sum())


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
    c = a + b * 2
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
    a3 = (dt_changes == 3.0).astype(int)
    a3.columns = [k + '_p3' for k in a3.columns]
    a2 = (dt_changes == 2.0).astype(int)
    a2.columns = [k + '_p2' for k in a2.columns]
    a1 = (dt_changes == 1.0).astype(int)
    a1.columns = [k + '_p1' for k in a1.columns]
    a0 = (dt_changes == 0.0).astype(int)
    a0.columns = [k + '_p0' for k in a0.columns]
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
        month_start = month_end - max_lag + 1

        # Create a DataFrame containing all the previous months up to the month_index month
        df = []
        for m in range(month_start, month_end + 1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        # ncodpers_list = df[-1].ncodpers.unique().tolist()

        df = pd.concat(df, ignore_index=True)

        # count patterns for customers with at least two months records
        dt = count_changes(df)

        # create patterns for all customers, fillna with 0.0 if less than two months records
        pattern_count = df.loc[df.fecha_dato == month_list[month_end], ['ncodpers']]
        pattern_count.set_index('ncodpers', drop=False, inplace=True)
        pattern_count = pattern_count.join(dt)
        pattern_count.drop('ncodpers', axis=1, inplace=True)
        pattern_count.fillna(0.0, inplace=True)

        del dt, df  # , ncodpers_list
        gc.collect()

        pattern_count.to_hdf('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag), 'pattern_count')
        return pattern_count


# def create_train_test(month, max_lag=5, target_flag=True, pattern_flag=False):
# '''Create train and test data for month'''

# start_time = time.time()

# month2 = month # the second month
# month1 = month_list[month_list.index(month2)-1] # the first month

# # check if max_lag and month are compatible
# assert month_list.index(month2)>=max_lag, 'max_lag should be less than the index of {}, which is {}'.format(
# month2, month_list.index(month2))

# print('Loading {} data'.format(month1))
# # first/early month
# df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
# print('Loading {} data'.format(month2))
# # second/later month
# df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')

# print('Products in {}'.format(month2))
# # second month products
# df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
# df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
# # a dataframe containing the ncodpers only
# df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
# # drop ncodpers from df2_target
# df2_target.drop('ncodpers', axis=1, inplace=True)

# print('Products in {}'.format(month1))
# # first month products for all the customers in the second month
# df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
# df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
# # obtain the products purchased by all the customers in the second month
# # by joining df1_target to df2_ncodpers, NAN filled by 0.0
# df1_target = df2_ncodpers.join(df1_target, how='left')
# df1_target.fillna(0.0, inplace=True)
# df1_target.drop('ncodpers', axis=1, inplace=True)

# print('New products added in {}'.format(month2))
# # new products from the first to second month
# target = df2_target.subtract(df1_target)
# target[target<0] = 0
# target.fillna(0.0, inplace=True)

# print('Join customer features and previous month products for {}'.format(month2))
# # feature of the second month:
# # 1. customer features in the second month
# # 2. products in the first month
# x_vars = df2[cat_cols].copy() # cat_cols already includes ncodpers
# x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
# x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
# x_vars_cols = x_vars.columns.tolist()
# x_vars_cols[0] = 'sample_order' # change the name of the new column
# x_vars.columns = x_vars_cols
# x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again
# x_vars = x_vars.join(df1_target) # direct join since df1_target contains all customers in month2

# print('Concatenate this and previous months ind_activadad_cliente')
# # concatenate this and previous month values of ind_activadad_cliente
# df2_ind_actividad_cliente = df2[['ncodpers', 'ind_actividad_cliente']].copy()
# df2_ind_actividad_cliente.set_index('ncodpers', inplace=True)
# df2_ind_actividad_cliente.sort_index(inplace=True)

# df1_ind_actividad_cliente = df1[['ncodpers', 'ind_actividad_cliente']].copy()
# df1_ind_actividad_cliente.set_index('ncodpers', inplace=True)
# df1_ind_actividad_cliente.sort_index(inplace=True)

# df2_ind_actividad_cliente = df2_ind_actividad_cliente.join(df1_ind_actividad_cliente, rsuffix='_p')
# df2_ind_actividad_cliente.fillna(2.0, inplace=True)
# df2_ind_actividad_cliente['ind_actividad_client_combine'] = 3*df2_ind_actividad_cliente.ind_actividad_cliente+df2_ind_actividad_cliente.ind_actividad_cliente_p
# df2_ind_actividad_cliente = pd.DataFrame(df2_ind_actividad_cliente.iloc[:, -1])

# x_vars = pd.merge(x_vars, df2_ind_actividad_cliente, left_index=True, right_index=True, how='left')

# print('Concatenate this and previous months tiprel_1mes')
# # concatenate this and previous month value of tiprel_1mes
# df2_tiprel_1mes = df2[['ncodpers', 'tiprel_1mes']].copy()
# df2_tiprel_1mes.set_index('ncodpers', inplace=True)
# df2_tiprel_1mes.sort_index(inplace=True)

# df1_tiprel_1mes = df1[['ncodpers', 'tiprel_1mes']].copy()
# df1_tiprel_1mes.set_index('ncodpers', inplace=True)
# df1_tiprel_1mes.sort_index(inplace=True)

# df2_tiprel_1mes = df2_tiprel_1mes.join(df1_tiprel_1mes, rsuffix='_p')
# df2_tiprel_1mes.fillna(0.0, inplace=True)
# df2_tiprel_1mes['tiprel_1mes_combine'] = 6*df2_tiprel_1mes.tiprel_1mes+df2_tiprel_1mes.tiprel_1mes_p
# df2_tiprel_1mes = pd.DataFrame(df2_tiprel_1mes.iloc[:, -1])

# x_vars = pd.merge(x_vars, df2_tiprel_1mes, left_index=True, right_index=True, how='left')

# print('Combine all products for each customer')
# # combination of target columns
# x_vars['target_combine'] = np.sum(x_vars[target_cols].values*
# np.float_power(2, np.arange(0, len(target_cols))), axis=1, dtype=np.float64)
# # Load mean encoding data and merge with x_vars
# target_mean_encoding = pd.read_hdf('../input/target_mean_encoding_2.hdf', 'target_mean_encoding')
# x_vars = x_vars.join(target_mean_encoding, on='target_combine')

# # number of purchased products in the previous month
# x_vars['n_products'] = x_vars[target_cols].sum(axis=1)

# del (df1_tiprel_1mes, df2_tiprel_1mes, df1_ind_actividad_cliente,
# df2_ind_actividad_cliente, df2_target, df1_target, df2_ncodpers)
# gc.collect()

# if pattern_flag:
# print('\nStart counting patterns:')
# # count patterns of historical products
# dp = count_pattern(month1, max_lag)
# x_vars = x_vars.join(dp)
# x_vars.loc[:, dp.columns] = x_vars.loc[:, dp.columns].fillna(-1)

# del dp
# gc.collect()

# # return x_vars if target_flag is False
# if not target_flag:
# x_vars.drop('sample_order', axis=1, inplace=True) # drop sample_order
# x_vars.reset_index(inplace=True, drop=False) # add ncodpers

# end_time = time.time()
# print('Time used: {:.3f} min'.format((end_time-start_time)/60.0))

# return x_vars

# if target_flag:
# print('Prepare target')
# # prepare target/label for each added product from the first to second month
# # join target to x_vars
# x_vars_new = x_vars.join(target, rsuffix='_t')
# # set ncodpers as one column
# x_vars_new.reset_index(inplace=True, drop=False)
# x_vars.reset_index(inplace=True, drop=False)
# var_cols = x_vars.columns.tolist()

# del x_vars
# gc.collect()

# # melt
# return x_vars_new
# x_vars_new = x_vars_new.melt(id_vars=var_cols)
# # mapping from target_cols to index
# target_cols_mapping = {c+'_t': n for (n, c) in enumerate(target_cols)}
# # replace column name by index
# x_vars_new.variable.replace(target_cols_mapping, inplace=True)
# # reorder rows
# x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)
# # keep new products
# x_vars_new = x_vars_new[x_vars_new.value>0]
# # drop sample_order and value
# x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
# # keep the order of rows as in the original data set
# x_vars_new.reset_index(drop=True, inplace=True)


# var_cols.remove('sample_order')
# # variable
# x_vars = x_vars_new.loc[:, var_cols].copy()
# # target/label
# target = x_vars_new.loc[:, 'variable'].copy()

# end_time = time.time()
# print('Time used: {:.3f} min'.format((end_time-start_time)/60.0))

# return x_vars, target


# def obtain_target(month):
#     '''Create train and test data for month'''
#
#     month2 = month # the second month
#     month1 = month_list[month_list.index(month2)-1] # the first month
#
#     # first/early month
#     df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
#     # second/later month
#     df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
#
#     # second month products
#     df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
#     df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
#     # a dataframe containing the ncodpers only
#     df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
#     # drop ncodpers from df2_target
#     df2_target.drop('ncodpers', axis=1, inplace=True)
#
#     # first month products for all the customers in the second month
#     df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
#     df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
#     # obtain the products purchased by all the customers in the second month
#     # by joining df1_target to df2_ncodpers, NAN filled by 0.0
#     df1_target = df2_ncodpers.join(df1_target, how='left')
#     df1_target.fillna(0.0, inplace=True)
#     df1_target.drop('ncodpers', axis=1, inplace=True)
#
#     # new products from the first to second month
#     target = df2_target.subtract(df1_target)
#     target[target<0] = 0
#     target.fillna(0.0, inplace=True)
#
#     # feature of the second month:
#     # 1. customer features in the second month
#     # 2. products in the first month
#     x_vars = df2[['ncodpers']].copy() # cat_cols already includes ncodpers
#     x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
#     x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
#     x_vars_cols = x_vars.columns.tolist()
#     x_vars_cols[0] = 'sample_order' # change the name of the new column
#     x_vars.columns = x_vars_cols
#     x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again
#
#     # prepare target/label for each added product from the first to second month
#     # join target to x_vars
#     x_vars_new = x_vars.join(target)
#     # set ncodpers as one column
#     x_vars_new.reset_index(inplace=True, drop=False)
#     x_vars.reset_index(inplace=True, drop=False)
#
#     # melt
#     x_vars_new = x_vars_new.melt(id_vars=x_vars.columns)
#     # mapping from target_cols to index
#     target_cols_mapping = {c: n for (n, c) in enumerate(target_cols)}
#     # replace column name by index
#     x_vars_new.variable.replace(target_cols_mapping, inplace=True)
#     # reorder rows
#     x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)
#
#     # keep new products
#     x_vars_new = x_vars_new[x_vars_new.value>0]
#     # drop sample_order and value
#     x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
#     # keep the order of rows as in the original data set
#     x_vars_new.reset_index(drop=True, inplace=True)
#
#     x_vars_new.columns = ['ncodpers', 'target']
#
#     return x_vars_new

# def check_target(month1, month2, target_flag=True):
#     '''Create train and test data between month1 and month2'''
#
#     # first/early month
#     df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
#     # second/later month
#     df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
#
#     # second month products
#     df2_target = df2.loc[:, ['ncodpers']+target_cols].copy()
#     df2_target.set_index('ncodpers', inplace=True, drop=False) # initially keep ncodpers as a column and drop it later
#     # a dataframe containing the ncodpers only
#     df2_ncodpers = pd.DataFrame(df2_target.ncodpers)
#     # drop ncodpers from df2_target
#     df2_target.drop('ncodpers', axis=1, inplace=True)
#
#     # first month products for all the customers in the second month
#     df1_target = df1.loc[:, ['ncodpers']+target_cols].copy()
#     df1_target.set_index('ncodpers', inplace=True, drop=True) # do not keep ncodpers as column
#     # obtain the products purchased by all the customers in the second month
#     # by joining df1_target to df2_ncodpers, NAN filled by 0.0
#     df1_target = df2_ncodpers.join(df1_target, how='left')
#     df1_target.fillna(0.0, inplace=True)
#     df1_target.drop('ncodpers', axis=1, inplace=True)
#
#     # new products from the first to second month
#     target = df2_target.subtract(df1_target)
#     target[target<0] = 0
#     target.fillna(0.0, inplace=True)
#
#     # feature of the second month:
#     # 1. customer features in the second month
#     # 2. products in the first month
#     x_vars = df2[cat_cols].copy() # cat_cols already includes ncodpers
#     x_vars.reset_index(inplace=True, drop=True) # drop original index and make a new one
#     x_vars.reset_index(inplace=True, drop=False) # also set the new index as a column for recoding row orders
#     x_vars_cols = x_vars.columns.tolist()
#     x_vars_cols[0] = 'sample_order' # change the name of the new column
#     x_vars.columns = x_vars_cols
#     x_vars.set_index('ncodpers', drop=True, inplace=True) # set the index to ncodpers again
#     x_vars = x_vars.join(df1_target) # direct join since df1_target contains all customers in month2
#
#     # return x_vars if target_flag is False
#     if not target_flag:
#         x_vars.drop('sample_order', axis=1, inplace=True) # drop sample_order
#         x_vars.reset_index(inplace=True, drop=False) # add ncodpers
#         return x_vars #, df2_ncodpers, df1, df2, df1_target, df2_target
#
#     if target_flag:
#         # prepare target/label for each added product from the first to second month
#         # join target to x_vars
#         x_vars_new = x_vars.join(target, rsuffix='_t')
#         # set ncodpers as one column
#         x_vars_new.reset_index(inplace=True, drop=False)
#         x_vars.reset_index(inplace=True, drop=False)
#
#         # melt
#         x_vars_new = x_vars_new.melt(id_vars=x_vars.columns)
#         # mapping from target_cols to index
#         target_cols_mapping = {c+'_t': n for (n, c) in enumerate(target_cols)}
#         # replace column name by index
#         x_vars_new.variable.replace(target_cols_mapping, inplace=True)
#         # reorder rows
#         x_vars_new.sort_values(['sample_order', 'variable'], inplace=True)
#         # keep new products
#         x_vars_new = x_vars_new[x_vars_new.value>0]
#         # drop sample_order and value
#         x_vars_new.drop(['sample_order', 'value'], axis=1, inplace=True)
#         # keep the order of rows as in the original data set
#         x_vars_new.reset_index(drop=True, inplace=True)
#
#         var_cols = x_vars.columns.tolist()
#         var_cols.remove('sample_order')
#         # variable
#         x_vars = x_vars_new.loc[:, var_cols].copy()
#         # target/label
#         target = x_vars_new.loc[:, 'variable'].copy()
#
#         return x_vars, target, x_vars_new
#

def count_zeros(month1, max_lag):
    if os.path.exists('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag)):
        df = pd.read_hdf('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag),
                         'count_zeros')

        return df
    else:
        month_new = month_list.index(month1) + 1
        month_end = month_list.index(month1)
        month_start = month_end - max_lag + 1

        # Check if month_new is the last month
        if month_new < len(month_list) - 1:
            # Customers with new products in month_new
            customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
            ncodpers_list = customer_product_pair.loc[customer_product_pair.fecha_dato == month_list[month_new],
                                                      'ncodpers'].unique().tolist()

        # Load data for all the lag related months
        df = []
        for m in range(month_start, month_end + 1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        # concatenate data
        df = pd.concat(df, ignore_index=True)
        df = df.loc[:, ['ncodpers', 'fecha_dato'] + target_cols]
        if month_new < len(month_list) - 1:
            # select customers if this is not test set
            df = df.loc[df.ncodpers.isin(ncodpers_list), :]
        # set ncodpers and fecha_dato as index
        df.set_index(['ncodpers', 'fecha_dato'], inplace=True)
        # unstack to make month as columns
        df = df.unstack(level=-1, fill_value=0)

        # count number of concatenating zeros before the second/current month
        df = df.groupby(level=0, axis=1).progress_apply(lambda x: (1 - x).iloc[:, ::-1].cummin(axis=1).sum(axis=1))
        df.columns = [k + '_zc' for k in df.columns]

        gc.collect()

        df.to_hdf('../input/count_zeros_{}_{}.hdf'.format(month1, max_lag), 'count_zeros')

        return df


def create_train(month, max_lag=5, fixed_lag=6, pattern_flag=True):
    '''Another method to create train data sets
    Input:
        month: the train month, customer information is in this month, other 
            product related information is from previous months
        max_lag: maximum lag (previous months) to look backward for product
            related statistical information
        fixed_lag: maximum lag (previous months) to look backward for raw
            product history
        pattern_flag: if consider patterns (0 and 1 combinations) when 
            processing product history, it is affected by max_lag
        
        We will always try to set max_lag>=fixed_lag, i.e., max_lag should 
        always include all the history. For 2015-02-28, max_lag is set to 1,
        whereas for 2016-05-28, max_lag is 16.
        But fixed_lag is a fixed value for all months, and np.nan is filled 
        if there is no history data.
    '''

    # First check if the data is saved.
    try:
        if os.path.exists('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag)):
            x_train = pd.read_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'x_train')
            y_train = pd.read_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'y_train')
            weight = pd.read_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'weight')

            return x_train, y_train, weight
    except:
        pass

    month2 = month  # the second month
    month1 = month_list[month_list.index(month2) - 1]  # the first month

    # Load customer product pair
    try:
        customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
    except:
        customer_product_pair = calculate_customer_product_pair()

    # Load second month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    df2 = df2.loc[:, cat_cols]
    df2 = df2.loc[
        df2.ncodpers.isin(customer_product_pair.loc[customer_product_pair.fecha_dato == month2].ncodpers.unique())]

    # Load first month
    df1_0 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    df1 = df1_0.loc[:, cat_cols + target_cols]
    df1_target = df1_0.loc[:, ['ncodpers'] + target_cols]

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
    df2_copy[
        'ind_actvidad_client_combine'] = df2_copy.ind_actividad_cliente.values * 3 + df2_copy.ind_actividad_cliente_prev.values
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
    df2_copy['tiprel_1mes_combine'] = df2_copy.tiprel_1mes.values * 6 + df2_copy.tiprel_1mes_prev.values
    # drop other columns
    df2_copy.drop(['tiprel_1mes', 'tiprel_1mes_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')

    # Combine target
    df2['target_combine'] = np.sum(df2[target_cols].values *
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
    cpp = customer_product_pair.loc[customer_product_pair.fecha_dato == month2,
                                    ['ncodpers', 'product']].copy()
    df2 = pd.merge(df2, cpp, on='ncodpers', how='right')

    # number of zero indexes
    zc = count_history(month1, max_lag, fixed_lag)
    df2 = df2.join(zc, on='ncodpers')

    if pattern_flag:
        # print('\nStart counting patterns:')
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
    x_train.to_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'x_train')
    y_train.to_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'y_train')
    weight.to_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'weight')

    return x_train, y_train, weight


def create_test(month='2016-06-28', max_lag=5, fixed_lag=6, pattern_flag=True):
    '''Another method to create train data sets
    :param month:
    :param max_lag:
    :param fixed_lag:
    :param pattern_flag:
    :return:
    '''

    # First check if the data is saved.
    if os.path.exists('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag)):
        x_train = pd.read_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'x_train')

        return x_train

    month2 = month  # the second month
    month1 = month_list[month_list.index(month2) - 1]  # the first month

    # Load second month
    df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(month2), 'data_month')
    df2 = df2.loc[:, cat_cols]

    # Load first month
    df1_0 = pd.read_hdf('../input/data_month_{}.hdf'.format(month1), 'data_month')
    df1 = df1_0.loc[:, cat_cols + target_cols]  # keep cat_cols and target_cols
    df1_target = df1_0.loc[:, ['ncodpers'] + target_cols]  # keep targets

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
    df2_copy[
        'ind_actvidad_client_combine'] = df2_copy.ind_actividad_cliente.values * 3 + df2_copy.ind_actividad_cliente_prev.values
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
    df2_copy['tiprel_1mes_combine'] = df2_copy.tiprel_1mes.values * 6 + df2_copy.tiprel_1mes_prev.values
    # drop other columns
    df2_copy.drop(['tiprel_1mes', 'tiprel_1mes_prev'], axis=1, inplace=True)
    # merge result back to df2
    df2 = df2.merge(df2_copy, how='left', left_on='ncodpers', right_on='ncodpers')

    # Combine target
    df2['target_combine'] = np.sum(df2[target_cols].values *
                                   np.float_power(2, np.arange(0, len(target_cols))), axis=1,
                                   dtype=np.float64)
    # Load mean encoding data
    mean_encoding_result = calculate_customer_product_pair()
    # mean_encoding_result = pd.read_hdf('../input/mean_encoding_result_eda_4_21.hdf',
    #                                    'mean_encoding_result')
    # Merge with mean encoding result
    df2 = df2.merge(mean_encoding_result, on='target_combine', how='left')

    # number of products in the first month
    df2['n_products'] = df2[target_cols].sum(axis=1)

    # number of history of data 
    zc = count_history(month1, max_lag, fixed_lag)
    df2 = df2.join(zc, on='ncodpers')

    if pattern_flag:
        # print('\nStart counting patterns:')
        # count patterns of historical products
        dp = count_pattern_2(month1, max_lag)
        df2 = df2.join(dp, on='ncodpers')
        df2.loc[:, dp.columns] = df2.loc[:, dp.columns].fillna(0.0)

        del dp
        gc.collect()

    # Save data if it does not exist
    df2.to_hdf('../input/x_train_{}_{}_{}.hdf'.format(month, max_lag, fixed_lag), 'x_train')

    return df2


def calculate_weight(x_train, y_train):
    '''Calculate weights for xgboost'''
    x_train_ncodpers = pd.concat((x_train.loc[:, 'ncodpers'], y_train), axis=1, ignore_index=True)
    x_train_ncodpers.columns = ['ncodpers', 'n_target']
    x_train_ncodpers = pd.DataFrame(x_train_ncodpers.groupby('ncodpers')['n_target'].count())
    #    x_train_ncodpers['xgb_weight_1'] = 1.0/x_train_ncodpers['n_target']
    x_train_ncodpers['xgb_weight_2'] = np.exp(1.0 / x_train_ncodpers['n_target'] - 1)
    #    x_train_ncodpers['xgb_weight_3'] = 1.0
    #    x_train_ncodpers['xgb_weight_4'] = x_train_ncodpers['n_target'].apply(lambda x: 1/sum(1/k for k in range(1, 1+x)))
    #    x_train_ncodpers['xgb_weight_5'] = np.exp(1.3/x_train_ncodpers['n_target']-1.3)
    #    x_train_ncodpers['xgb_weight_6'] = np.exp(1.2/x_train_ncodpers['n_target']-1.2)
    #    x_train_ncodpers['xgb_weight_7'] = np.exp(1.1/x_train_ncodpers['n_target']-1.1)
    #    x_train_ncodpers['xgb_weight_8'] = np.exp(0.9/x_train_ncodpers['n_target']-0.9)
    #    x_train_ncodpers['xgb_weight_9'] = np.exp(0.8/x_train_ncodpers['n_target']-0.8)
    #    x_train_ncodpers['xgb_weight_10'] = np.exp(0.7/x_train_ncodpers['n_target']-0.7)
    #    x_train_ncodpers['xgb_weight_11'] = np.exp(0.6/x_train_ncodpers['n_target']-0.6)
    #    x_train_ncodpers['xgb_weight_12'] = np.exp(0.5/x_train_ncodpers['n_target']-0.5)
    #    x_train_ncodpers['xgb_weight_13'] = np.exp(0.4/x_train_ncodpers['n_target']-0.4)
    #    x_train_ncodpers['xgb_weight_14'] = np.exp(0.3/x_train_ncodpers['n_target']-0.3)
    #    x_train_ncodpers['xgb_weight_15'] = np.exp(0.2/x_train_ncodpers['n_target']-0.2)
    #    x_train_ncodpers['xgb_weight_16'] = np.exp(0.1/x_train_ncodpers['n_target']-0.1)

    xgb_weight = pd.DataFrame(x_train.loc[:, 'ncodpers'].copy()).join(x_train_ncodpers, on='ncodpers')
    xgb_weight.drop('n_target', axis=1, inplace=True)
    xgb_weight = xgb_weight.iloc[:, 1:].copy()

    return xgb_weight


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
        month_new = month_list.index(month1) + 1  # the second month
        month_end = month_list.index(month1)  # the first month
        month_start = month_end - max_lag + 1  # the first lagging month

        # Create a DataFrame containing all the previous months up to the month_index month
        df = []
        for m in range(month_start, month_end + 1):
            df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

        # If this is a train/val month, only keep customers with new products in the second month,
        # else, if this is a test month (2-16-06-28), we have to keep all the customers in the first month,
        # since the second month products are unknown
        if month_new < len(month_list) - 1:  # if this is not the last month in month_list
            # Load customer product pair
            customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
            ncodpers_list = list(set(customer_product_pair.loc[
                                         customer_product_pair.fecha_dato == month_list[month_new], 'ncodpers'].values))
        else:
            ncodpers_list = df[-1].ncodpers.unique().tolist()

        df = pd.concat(df, ignore_index=True)
        df = df.loc[df.ncodpers.isin(ncodpers_list), :]

        # count patterns for customers with at least two months records
        dt = count_changes(df)

        # create patterns for all customers, fillna with 0.0 if less than two months records
        pattern_count = df.loc[df.fecha_dato == month_list[month_end], ['ncodpers']]
        pattern_count.set_index('ncodpers', drop=False, inplace=True)
        pattern_count = pattern_count.join(dt)
        pattern_count.drop('ncodpers', axis=1, inplace=True)
        pattern_count.fillna(0.0, inplace=True)

        del dt, df, ncodpers_list
        gc.collect()

        # save data: pattern count that ends in month1 and count backward max_lag months
        pattern_count.to_hdf('../input/count_pattern_{}_{}.hdf'.format(month1, max_lag), 'pattern_count')
        return pattern_count


# count history
# Distance to the last positive flank
def dist_pos_flank(x):
    x = x.values[:, ::-1]
    x = np.hstack((x, np.ones((x.shape[0], 1)), np.zeros((x.shape[0], 1))))
    x = np.diff(x, axis=1)
    x = np.argmin(x, axis=1)
    return x


# Distance to the first positive flank
def dist_pos_flank_first(x):
    x = x.values[:, ::-1]
    x = np.hstack((np.ones((x.shape[0], 1)), np.zeros((x.shape[0], 1)), x))
    x = np.diff(x, axis=1)[:, ::-1]
    x = np.argmin(x, axis=1)
    return x


# Distance to the last negative flank
def dist_neg_flank(x):
    x = x.values[:, ::-1]
    x = np.hstack((x, np.zeros((x.shape[0], 1)), np.ones((x.shape[0], 1))))
    x = np.diff(x, axis=1)
    x = np.argmax(x, axis=1)
    return x


# Distance to the first negative flank
def dist_neg_flank_first(x):
    x = x.values[:, ::-1]
    x = np.hstack((np.zeros((x.shape[0], 1)), np.ones((x.shape[0], 1)), x))
    x = np.diff(x, axis=1)[:, ::-1]
    x = np.argmax(x, axis=1)
    return x


# Distance to the first 1
def dist_first_one(x):
    x = x.values
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    x = x.shape[1] - 2 - np.argmax(x, axis=1)
    return x


# Distance to the last 1
def dist_last_one(x):
    x = 1 - x
    return x.iloc[:, ::-1].cummin(axis=1).sum(axis=1).values


def valid_active_month(x, month1):
    '''Calculate the first valid month for each customer, x is the whole unstacked DataFrame'''
    # Valid: the records are not NAN
    # Active: buys any product

    # Use all the target features to determine if a customer is valid or active
    valid_active = x[target_cols].isnull().sum(axis=1, level=1)
    # First valid month: the first month that does not have any NAN, represented by index in month_list
    valid_active = pd.DataFrame(valid_active.apply(lambda u: month_list.index((u == 0.0).idxmax()), axis=1))
    valid_active.columns = ['first_valid_month']
    valid_active['dist_first_valid_month'] = month_list.index(month1) - valid_active['first_valid_month'] + 1

    # First active month: the first month that buys products
    activity = x[target_cols].sum(axis=1, level=1)
    valid_active['active_percentage'] = (activity >= 1.0).sum(axis=1) / activity.shape[1]

    return valid_active


def count_history(month1, max_lag, fixed_lag):
    '''Statistics about historical data
    max_lag should be greater than fixed_lag when previous months are available
    This will include as much history information as possible in statistics
    '''

    if os.path.exists('../input/history_count_{}_{}_{}.hdf'.format(month1, max_lag, fixed_lag)):
        df = pd.read_hdf(
            '../input/history_count_{}_{}_{}.hdf'.format(month1, max_lag, fixed_lag),
            'history_count')

        return df

    month_new = month_list.index(month1) + 1
    month_end = month_list.index(month1)
    month_start = month_end - max_lag + 1
    month_start_fixed_lag = month_end - fixed_lag + 1  # the month index for fixed_lag

    # Check if month_new is the last month
    customer_product_pair = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
    if month_new < len(month_list) - 1:
        # Customers with new products in month_new
        ncodpers_list = customer_product_pair.loc[
            customer_product_pair.fecha_dato == month_list[month_new], 'ncodpers'].unique().tolist()

    # Load data for all the lag related months
    df = []
    for m in range(month_start, month_end + 1):
        df.append(pd.read_hdf('../input/data_month_{}.hdf'.format(month_list[m]), 'data_month'))

    # concatenate data
    df = pd.concat(df, ignore_index=True)

    # Drop irrelevant features, which never occur in models
    df = df.loc[:, ['fecha_dato'] + cat_cols + target_cols]
    # If month_new is not the last month, keep only customers with new products 
    if month_new < len(month_list) - 1:
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
    mean_product['ncodpers'] = df.index.tolist()  # Note: orders of ncodpers in df and ncodpers_list are different!
    for k in target_cols:
        mean_product[k + '_lag_mean'] = group0.get_group(k).mean(axis=1).values
    mean_product.set_index('ncodpers', inplace=True)

    # Exponent average of products for each (customer, product) pair with 
    # different decay factors 
    mean_exp_product = pd.DataFrame()
    # Note: orders of ncodpers in df and ncodpers_list are different!
    mean_exp_product['ncodpers'] = df.index.tolist()
    mean_exp_alpha1 = 0.1
    mean_exp_weight1 = np.float_power(1 - mean_exp_alpha1, np.arange(0, max_lag))
    mean_exp_weight1 = mean_exp_weight1[::-1] / np.sum(mean_exp_weight1)
    mean_exp_alpha2 = 0.5
    mean_exp_weight2 = np.float_power(1 - mean_exp_alpha2, np.arange(0, max_lag))
    mean_exp_weight2 = mean_exp_weight2[::-1] / np.sum(mean_exp_weight2)
    for k in target_cols:
        mean_exp_product[k + '_lag_exp_mean1'] = \
            np.average(group0.get_group(k).values, axis=1, weights=mean_exp_weight1)
        mean_exp_product[k + '_lag_exp_mean2'] = \
            np.average(group0.get_group(k).values, axis=1, weights=mean_exp_weight2)

    mean_exp_product.set_index('ncodpers', inplace=True)

    # Distance to the last positive flank (01)
    distance_positive_flank = pd.DataFrame()
    distance_positive_flank['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_positive_flank[k + '_dist_pos_flank'] = dist_pos_flank(group0.get_group(k))
    distance_positive_flank.set_index('ncodpers', inplace=True)

    # Distance to the first positive flank (01)
    distance_positive_flank_first = pd.DataFrame()
    distance_positive_flank_first['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_positive_flank_first[k + '_dist_pos_flank_first'] = dist_pos_flank_first(group0.get_group(k))
    distance_positive_flank_first.set_index('ncodpers', inplace=True)

    # Distance to the last negative flank (10)
    distance_negative_flank = pd.DataFrame()
    distance_negative_flank['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_negative_flank[k + '_dist_neg_flank'] = dist_neg_flank(group0.get_group(k))
    distance_negative_flank.set_index('ncodpers', inplace=True)

    # Distance to the first negative flank (10)
    distance_negative_flank_first = pd.DataFrame()
    distance_negative_flank_first['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_negative_flank_first[k + '_dist_neg_flank_first'] = dist_neg_flank_first(group0.get_group(k))
    distance_negative_flank_first.set_index('ncodpers', inplace=True)

    # Distance to the first 1
    distance_first_one = pd.DataFrame()
    distance_first_one['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_first_one[k + '_dist_first_one'] = dist_first_one(group0.get_group(k))
    distance_first_one.set_index('ncodpers', inplace=True)

    # Count number of concatenating zeros before the second/current month
    # Or equivalently, distance to the last 1
    distance_last_one = pd.DataFrame()
    distance_last_one['ncodpers'] = df.index.tolist()
    for k in target_cols:
        distance_last_one[k + '_dist_last_one'] = dist_last_one(group0.get_group(k))
    distance_last_one.set_index('ncodpers', inplace=True)

    # First valid month, distance to first valid month, active month percentage
    valid_active = valid_active_month(df, month1)

    # Lags
    # Will not use this part because lags may differ, but model requires a 
    # fixed lag
    lags = df[target_cols].copy()
    # Convert MultiIndex columns to single index
    level1 = [month_list.index(month1) + 1 - month_list.index(k) for k in lags.columns.get_level_values(1)]
    level0 = lags.columns.get_level_values(0).tolist()
    lags.columns = [l0 + '_lag_' + str(l1) for l0, l1 in zip(level0, level1)]
    # Make number of lag months equals to fixed_lag
    if fixed_lag < max_lag:
        # Remove months if there are more than needed
        # Note that l1 starts from 1 and is goes to max_lag
        lags_cols = [l0 + '_lag_' + str(l1) for l0, l1 in zip(level0, level1) if l1 <= fixed_lag]
        lags = lags[lags_cols]
    elif fixed_lag > max_lag:
        # Add NAN columns if there is less 
        for c in target_cols:
            for l in range(max_lag + 1, fixed_lag + 1):
                lags[c + '_lag_' + str(l)] = np.nan
    # Sort columns first by the order in target_cols and then by lag number
    # matched_cols = [c for c in cols if re.match(r'^ind_.*_ult1_lag_[0-9]+$', c)]
    lags_cols = sorted(lags.columns.tolist(),
                       key=lambda x: (x[::-1].split('_', 1)[1][::-1],
                                      int(x[::-1].split('_', 1)[0][::-1])))
    lags = lags[lags_cols]

    # New products purchased by each customer in each month backward fixed_lag
    # months
    # Count the number of products
    # n_new_1 = customer_product_pair.groupby(['ncodpers', 'fecha_dato'])['product'].count()
    # n_new_1 = n_new_1.unstack(level=1).fillna(0.0)
    # current_month_range = list(range(month_start_fixed_lag, month_new))
    # n_new = pd.DataFrame(index=n_new_1.index)
    # for i, m in enumerate(current_month_range):
    # if m<1:
    # n_new['n_new_products_lag_'+str(fixed_lag-i)] = np.nan
    # else:
    # n_new['n_new_products_lag_'+str(fixed_lag-i)] = n_new_1[month_list[m]]

    # Target history of each product in each month
    # history_target = []
    # for t in range(len(target_cols)):
    # tmp = customer_product_pair.loc[customer_product_pair['product']==t, :].copy()
    # tmp = tmp.set_index(['ncodpers', 'fecha_dato']).unstack(level=1)
    # tmp = tmp.replace({np.nan: 0, t: 1})
    # tmp.columns = tmp.columns.levels[1]
    # target_tmp = pd.DataFrame(index=tmp.index)
    # for i, m in enumerate(current_month_range):
    # if i == fixed_lag-3:
    # break
    # if m<1:
    # target_tmp[target_cols[t]+'_target_lag_'+str(fixed_lag-i)] = np.nan
    # else:
    # target_tmp[target_cols[t]+'_target_lag_'+str(fixed_lag-i)] = tmp[month_list[m]]
    # history_target.append(target_tmp)
    # history_target = pd.concat(history_target, axis=1)

    history = distance_last_one.join((distance_first_one, distance_negative_flank,
                                      distance_positive_flank, mean_exp_product, mean_product,
                                      distance_positive_flank_first, distance_negative_flank_first,
                                      valid_active, lags))
    history.to_hdf('../input/history_count_{}_{}_{}.hdf'.format(month1, max_lag, fixed_lag), 'history_count')

    return history


# CV related functions
# def cv_xgb_skfrm(params, x_train, y_train, num_boost_round=3, n_splits=3, 
# n_repeats=2, random_state=0, verbose_eval=False):
# '''
# CV of xgb using Stratified KFold Repeated Models (SKFRM)
# verbose_eval is the same as in xgb.train
# '''
# cv_results = {}
# clfs = {}
# running_time = {}

# eval_metric = params['eval_metric']

# np.random.seed(random_state)
# skf = StratifiedKFold(n_splits=n_splits, random_state=np.random.randint(10**6), shuffle=True)

# for m in range(n_repeats):
# for n, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):

# start_time = time.time()

# # Construct DMatrix
# dtrain = xgb.DMatrix(x_train.iloc[train_index], label=y_train.iloc[train_index])
# dval = xgb.DMatrix(x_train.iloc[val_index], label=y_train.iloc[val_index])

# # Placeholder for evals_result
# cv_results[m, n] = {}
# params['seed'] = np.random.randint(10**6)
# clfs[m, n] = xgb.train(params, dtrain, num_boost_round=num_boost_round,
# evals=[(dtrain, 'train'), (dval, 'val')],
# maximize=True, early_stopping_rounds=None,
# evals_result=cv_results[m, n], verbose_eval=verbose_eval)

# running_time[m, n] = time.time() - start_time

# print('Repeat {}, split {}, validate score = {:.3f}, running time = {:.3f} min'.format(m, n,
# cv_results[m, n]['val'][eval_metric][-1], running_time[m, n]/60))

# # Post-process cv_results
# cv_results_final = {}
# for m in range(n_repeats):
# for n in range(n_splits):
# cv_results_final['train', m, n] = cv_results[m, n]['train'][eval_metric]
# cv_results_final['val', m, n] = cv_results[m, n]['val'][eval_metric]

# df = pd.DataFrame.from_dict(cv_results_final)
# df.index.name = 'iteration'
# df.columns.names = ['dataset', 'repeat', 'split']

# print('Score mean = {:.3f}, std = {:.3f}'.format(df['val'].iloc[-1].mean(), df['val'].iloc[-1].std()))

# return df, clfs, running_time

# def plot_cv_results(df, eval_metric, font_size=22):
# '''plot CV results stored in df'''
# train_mean = df.mean(axis=1, level=0)['train']
# val_mean = df.mean(axis=1, level=0)['val']
# train_std = df.std(axis=1, level=0)['train']*10
# val_std = df.std(axis=1, level=0)['val']*10

# plt.figure(figsize=(16, 9))
# plt.rcParams.update({'font.size': 22})
# plt.plot(df.index, train_mean, label='train')
# plt.plot(df.index, val_mean, label='validate')
# plt.fill_between(df.index, train_mean-train_std, train_mean+train_std, alpha=0.5)
# plt.fill_between(df.index, val_mean-val_std, val_mean+val_std, alpha=0.5)
# plt.grid()
# plt.legend()


# def cv_month(param, num_rounds, month_train, month_val, n_repeat=2, random_seed=0,
# lag_train=5, lag_val=5, weight_set=(1), verbose_eval=True):
# '''Train on one month and validate on another with different weights. Now it is not used anymore.'''
# history = {}
# model_dict = {}

# x_train, y_train, weight_train = create_train(month_train, max_lag=lag_train, pattern_flag=True)
# x_val, y_val, weight_val = create_train(month_val, max_lag=lag_val, pattern_flag=True)

# gt_train = prep_map(x_train, y_train)
# gt_val = prep_map(x_val, y_val)

# dtrain = xgb.DMatrix(x_train, y_train)
# dval = xgb.DMatrix(x_val, y_val)

# ground_truth = {'train': gt_train, 'val': gt_val}
# data_len = {'train': len(dtrain.get_label()), 'val': len(dval.get_label())}

# for weight_index in weight_set:
# history[weight_index] = {}
# model_dict[weight_index] = []

# dtrain.set_weight(weight_train.values[:, weight_index])
# dval.set_weight(weight_val.values[:, weight_index])

# print('Start weight index {}'.format(weight_index))
# print('#'*50)

# for n in range(n_repeat):
# history[weight_index][n] = {}

# param['seed'] = np.random.randint(10**6)

# time_start = time.time()
# print('Train with weight {}, repetition {} of {}'.format(weight_index, n, n_repeat))
# model = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, 'train'), (dval, 'val')],
# verbose_eval=verbose_eval, feval=eval_map, evals_result=history[weight_index][n],
# gt=ground_truth, ts=data_len)
# model_dict[weight_index].append(model)
# time_end = time.time()
# print('Validate logloss = {:.5f}, MAP@7 = {:.5f}, time = {:.2f} min'.format(
# history[weight_index][n]['val'][eval_metric][-1],
# history[weight_index][n]['val']['MAP@7'][-1], (time_end-time_start)/60))
# print('-'*50)
# print('')
# print('')

# history = {(w, n, d, m): history[w][n][d][m]
# for w in weight_set
# for n in range(n_repeat)
# for d in ['train', 'val']
# for m in [eval_metric, 'MAP@7']}
# history = pd.DataFrame(history)
# history.columns.names = ['weight_index', 'repetition', 'data_set', 'metrics']

# return history, model_dict

def train_test_month(param, num_rounds, month_train, month_val, sub_name=None,
                     month_test='2016-06-28', n_repeat=2,
                     random_seed=0, max_lag=5,
                     fixed_lag=6, verbose_eval=True,
                     n_features=250, eval_train_flag=True):
    '''Train on one month and validate on another'''
    history = {}
    model_dict = {}
    y_pred = []

    # eval_metric in param
    eval_metric = param['eval_metric']

    # Train data
    x_train, y_train, w_train = create_train(month_train, max_lag=max_lag, fixed_lag=fixed_lag, pattern_flag=True)
    # Validation data
    x_val, y_val, w_val = create_train(month_val, max_lag=max_lag, fixed_lag=fixed_lag, pattern_flag=True)
    # Test data
    x_test = create_test(month_test, max_lag=max_lag, fixed_lag=fixed_lag, pattern_flag=True)

    # Select features
    if n_features is not None:
        try:
            fi = pd.read_csv('feature_importance.csv', )
            fi = fi.iloc[:min(n_features, fi.shape[0]), 0].values.tolist()
            fi = list(set(fi) | set(target_cols) | set(cat_cols))
            x_train = x_train[fi]
            x_val = x_val[fi]
            x_test = x_test[fi]
        except:
            pass

    gt_train = prep_map(x_train, y_train)
    gt_val = prep_map(x_val, y_val)

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
    dtest = xgb.DMatrix(x_test)

    ground_truth = {'train': gt_train, 'val': gt_val}
    # If evaluate train set or not? 
    # Disable can save time, and I do not care too much about train evaluation.
    if eval_train_flag:
        data_len = {'train': len(dtrain.get_label()), 'val': len(dval.get_label())}
    else:
        data_len = {'train': 0, 'val': len(dval.get_label())}

    np.random.seed(random_seed)
    for n in range(n_repeat):
        history[n] = {}

        param['seed'] = np.random.randint(10 ** 6)

        time_start = time.time()
        print('Train, validate, and predict, repetition {} of {}'.format(n, n_repeat))
        model = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, 'train'), (dval, 'val')],
                          verbose_eval=verbose_eval, feval=eval_map, evals_result=history[n],
                          gt=ground_truth, ts=data_len)
        model_dict[n] = model
        y_tmp = model.predict(dtest)
        y_tmp[x_test[target_cols] == 1] = 0
        y_pred.append(y_tmp)

        time_end = time.time()
        print('Validate {} = {:.5f}, MAP@7 = {:.5f}, time = {:.2f} min'.format(
            eval_metric,
            history[n]['val'][eval_metric][-1],
            history[n]['val']['MAP@7'][-1], (time_end - time_start) / 60))
        print('-' * 50)
        print('')

    # Process training history
    history = {(n, d, m): history[n][d][m]
               for n in range(n_repeat)
               for d in ['train', 'val']
               for m in [eval_metric, 'MAP@7']}
    history = pd.DataFrame(history)
    history.columns.names = ['repetition', 'data_set', 'metrics']

    # Process test result
    y_pred = np.array(y_pred)
    y_sub = np.mean(y_pred, axis=0)
    y_sub = np.argsort(y_sub, axis=1)
    y_sub = np.fliplr(y_sub)[:, :7]
    # Prepare submission
    test_id = x_test.loc[:, 'ncodpers'].values
    y_sub = [' '.join([target_cols[k] for k in pred]) for pred in y_sub]
    y_sub = pd.DataFrame({'ncodpers': test_id, 'added_products': y_sub})
    if sub_name is not None:
        y_sub.to_csv(sub_name, compression='gzip', index=False)

    return history, model_dict, y_pred, y_sub


def cv_all_month(params, train, val=None, n_features=350, num_boost_round=3,
                 n_repeats=2, random_state=0, verbose_eval=False):
    '''
    CV of xgb using Stratified KFold Repeated Models (SKFRM)
    verbose_eval is the same as in xgb.train
    '''
    cv_results = {}
    clfs = {}
    running_time = {}

    eval_metric = params['eval_metric']

    x_train = train['x']
    y_train = train['y']
    w_train = train['w']

    if val is not None:
        x_val = val['x']
        y_val = val['y']
        w_val = val['w']

    # Select features
    if n_features is not None:
        fi = pd.read_csv('feature_importance.csv', )
        fi = fi.iloc[:min(n_features, fi.shape[0]), 0].values.tolist()
        fi = list(set(fi) | set(target_cols) | set(cat_cols))
        x_train = x_train[fi]
        if val is not None:
            x_val = x_val[fi]

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)

    if val is not None:
        dval = xgb.DMatrix(x_val, label=y_val, weight=w_val)
        eval_list = [(dtrain, 'train'), (dval, 'val')]
        gt_val = prep_map(x_val, y_val)
        # Ground truth parameters for evaluation of MAP@7
        # We do not evaluate on the train data
        gt_val = prep_map(x_val, y_val)
        ground_truth = {'train': None, 'val': gt_val}
        data_len = {'train': None, 'val': len(dval.get_label())}
    else:
        eval_list = [(dtrain, 'train')]
        ground_truth = {'train': None, 'val': None}
        data_len = {'train': None, 'val': None}

    np.random.seed(random_state)

    for m in range(n_repeats):

        start_time = time.time()

        # Placeholder for evals_result
        cv_results[m] = {}
        params['seed'] = np.random.randint(10 ** 6)
        clfs[m] = xgb.train(params, dtrain,
                            num_boost_round=num_boost_round,
                            evals=eval_list,
                            evals_result=cv_results[m],
                            verbose_eval=verbose_eval, feval=eval_map,
                            gt=ground_truth, ts=data_len)

        running_time[m] = time.time() - start_time

        if val is not None:
            print('Repeat {}, validate score = {:.3f}, running time = {:.3f} min'.format(m,
                                                                                         cv_results[m]['val'][
                                                                                             eval_metric][-1],
                                                                                         running_time[m] / 60))
        else:
            print('Repeat {}, running time = {:.3f} min'.format(m, running_time[m] / 60))

    # Post-process cv_results
    if val is not None:
        cv_results_final = {}
        for m in range(n_repeats):
            cv_results_final['train', m] = cv_results[m]['train'][eval_metric]
            cv_results_final['val', m] = cv_results[m]['val'][eval_metric]

        df = pd.DataFrame.from_dict(cv_results_final)
        df.index.name = 'iteration'
        df.columns.names = ['dataset', 'repeat']

        print('Score mean = {:.3f}, std = {:.3f}'.format(df['val'].iloc[-1].mean(), df['val'].iloc[-1].std()))

        return df, clfs, running_time
    else:
        return clfs, running_time


def predict_all_month(model_dict, x_test, sub_name, n_features=350, n_trees=0):
    '''
    Predict on test set with multliple models in dict, data from all month.
    This function is used together with cv_all_month
    :rtype: object
    :param model_dict:
    :param x_test:
    :param sub_name:
    :param n_features:
    :return:
    '''

    y_pred = []

    if n_features is not None:
        fi = pd.read_csv('feature_importance.csv', )
        fi = fi.iloc[:min(n_features, fi.shape[0]), 0].values.tolist()
        fi = list(set(fi) | set(target_cols) | set(cat_cols))
        x_test = x_test[fi]
    dtest = xgb.DMatrix(x_test)

    for n in model_dict.keys():
        y_tmp = model_dict[n].predict(dtest, ntree_limit=n_trees)
        y_tmp[x_test[target_cols] == 1] = 0
        y_pred.append(y_tmp)

    # Process test result
    y_pred = np.array(y_pred)
    y_sub = np.mean(y_pred, axis=0)
    y_sub = np.argsort(y_sub, axis=1)
    y_sub = np.fliplr(y_sub)[:, :7]
    # Prepare submission
    test_id = x_test.loc[:, 'ncodpers'].values
    y_sub = [' '.join([target_cols[k] for k in pred]) for pred in y_sub]
    y_sub = pd.DataFrame({'ncodpers': test_id, 'added_products': y_sub})
    y_sub.to_csv(sub_name, compression='gzip', index=False)

    return y_pred, y_sub


# ===========================================================================
#
#  MAP ##
# This part implements a hacky way of evaluating MAP during xgboost training.
# This method is suitable when training on one month and validate on another
# month, since ncodpers is the key in ground truth dictionaries.
#
# The procedure of MAP evaluation in train is:
# 1. create train and validation sets, their names must be x_train, y_train,
#   and x_val, y_val
# 2. create ground truth value and index for both train and validation sets
# 3. run xgboost train
#
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
    if actual.size == 0:
        return default

    if predicted.size > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(actual.size, k)


@jit
def eval_map(y_prob, dtrain, gt={}, ts={}):
    '''
    Evaluate MAP@7 for train and validation sets---
    '''
    # Check which set is it?
    if len(dtrain.get_label()) == ts['train']:
        glist = gt['train']
    elif len(dtrain.get_label()) == ts['val']:
        glist = gt['val']
    else:
        return 'MAP@7', 0

    n = len(glist)
    score = np.zeros(n)
    for i in range(n):
        tmp = np.mean(y_prob[glist[i][1], :], axis=0)
        tmp = np.argsort(tmp)[:-8:-1]
        score[i] = apk(glist[i][0], tmp)
    score = np.mean(score)

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

    gt = np.array([[gtv[k], gti[k]] for k in gtv.keys()])

    return gt


# pickle
def save_pickle(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def calculate_customer_product_pair():
    if os.path.exists('../input/customer_product_pair.hdf'):
        target = pd.read_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
        return target

    target = []
    for m1, m2 in tqdm.tqdm(list(zip(month_list[:-2], month_list[1:-1]))):
        df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(m1)).loc[:, ['ncodpers'] + target_cols]
        df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(m2)).loc[:, ['ncodpers'] + target_cols]

        df1.set_index('ncodpers', inplace=True)
        df2.set_index('ncodpers', inplace=True)

        dt = df2.join(df1, how='left', lsuffix='_2', rsuffix='_1').fillna(0.0)
        dt = pd.DataFrame(dt.values[:, :19] - dt.values[:, 19:], index=df2.index, columns=target_cols)
        dt.reset_index(inplace=True)
        dt = dt.melt(id_vars='ncodpers')
        dt['variable'] = dt['variable'].map({k: i for i, k in enumerate(target_cols)})
        dt = dt.loc[dt['value'] == 1.0]
        dt.drop('value', axis=1, inplace=True)
        dt.reset_index(drop=True, inplace=True)
        dt['fecha_dato'] = m2

        target.append(dt)

    target = pd.concat(target, ignore_index=True)
    target.columns = ['ncodpers', 'product', 'fecha_dato']

    target.to_hdf('../input/customer_product_pair.hdf', 'customer_product_pair')
    return target

def calculate_customer_product_pair_binary():
    '''
    Prepare customer product pairs for binary classification
    '''
    if os.path.exists('../input/customer_product_pair_binary.hdf'):
        target = pd.read_hdf('../input/customer_product_pair_binary.hdf', 'customer_product_pair')
        return target

    target = []
    for m1, m2 in tqdm.tqdm(list(zip(month_list[:-2], month_list[1:-1]))):
        df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(m1)).loc[:, ['ncodpers'] + target_cols]
        df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(m2)).loc[:, ['ncodpers'] + target_cols]
    
        df1.set_index('ncodpers', inplace=True)
        df2.set_index('ncodpers', inplace=True)
    
        dt = df2.join(df1, how='left', lsuffix='_2', rsuffix='_1').fillna(0.0)
        dt = pd.DataFrame(dt.values[:, :19] - dt.values[:, 19:], index=df2.index, columns=target_cols)
        dt.reset_index(inplace=True)
        dt = dt.melt(id_vars='ncodpers')
        dt['variable'] = dt['variable'].map({k: i for i, k in enumerate(target_cols)})
        
        dt2 = df2.copy()
        dt2.reset_index(inplace=True)
        dt2 = dt2.melt(id_vars='ncodpers')
        dt2['variable'] = dt2['variable'].map({k: i for i, k in enumerate(target_cols)})
        
        dt = dt.loc[dt2['value'] == 0.0]
        dt.drop('value', axis=1, inplace=True)
        dt.reset_index(drop=True, inplace=True)
        dt['fecha_dato'] = m2
    
        target.append(dt)

    target = pd.concat(target, ignore_index=True)
    target.columns = ['ncodpers', 'product', 'fecha_dato']

    target.to_hdf('../input/customer_product_pair_binary.hdf', 'customer_product_pair')
    return target


def mean_encoding_month_product():
    '''
    Encode previous month products with mean of buying each product in the next month
    '''
    # column names for newly purchased products

    if os.path.exists('../input/mean_encoding_result_eda_4_21.hdf'):
        mean_encoding_result = pd.read_hdf('../input/mean_encoding_result_eda_4_21.hdf',
                                           'mean_encoding_result')
        return mean_encoding_result

    new_cols = [k + '_new' for k in target_cols]
    new_cols_map = {k + '_new': n for n, k in enumerate(target_cols)}
    # ordered dict containing new products in each month, key is the first month
    # du = collections.OrderedDict()
    # new products information for mean encoding
    mean_encoding = []
    # (customer, product) pair for new products in every month
    customer_product_pair = {}
    for m1, m2 in tqdm.tqdm(list(zip(month_list[:-2], month_list[1:-1]))):
        # load first month data
        df1 = pd.read_hdf('../input/data_month_{}.hdf'.format(m1), 'data_month')
        # load second month data
        df2 = pd.read_hdf('../input/data_month_{}.hdf'.format(m2), 'data_month')

        # only keep products information
        df1 = df1[['ncodpers'] + target_cols]
        df2 = df2[['ncodpers'] + target_cols]

        # calculate new products
        # merge first and second month products
        x = df2.merge(df1, on=['ncodpers'], how='left', suffixes=('_l', ''))
        x.fillna(0.0, inplace=True)
        # calculate difference 
        x = x.iloc[:, 1:20].values - x.iloc[:, 20:].values
        x = pd.DataFrame(x, index=df2.ncodpers, columns=new_cols)
        # remove negative elements
        x[x < 0] = 0
        # only keep customers with new products
        x = x[x.sum(axis=1) > 0]

        # keep copy of customers with new products
        new_product = x.copy()

        # obtain (customer, product) pairs
        x = pd.DataFrame(x.stack())
        x.reset_index(inplace=True)
        x.columns = ['ncodpers', 'product', 'indicator']
        x.loc[:, 'product'] = x.loc[:, 'product'].map(new_cols_map)
        # only keep (customer, product) pairs for new products
        x = x.loc[x.indicator > 0]
        x.drop('indicator', axis=1, inplace=True)
        x.reset_index(inplace=True, drop=True)
        # a list of customers with new products
        ncodpers_new_product = x.ncodpers.unique()
        customer_product_pair[m2] = x
        #     x.to_hdf('../input/customer_product_pair_{}.hdf'.format(m2),
        #         'customer_product_pair')

        # only keep customers with new products in the second month
        # df2 can be used to create train data
        df2 = df2.loc[df2.ncodpers.isin(ncodpers_new_product)]

        # prepare mean encoding 
        # product pattern in the first month
        df1['target_combine'] = np.sum(df1.values[:, 1:] *
                                       np.float_power(2, np.arange(0, len(target_cols))),
                                       axis=1, dtype=np.float64)
        df1.drop(target_cols, axis=1, inplace=True)
        # number and indicator of new products
        new_product['n_new'] = new_product.loc[:, new_cols].sum(axis=1)
        new_product['ind_new'] = new_product.loc[:, new_cols].max(axis=1)
        # join with the first month 
        df1 = df1.join(new_product, on='ncodpers', how='left')
        df1.fillna(0.0, inplace=True)
        df1.drop('ncodpers', axis=1, inplace=True)

        # add results to list
        mean_encoding.append(df1)

    # concatenate all data
    mean_encoding = pd.concat(mean_encoding, ignore_index=True)
    # calculate mean vaues 
    mean_encoding_result = mean_encoding.groupby('target_combine').mean()
    # save mean encoding result
    mean_encoding_result.to_hdf('../input/mean_encoding_result_eda_4_21.hdf',
                                'mean_encoding_result')

    return mean_encoding_result


def calculate_feature_importance(param,
                                 num_rounds=3,
                                 n_repeat=2,
                                 month_train='2015-06-28',
                                 month_val='2016-05-28',
                                 random_seed=42,
                                 fi_name=None):
    """
    Calculate feature importance using xgboost
    Many parameters of train_test_month are set to the default values
    """
    history, model_dict, y_pred, y_sub = \
        train_test_month(param, num_rounds, month_train, month_val,
                         sub_name=None, n_repeat=n_repeat, random_seed=random_seed, n_features=None)

    fi = pd.DataFrame({i: model_dict[i].get_score(importance_type='gain') for i in range(n_repeat)})
    fi['mean'] = fi.iloc[:, :n_repeat].mean(axis=1)
    fi['std'] = fi.iloc[:, :n_repeat].std(axis=1)
    fi.sort_values(by=['mean'], inplace=True, ascending=False)
    if fi_name is not None:
        fi.to_csv(fi_name)

    return fi


def plot_history_val(history):
    '''Plot CV history of validation'''

    n_repeat = np.unique(history.columns.get_level_values(0).values).shape[0]
    history_val = history.xs(axis=1, level=[1, 2], key=['val', 'MAP@7']).copy()
    history_val['mean'] = history_val.iloc[:, :n_repeat].mean(axis=1)
    history_val['std'] = history_val.iloc[:, :n_repeat].std(axis=1)

    plt.figure(figsize=(16, 9))
    plt.plot(history_val['mean'])
    plt.fill_between(
        history_val.index,
        history_val['mean'] + history_val['std'],
        history_val['mean'] - history_val['std'],
        alpha=0.3)
    plt.grid()

    return None


def plot_feature_importance(model_dict):
    '''Plot feature importance of train_test_month in bar plot.'''
    n_repeat = len(model_dict)
    # Feature importance
    fi = pd.DataFrame({i: model_dict[i].get_score(importance_type='gain') for i in range(n_repeat)})
    fi['mean'] = fi.iloc[:, :n_repeat].mean(axis=1)
    fi['std'] = fi.iloc[:, :n_repeat].std(axis=1)
    fi.sort_values(by=['mean'], inplace=True, ascending=False)

    plt.rcParams.update({'figure.figsize': '16, 240'})
    plt.rcParams.update({'font.size': '22'})
    fig, ax = plt.subplots()
    ax.barh(fi.index, fi['mean'].values, log=True, xerr=fi['std'].values)
    ax.grid()
    ax.tick_params(labelbottom=True, labeltop=True)
    ax.set_ylim(fi.shape[0], -0.5)

    plt.figure(figsize=(16, 9))
    plt.plot(fi.values[:, n_repeat])
    plt.grid()
    plt.yscale('log')

    return fi

