# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:30:37 2018

@author: lyaa
"""

import os
if os.name=='nt':
    try:
        mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
        os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    except:
        pass

import csv
import datetime
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
import tqdm

#%%
df_1 = pd.read_hdf('../input/data_all.hdf', 'train_test_converted')

#%%
df_1.to_csv('../input/df_1.csv', index=False)

#%%
target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
               'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
               'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
               'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1',
               'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 
               'ind_recibo_ult1']
target_cols = sorted(target_cols)

cat_cols = ['ind_empleado',
                'sexo', 'ind_nuevo', 
                'indrel', 'indrel_1mes', 
                'tiprel_1mes', 'indresi', 
                'indext', 'conyuemp', 
                'indfall', 'tipodom', 
                'ind_actividad_cliente', 'segmento', 
                'pais_residencia', 'canal_entrada', 
                'age', 'renta', 'antiguedad']
cat_cols = sorted(feature_cols)

#%%
month_cols = df_1.fecha_dato.unique()
for n in range(df_1.shape[0]):
    row = df_1.iloc[n, :]
    print(row[target_cols])
    if n==10:
        break

#%%
    
def processData(df, cust_dict, test=True):
    x_vars_list = []
    y_vars_list = []
    for n in tqdm.tqdm(range(df.shape[0])):
        row = df.iloc[n, :]
        if row['fecha_dato'] not in [120, 151, 486, 517]:
            continue
        
        cust_id = int(row['ncodpers'])
        if row['fecha_dato'] in [120, 486]:
            target_list = row[target_cols]
            
        x_vars = []
        x_vars.append(row[cat_cols])
        
        if row['fecha_dato'] == 517 and test: # 2016-06-28
            prev_target_list = cust_dict.get(cust_id, [0]*22)
            x_vars_list.append(x_vars + prev_target_list)
        elif row['fecha_dato'] == 486 and not test:
            prev_target_list = cust_dict.get(cust_id, [0]*22)
            target_list = row[target_cols]
            new_products = [max(x1-x2, 0) for (x1, x2) in 
                            zip(target_list, prev_target_list)]
            if sum(new_products)>0:
                for ind, prod in enumerate(new_products):
                    if prod>0:
                        assert len(prev_target_list)==22
                        x_vars_list.append(x_vars+prev_target_list)
                        y_vars_list.append(ind)
                        
    return x_vars_list, y_vars_list, cust_dict
        
#%%
x_vars_list, y_vars_list, cust_dict = processData(df_1, {}, test=False)