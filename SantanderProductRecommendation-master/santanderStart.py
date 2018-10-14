# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:00:01 2016

@author: celin

to-do: 1. read_data and save_data can read and write .pkl.zip files
"""

import pandas as pd
import numpy as np
import gc
import csv
import datetime
from operator import sub
import pickle
from multiprocessing import Queue, Process
import multiprocessing as mp
import os
import zipfile

import xgboost as xgb
from sklearn import preprocessing, ensemble, model_selection, metrics

#mapping_dict = {
#'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
#'sexo'          : {'V':0, 'H':1, -99:2},
#'ind_nuevo'     : {'0':0, '1':1, -99:2},
#'indrel'        : {'1':0, '99':1, -99:2},
#'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
#'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
#'indresi'       : {-99:0, 'S':1, 'N':2},
#'indext'        : {-99:0, 'S':1, 'N':2},
#'conyuemp'      : {-99:0, 'S':1, 'N':2},
#'indfall'       : {-99:0, 'S':1, 'N':2},
#'tipodom'       : {-99:0, '1':1},
#'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
#'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},
#'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
#'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
#}
#cat_cols = list(mapping_dict.keys())
#
#target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
#target_cols = target_cols[2:]

def getTarget(row):
	tlist = []
	for col in target_cols:
		if row[col].strip() in ['', 'NA']:
			target = 0
		else:
			target = int(float(row[col]))
		tlist.append(target)
	return tlist

def getIndex(row, col):
	val = row[col].strip()
	if val not in ['','NA']:
		ind = mapping_dict[col][val]
	else:
		ind = mapping_dict[col][-99]
	return ind

def getAge(row):
	mean_age = 40.
	min_age = 20.
	max_age = 90.
	range_age = max_age - min_age
	age = row['age'].strip()
	if age == 'NA' or age == '':
		age = mean_age
	else:
		age = float(age)
		if age < min_age:
			age = min_age
		elif age > max_age:
			age = max_age
	return round( (age - min_age) / range_age, 4)

def getCustSeniority(row):
	min_value = 0.
	max_value = 256.
	range_value = max_value - min_value
	missing_value = 0.
	cust_seniority = row['antiguedad'].strip()
	if cust_seniority == 'NA' or cust_seniority == '':
		cust_seniority = missing_value
	else:
		cust_seniority = float(cust_seniority)
		if cust_seniority < min_value:
			cust_seniority = min_value
		elif cust_seniority > max_value:
			cust_seniority = max_value
	return round((cust_seniority-min_value) / range_value, 4)

def getRent(row):
	min_value = 0.
	max_value = 1500000.
	range_value = max_value - min_value
	missing_value = 101850.
	rent = row['renta'].strip()
	if rent == 'NA' or rent == '':
		rent = missing_value
	else:
		rent = float(rent)
		if rent < min_value:
			rent = min_value
		elif rent > max_value:
			rent = max_value
	return round((rent-min_value) / range_value, 6)

def processData(in_file_name, cust_dict):
	x_vars_list = []
	y_vars_list = []
	for row in csv.DictReader(in_file_name):
		# use only the four months as specified by breakfastpirate #
		if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:
			continue

		cust_id = int(row['ncodpers'])
		if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:	
			target_list = getTarget(row)
			cust_dict[cust_id] =  target_list[:]
			continue

		x_vars = []
		for col in cat_cols:
			x_vars.append( getIndex(row, col) )
		x_vars.append( getAge(row) )
		x_vars.append( getCustSeniority(row) )
		x_vars.append( getRent(row) )

		if row['fecha_dato'] == '2016-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*22)
			x_vars_list.append(x_vars + prev_target_list)
		elif row['fecha_dato'] == '2015-06-28':
			prev_target_list = cust_dict.get(cust_id, [0]*22)
			target_list = getTarget(row)
			new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
			if sum(new_products) > 0:
				for ind, prod in enumerate(new_products):
					if prod>0:
						assert len(prev_target_list) == 22
						x_vars_list.append(x_vars+prev_target_list)
						y_vars_list.append(ind)

	return x_vars_list, y_vars_list, cust_dict
			
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


def process_chunk_data(df):
    """Clean chunk data
    """
    # date
    df.fecha_dato = pd.to_datetime(df.fecha_dato, format='%Y-%m-%d') 
    df.fecha_alta = pd.to_datetime(df.fecha_alta, format='%Y-%m-%d') 
    df['month'] = pd.DatetimeIndex(df.fecha_dato).month
    df['age'] = pd.to_numeric(df.age, errors='coerce')
    
    # ind_nuevo
    df.loc[df.ind_nuevo.isnull(), 'ind_nuevo'] = 1
           
    # antiguedad, seniority (in months)
    df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
    df.loc[df.antiguedad.isnull(), "antiguedad"] = 0
    df.antiguedad = df.antiguedad.astype(int)
    df.loc[df.antiguedad<0, "antiguedad"] = 0

    # indrel, 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
    df.loc[df.indrel.isnull(),"indrel"] = 1
           
    # tipodom, nomprov, cod_prov
    df.drop(["tipodom","nomprov"],axis=1,inplace=True)
    df.loc[df.cod_prov.isnull(), 'cod_prov'] = -1

    return df
    
def find_may_june(df):
    """Find row indexes whose fecha_dato is in May or June
    """
    a1 = df.fecha_dato==pd.Timestamp('2015-05-28 00:00:00')
    a2 = df.fecha_dato==pd.Timestamp('2015-06-28 00:00:00')
    a3 = df.fecha_dato==pd.Timestamp('2016-05-28 00:00:00')
    a4 = df.fecha_dato==pd.Timestamp('2016-06-28 00:00:00')
    month_idx = a1.values | a2.values | a3.values | a4.values
    month_idx = np.where(month_idx==True)[0]
    
    return month_idx
    
def status_change(x):
    diffs = x.diff().fillna(0)# first occurrence will be considered Maintained, 
    #which is a little lazy. A better way would be to check if 
    #the earliest date was the same as the earliest we have in the dataset
    #and consider those separately. Entries with earliest dates later than that have 
    #joined and should be labeled as "Added"
    label = ["Added" if i==1 \
         else "Dropped" if i==-1 \
         else "Maintained" for i in diffs]
    return label

    
def post_clean_feature(df):
    """Clean data after loading all interesting rows, 
    because the cleaning needs information from all the rows
    """
    # age
    df.loc[df.age<18, 'age'] = df.loc[(df.age>18) & (df.age<30), 'age'].mean(skipna=True)
    df.loc[df.age>100, 'age'] = df.loc[(df.age<100) & (df.age>30), 'age'].mean(skipna=True)
    df.age.fillna(df.age.mean(), inplace=True)
    df.age = df.age.astype(int)
    
    # fecha_alta
    dates = df.loc[:,"fecha_alta"].sort_values().reset_index()
    median_date = int(np.median(dates.index.values))
    df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
           
    # renta
    df.renta = pd.to_numeric(df.renta, errors='coerce')
    incomes = df.loc[df.renta.notnull(),:].groupby("cod_prov").agg({"renta":{"MedianIncome":np.median}})
    incomes.sort_values(by=("renta","MedianIncome"),inplace=True)
    incomes.reset_index(inplace=True)
    incomes.cod_prov = incomes.cod_prov.astype("category", categories=[i for i in df.cod_prov.unique()],ordered=False)
    incomes.head()
    
    grouped = df.groupby("cod_prov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
    new_incomes = pd.merge(df,grouped,how="inner",on="cod_prov").loc[:, ["cod_prov","renta_y"]]
    new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("cod_prov")
    df.sort_values("cod_prov",inplace=True)
    df = df.reset_index()
    new_incomes = new_incomes.reset_index()
    
    df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
    df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
    df.sort_values(by="fecha_dato",inplace=True)
    
    return df
    
def post_clean_target(df):
    """Clean target
    """
    # ind_nomina_ult1
    df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
           
    # 
    string_data = df.select_dtypes(include=["object"])
    missing_columns = [col for col in string_data if string_data[col].isnull().any()]
    
    df.loc[df.indfall.isnull(),"indfall"] = "N"
    df.loc[df.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
    df.tiprel_1mes = df.tiprel_1mes.astype("category")
    
    # As suggested by @StephenSmith
    map_dict = { 1.0  : "1",
                "1.0" : "1",
                "1"   : "1",
                "3.0" : "3",
                "P"   : "P",
                3.0   : "3",
                2.0   : "2",
                "3"   : "3",
                "2.0" : "2",
                "4.0" : "4",
                "4"   : "4",
                "2"   : "2"}
    
    df.indrel_1mes.fillna("P",inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))
    df.indrel_1mes = df.indrel_1mes.astype("category")

    unknown_cols = [col for col in missing_columns 
                    if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
    for col in unknown_cols:
        df.loc[df[col].isnull(),col] = "UNKNOWN"

    target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
    for col in target_cols:
        df[col] = df[col].astype(int)

    unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).\
                                 sort_values()).reset_index(drop=True)
    unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
    unique_months["month_next_id"] = 1 + unique_months["month_id"]
    unique_months.rename(columns={0:"fecha_dato"},inplace=True)
    df = pd.merge(df,unique_months,on="fecha_dato")

    print('changing status')

    df.loc[:, target_cols] = df.loc[:, [i for i in target_cols]+["ncodpers"]].\
        groupby("ncodpers").transform(lambda x: x.diff().fillna(0))
#    first_month = df.groupby('ncodpers')[['ncodpers', 'fecha_dato']].min()
#    new_customer_june = first_month[first_month['fecha_dato']==
#                                    pd.Timestamp('2015-06-28 00:00:00'), 
#                                    'ncodpers'].values.tolist()

    print('melting')

    df = pd.melt(df, id_vars=[col for col in df.columns 
                              if col not in target_cols],
                value_vars= [col for col in target_cols])
#    df = df.loc[(df.value==1),:]
    
    return df
    
def save_data(file_name, data):
    """File name must ends with .pkl or .pkl.zip
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data
    
def xgb_randomcv(reg, params, x_train, y_train, x_test,
                 n_iters=10, cv=3, random_state=0):
    np.random.seed(random_state)
    seed1 = np.random.randint(10000)
    seed2 = np.random.randint(10000)    
    param_list = list(model_selection.ParameterSampler(params, n_iters, seed1))
    y_test_pred_list = []
    y_train_pred_list = []
    obj_list = []
    ntree_list = []
    for p in param_list:
        reg.set_params(**p)
        y_test_pred_, y_train_pred_, obj_, ntree_ = \
            cv_predict_xgb(reg, x_train, y_train, x_test, cv, seed2)
        y_test_pred_list.append(y_test_pred_)
        y_train_pred_list.append(y_train_pred_)
        obj_list.append(obj_)
        ntree_list.append(ntree_)
        
        
    return y_test_pred_list,y_train_pred_list,obj_list,ntree_list,param_list
    
def xgb_gridcv(reg, params, x_train, y_train, x_test, cv=3, random_state=0, esr=10):
    np.random.seed(random_state)
    seed2 = np.random.randint(10000)    
    y_test_pred_list = []
    y_train_pred_list = []
    obj_list = []
    ntree_list = []
    if type(params) == dict:
        params = [params]
    for p in params:
        reg.set_params(**p)
        y_test_pred_, y_train_pred_, obj_, ntree_ = \
            cv_predict_xgb(reg, x_train, y_train, x_test, cv, seed2, esr)
        y_test_pred_list.append(y_test_pred_)
        y_train_pred_list.append(y_train_pred_)
        obj_list.append(obj_)
        ntree_list.append(ntree_)
          
    return y_test_pred_list,y_train_pred_list,obj_list,ntree_list,params
    
def cv_predict_xgb(clfxgb, x_train, y_train, x_test, cv=3, random_state=0, esr=10):
    kf = model_selection.StratifiedKFold(n_splits=cv, shuffle=True, 
                               random_state=random_state)
    mlogloss = []
    ntree = []
    y_test_pred = []
    n_classes = np.unique(y_train).shape[0]
    y_train_pred = np.zeros((y_train.shape[0],n_classes))
    for train_index, test_index in kf.split(x_train, y_train):
        x_train1 = x_train.iloc[train_index]
        y_train1 = y_train.iloc[train_index]
        x_train2 = x_train.iloc[test_index]
        y_train2 = y_train.iloc[test_index]
        clfxgb.fit(x_train1, y_train1, eval_metric='mlogloss', 
                  eval_set=[(x_train1, y_train1), (x_train2, y_train2)],
                  early_stopping_rounds=esr, verbose=True)        
        best_ntree_limit = clfxgb.best_ntree_limit
        best_ntree_limit2 = int(best_ntree_limit*cv/(cv-1))
        ntree.append(best_ntree_limit)
        # validation set
        y_pred2 = clfxgb.predict_proba(x_train2, ntree_limit=best_ntree_limit)
        mlogloss.append(metrics.log_loss(y_train2, y_pred2, 
                        labels=list(range(n_classes))))
        y_train_pred[test_index,:] = y_pred2
        # test set
        preds = clfxgb.predict_proba(x_test, ntree_limit=best_ntree_limit2)
        y_test_pred.append(preds)
    mlogloss = np.array(mlogloss)
    print('Mean: ', mlogloss.mean(), ' std: ', mlogloss.std())
    y_test_pred = np.mean(y_test_pred, axis=0)
    
    return y_test_pred, y_train_pred, mlogloss, ntree
    
def cv_predict_xgb_repeat(reg, x_train, y_train, x_test, 
                          cv=3, random_state=0, rep=10, esr=10):
    y_test_pred = []
    y_train_pred = []
    np.random.seed(random_state)
    
    for i in range(rep):
        tmp_test, tmp_train, _, _ = cv_predict_xgb(reg, x_train, y_train, 
                                                   x_test, cv, 
                                                   np.random.randint(1000), 
                                                   esr=esr)
        y_test_pred.append(tmp_test)
        y_train_pred.append(tmp_train)
        
    y_test_pred_mean = np.mean(y_test_pred, axis=0)
    y_train_pred_mean = np.mean(y_train_pred, axis=0)
    
    return y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred
    
def add_lag_features(df):
    """Add lag features. df is the dataframe with all the records of one 
    customer
    """
    
    target_cols = list(df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values)
    lag_cols = target_cols+['age', 'segmento']
    target_cols_diff = [i+'_diff' for i in target_cols]
    
    for lag in range(1, 6):
        lag_tmp = [i+str(lag) for i in lag_cols]
        df[lag_tmp] = df[lag_cols]
        df[lag_tmp] = df[lag_tmp].shift(lag).fillna(-2)
            
    df[target_cols_diff] = df[target_cols]
    df[target_cols_diff] = df[target_cols_diff].diff().fillna(0)
    if df.fecha_dato.min()=='2015-06-28':
        df[df.fecha_dato=='2015-06-28', target_cols_diff] = \
            df[df.fecha_dato=='2015-06-28', target_cols]
            
    dg = df.drop(target_cols, axis=1)
    dg = pd.melt(dg, id_vars=[col for col in dg.columns 
                              if col not in target_cols_diff], 
                              value_vars=target_cols_diff)
    
    if dg.fecha_dato.min()=='2015-06-28':
        dg_train = dg.loc[dg.fecha_dato=='2015-06-28',:]
    elif dg.fecha_dato.min()<'2015-06-28':
        dg_train = dg.loc[(dg.value==1)&(dg.fecha_dato=='2015-06-28'),:]
    else:
        dg_train = None
    dg_test = dg.loc[dg.fecha_dato=='2016-06-28',:]
    
    result = (df, dg, dg_train, dg_test)
        
    return result
    
def mp_lag(mp_lag_in):
    df_ncodpers, ncodpers_list = mp_lag_in
    def worker(inlist, out_q):
        """The worker function is invoked in a process. inlist is a list of 
        input item indexes. The results are placed in a dictionary that is 
        pushed to a queue.
        """
        outdict = {}
        for i in inlist:
            outdict[i] = add_lag_features(
                df_ncodpers.get_group(ncodpers_list[i]).copy())
            if i%1000==0:
                print('{}/{}'.format(i,len(ncodpers_list)))
        out_q.put(outdict)
        
    # Each process will get 'chunksize' input items and a queue to put his out
    # dict into
    out_q = Queue()
    nprocs = os.cpu_count()
    chunksize = int(np.ceil(len(ncodpers_list)/float(nprocs)))
    print(chunksize)
    procs = []
    
    for i in range(nprocs):
        p = Process(target=worker, 
                    args=(list(range(chunksize*i,chunksize*(i+1))),out_q))
        procs.append(p)
        p.start()
        
    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())
        
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
        
    return resultdict

def mp_lag2(mp_lag_in):
    df_ncodpers, ncodpers_list = mp_lag_in

    pool = mp.Pool(processes=os.cpu_count()-1, maxtasksperchild=1000)
    tasks = [pool.apply_async(add_lag_features, 
    	(df_ncodpers.get_group(i).copy(),) ) for i in ncodpers_list]
    for f in tasks:
    	print(f.get())
    pool.close()
    pool.join()

        
    return resultdict

def save_submission(filename, y_pred, le, x_test, test_may):
#    target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
#    'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
#    'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
#    'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
#    'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
#    'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
#    'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
#    'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
#    target_cols = target_cols[2:]
#    target_cols = np.array(target_cols)
    
    test_id = x_test.ncodpers.copy()
    y_pred_df = pd.DataFrame(y_pred, index=test_id, 
                             columns=le.inverse_transform(list(range(22))))
    y_pred_df.reset_index(drop=False, inplace=True)
    y_pred_df = y_pred_df.subtract(test_may, axis='index')
    y_pred_df.drop(['ncodpers'], axis=1, inplace=True)
    y_pred = y_pred_df.as_matrix()
    preds = np.argsort(y_pred, axis=1)
    preds = np.fliplr(preds)[:,:7]
    final_preds = le.inverse_transform(preds)
    final_preds = [" ".join(x) for x in final_preds]
    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
    out_df.to_csv(filename, index=False)
    
def cv_predict(reg, x_train, y_train, x_test, cv=3, random_state=0):
    kf = model_selection.StratifiedKFold(n_splits=cv, shuffle=True, 
                               random_state=random_state)
    obj = []
    y_test_pred = []
    y_train_pred = np.zeros((y_train.shape[0],))
    for train_index, test_index in kf.split(x_train, y_train):
        x_train1 = x_train.iloc[train_index]
        y_train1 = y_train.iloc[train_index]
        x_train2 = x_train.iloc[test_index]
        y_train2 = y_train.iloc[test_index]
        reg.fit(x_train1, y_train1)
        y_pred2 = reg.predict_proba(x_train2)
        y_train_pred[test_index] = y_pred2
        obj.append(metrics.log_loss(y_train2, y_pred2, labels=list(range(22))))
        y_test_pred.append(reg.predict(x_test))
        
    obj = np.array(obj)
    print('Mean: ', obj.mean(), ' std: ', obj.std())
    y_test_pred = np.mean(y_test_pred, axis=0)
    
    return y_test_pred, y_train_pred, obj

def obj_opt(w, y_true, y_pred):
    y_pred = np.average(y_pred, axis=0, weights=w)
    return metrics.log_loss(y_true, y_pred)
    
def optimize_weights(initial_weights, y_train_preds, y_test_preds, y_train):
    """Optimize weights
    """
    ndim = y_train_preds.shape[1]
#    initial_weights = 1.0/ndim*np.ones((ndim, ))
    bounds = [(0, 1) for i in range(ndim)]
    constraints = {'type': 'eq', 'fun': lambda w: 1-sum(w)}
    obj = partial(obj_opt, y_true=y_train, y_pred=y_train_pred)
    res = optimize.minimize(obj, initial_weights,
        bounds=bounds, constraints=constraints)
    final_weights = res.x
    weight_optimize_res = res
    y_val = np.dot(y_test_preds, final_weights)
    
    return y_val, final_weights, weight_optimize_res, 
    
#if __name__=='__main__':
