# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import pandas as pd

limit_rows = 1700000 # 7000000
limit_people = 12000 #120000
df = pd.read_csv('train_ver2.csv.zip', dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str}, 
                                                    nrows=limit_rows)
#test_file = pd.read_csv('test_ver2.csv.zip')
#sample_submission = pd.read_csv('sample_submission.csv.zip')

unique_ids = pd.Series(df['ncodpers'].unique())
unique_id = unique_ids.sample(n=limit_people)
df = df[df['ncodpers'].isin(unique_id)]
df.describe()
df = df[df.ind_nuevo.notnull()]

#%%
df.fecha_dato = pd.to_datetime(df.fecha_dato, format='%Y-%m-%d') # row date
df.fecha_alta = pd.to_datetime(df.fecha_alta, format='%Y-%m-%d') # join date
df.fecha_dato.unique()

df['month'] = pd.DatetimeIndex(df.fecha_dato).month
df['age'] = pd.to_numeric(df.age, errors='coerce')

#%% age
#df.isnull.any() # too slow
df.loc[df.age<18, 'age'] = df.loc[(df.age>18) & (df.age<30), 'age'].mean(skipna=True)
df.loc[df.age>100, 'age'] = df.loc[(df.age<100) & (df.age>30), 'age'].mean(skipna=True)
df.age.fillna(df.age.mean(), inplace=True)
df.age = df.age.astype(int)

#%% ind_nuevo
df['ind_nuevo'].isnull().sum()

df.loc[df.ind_nuevo.isnull(), 'ind_nuevo'] = -1
df.ind_nuevo = df.ind_nuevo.astype(float)
# if a custmer is new, ind_nuevo is always new, otherwise ind_nuevo is always 0
a = df[['ind_nuevo', 'ncodpers']].groupby('ncodpers', sort=False).mean()
b = df[['ind_nuevo', 'ncodpers']].groupby('ncodpers', sort=False).size()

months_active = df.loc[df.ind_nuevo==-1, :].groupby('ncodpers', sort=False).size()
df.loc[months_active[months_active>1], 'ind_nuevo'] = 0 # occur more than once
df.loc[months_active[months_active==1], 'ind_nuevo'] = 1 # only once

#%% antiguedad
df.antiguedad = df.antiguedad.astype(int)
df.loc[df.antiguedad.isnull(),"antiguedad"] = df.antiguedad.min()
df.loc[df.antiguedad<0, "antiguedad"] = 0

#%% fecha_alta
dates=df.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
df.loc[df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

#%% indrel
df.loc[df.indrel.isnull(),"indrel"] = 1

#%% tipodom, nomprov, cod_prov
df.drop(["tipodom","nomprov"],axis=1,inplace=True)
df.loc[df.cod_prov.isnull(), 'cod_prov'] = -1

#%% renta
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

#%% ind_nomina_ult1
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

#%% other products
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

unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    df.loc[df[col].isnull(),col] = "UNKNOWN"
    
#%% check if data is clean
a = []
cols = df.columns
for i, c in enumerate(cols):
    print('{}: {}'.format(c, df[c].isnull().any()))
    a.append(df[c].isnull().any())
    
#%% convert target columns to number
target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in target_cols:
    df[col] = df[col].astype(int)
    
unique_months = pd.DataFrame(pd.Series(df.fecha_dato.unique()).sort_values()).reset_index(drop=True)
unique_months["month_id"] = pd.Series(range(1,1+unique_months.size)) 
unique_months["month_next_id"] = 1 + unique_months["month_id"]
unique_months.rename(columns={0:"fecha_dato"},inplace=True)
df = pd.merge(df,unique_months,on="fecha_dato")
df.drop(['month'], axis=1, inplace=True)

#%%
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
    
df.loc[:, target_cols] = df.loc[:, [i for i in target_cols]+["ncodpers"]].groupby("ncodpers").transform(status_change)

#%%
df = pd.melt(df, id_vars   = [col for col in df.columns if col not in target_cols],
            value_vars= [col for col in target_cols])
df = df.loc[df.value=="Added",:]
df.shape

#%% find May and June
month_idx = (df.fecha_dato == pd.Timestamp('2015-01-28 00:00:00'))|\
            (df.fecha_dato == pd.Timestamp('2015-02-28 00:00:00'))
month_idx = month_idx[month_idx]