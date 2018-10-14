# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 21:31:48 2016

@author: celin
"""

from santanderStart3 import *

#train = pd.read_csv('train_ver2.csv.zip', dtype={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str, "indext":str}, 
#                    low_memory=False, nrows=2000000)
#train = process_chunk_data(train)
#train = post_clean_feature(train)
#
#target_cols = train.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values.tolist()

#%% 
# sample train
df = pd.read_csv('train_ver2.csv.zip', nrows=4000000, low_memory=False)
unique_ids = pd.Series(df['ncodpers'].unique())
unique_id = unique_ids.sample(n=120000)
df = df[df.ncodpers.isin(unique_id)]
print(df.fecha_dato.unique())
df = process_chunk_data(df)
df = post_clean_feature(df)

#%% 
month_current = '2015-07-28'
month_lag = '2015-02-28'
target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values

df_ncodpers_group = df.groupby('ncodpers')
ncodpers = df.ncodpers.unique()
a = {}
for i, custid in enumerate(ncodpers):
    tmp = df_ncodpers_group.get_group(custid)
    if tmp.shape[0]<6:
        a[i] = tmp
        print(i)