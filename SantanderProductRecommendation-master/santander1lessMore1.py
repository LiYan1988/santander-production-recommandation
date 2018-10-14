# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:20:50 2016

@author: celin
"""
from santanderStart3 import *

chunksize = 100000
train_acc = []
train = pd.read_csv('train_ver2.csv.zip', chunksize=chunksize,
                    dtype={"sexo":str, 
                           "ind_nuevo":str, 
                           "ult_fec_cli_1t":str,
                           "indext":str}, low_memory=False)
for i, train_chunk in enumerate(train):
    train_chunk = process_chunk_data(train_chunk)
    month_idx_tmp = find_may_june(train_chunk)
    train_tmp = train_chunk.iloc[month_idx_tmp, :]
    if train_tmp.shape[0]>0:
        train_acc.append(train_tmp)
    print('Chunk {}, train_tmp shape {}'.format(i, train_tmp.shape))
#        if i>25:
#            break
    
train_acc = pd.concat(train_acc)
train_acc = post_clean_feature(train_acc)
train_acc = post_clean_target(train_acc)
train_acc.to_csv('train_mj1.csv', index=False)

test_acc = pd.read_csv('test_ver2.csv.zip', 
                       dtype={"sexo":str, 
                       "ind_nuevo":str, 
                       "ult_fec_cli_1t":str,
                       "indext":str}, low_memory=False)
test_acc = process_chunk_data(test_acc)
test_acc = post_clean_feature(test_acc)
test_acc.to_csv('test_clean1.csv', index=False)