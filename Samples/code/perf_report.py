# -*- coding: utf-8 -*-

import pandas as pd

train_dat = pd.read_excel('../../Data/data_01/model_data/Train.xlsx',index_col=0)
valid_dat = pd.read_excel('../../Data/data_01/model_data/Valid.xlsx',index_col=0)
oot_dat = pd.read_excel('../../Data/data_01/oot_data/oot.xlsx',index_col=0)
train_X = train_dat.loc[:,~train_dat.columns.isin(['y','data_category','sample_weight'])]
valid_X = valid_dat.loc[:,~valid_dat.columns.isin(['y','data_category','sample_weight'])]
oot_X = oot_dat.loc[:,~oot_dat.columns.isin(['y','data_category','sample_weight'])]

#仅仅为了生成一个模型，可以替换成任何一个用户自己的模型
from rascpy.Tree import auto_xgb
perf_cands,params_cands,clf_cands,vars_cands = auto_xgb(train_X,train_dat.y,valid_X,valid_dat.y,train_w=train_dat.sample_weight,val_w=valid_dat.sample_weight,metric='auc',cost_time=60*5,cands_num=5,variance_level=1)
clf = clf_cands[0]

from rascpy.Report import write_performance
from rascpy.Tool import prob2score,predict_proba
train_hat = predict_proba(clf,train_X).apply(prob2score)
valid_hat = predict_proba(clf,valid_X).apply(prob2score)
oot_hat = predict_proba(clf,oot_X).apply(prob2score)
dats = {}
dats['train']=(train_dat.y,train_hat,train_dat.sample_weight) 
dats['val']=(valid_dat.y,valid_hat,valid_dat.sample_weight)
dats['oot']=(oot_dat.y,oot_hat,oot_dat.sample_weight)
_=write_performance(dats,lift=(5,10,20),wide=0.1,thin=None,filePath='./perf_report.xlsx')