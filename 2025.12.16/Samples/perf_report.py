# -*- coding: utf-8 -*-

import pandas as pd

train_dat = pd.read_excel('../../Data/data_01/model_data/Train.xlsx')
valid_dat = pd.read_excel('../../Data/data_01/model_data/Valid.xlsx')
oot_dat = pd.read_excel('../../Data/data_01/oot_data/oot.xlsx')
train_X = train_dat.loc[:,~train_dat.columns.isin(['y','data_category'])]
valid_X = valid_dat.loc[:,valid_dat.columns.isin(['y','data_category'])]

# only for the purpose of generating a model, it can be replaced with any user's own model.
from rascpy.Tree import auto_xgb
perf_cands,params_cands,clf_cands,vars_cands = auto_xgb(train_X,train_dat.y,valid_X,valid_dat.y,train_w=train_dat.sample_weight,val_w=valid_dat.sample_weight,metric='auc',cost_time=60*5,cands_num=5,variance_level=1)
clf = clf_cands[0]

from rasc.Report import write_performance
from rasc.Tool import prob2score,predict_proba
train_hat = predict_proba(clf,train_X).apply(prob2score)
valid_hat = predict_proba(clf,valid_X).apply(prob2score)
dats = {}
dats['train']=(train_dat.y,train_hat,train_dat.sample_weight) 
dats['val']=(valid_dat.y,valid_hat,valid_dat.sample_weight)
# (datas,target_label=None,cut_data_name=None,wide=0.05,thin=0.01,thin_head=10,lift=None,score_reverse=True,writer=None,sheet_name=lan['Performance of the model'],filePath=None):
_=write_performance(dats,lift=(5,10,20),wide=0.1,thin=None,filePath='./perf_report.xlsx')



