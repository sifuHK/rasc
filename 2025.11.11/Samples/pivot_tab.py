# -*- coding: utf-8 -*-
"""
Pandas DataFrame output to Excel beautification demo
Supports multiple DataFrames arranged and output to the same sheet
Supports decimal place retention
Supports percentage display for specified columns
Supports color scale display (can set single-column or multi-column color scales)
Color scales can be set with maximum value as red or green
When outputting a single DataFrame or Series, you can use bfy_df_like_excel_one, usage is similar to bfy_df_like_excel. Refer to API: https://github.com/sifuHK/rasc/blob/main/2025.11.11/API.pdf
or https://gitee.com/sifuHK/rasc/blob/master/2025.11.11/API_CN.pdf
"""
import pandas as pd
import numpy as np
from rascpy.Report import bfy_df_like_excel

df_grid=[]
df_rows1=[]# Tables output to row 1 are placed in this list
df_rows2=[]# Tables output to row 2 are placed in this list
df_rows3=[]# Tables output to row 3 are placed in this list
df_rows4=[]# Tables output to row 4 are placed in this list
# Note: The "rows" here are not the same as "rows" in Excel
df_grid.append(df_rows1)
df_grid.append(df_rows2)
df_grid.append(df_rows3)
df_grid.append(df_rows4)

df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A1', 'B1', 'C1', 'D1'])
df1['SCORE_BIN']=['[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800)','[800,900)','[900,1000]']
# Output df1 to the first table in row 1
df_rows1.append({'df':df1[['SCORE_BIN','A1', 'B1', 'C1', 'D1']],'title':['notAnum'],'percent_cols':df1.columns,'color_gradient_sep':True})

df2 = pd.DataFrame(np.random.randn(2, 4), columns=['A2', 'B2', 'C2', 'D2'])
# Output df2 to the second table in row 1
df_rows1.append({'df':df2,'color_gradient_cols':['A2','C2'],'title':['percent for BC','gradient_color for AC'],'percent_cols':['B2','C2']})

df3 = pd.DataFrame(np.random.randn(15, 4), columns=['A3', 'B3', 'C3', 'D3'])
# Output df3 to the first table in row 2
df_rows2.append({'df':df3,'color_gradient_cols':['B3'],'title':['red_max==True'],'red_max':True})

df4 = pd.DataFrame(np.random.randn(4, 5), columns=['A4', 'B4', 'C4', 'D4', 'E4'])
# Output df4 to the second table in row 2
df_rows2.append({'df':df4,'color_gradient_sep':False,'text_lum':0.4,'title':['text_lum=0.4']})

df5 = pd.DataFrame(np.random.randn(10, 15), columns=map(lambda x:'Col_%d'%x,range(1,16)))
# Output df5 to the first table in row 3
df_rows3.append({'df':df5,'color_gradient_sep':True,'title':['align==False']})

df6 = pd.DataFrame({'A6':[1,2,3,4,None],'B6':[0.1,1.2,100.5,7.4,1]})
# Output df6 to the first table in row 4
df_rows4.append({'df':df6,'color_gradient_sep':True,'percent_cols':['A6']})

df7 = pd.DataFrame({'A7':[1,2,3,4],'B7':[0.1,1.2,100.5,7.4]})
# Output df7 to the second table in row 4
df_rows4.append({'df':df7,'not_color_gradient_cols':df7.columns,'title':['not_color_gradient_cols=df.columns']})

# Two output methods, choose either one as needed
# last_row_index, last_col_index are the row number of the last row and column number of the last column in the Excel sheet (i.e., the bottom-right corner of the entire sheet). Output these positions to facilitate users to continue adding tables to the sheet in their own way
last_row_index,last_col_index = bfy_df_like_excel(df_grid,'pivot.xlsx',sheet_name='demo',default_red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2,align=True,ex_align=[2])
# or
with pd.ExcelWriter('pivot.xlsx') as writer:
    last_row_index,last_col_index = bfy_df_like_excel(df_grid,writer,sheet_name='demo',default_red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2,align=True,ex_align=[2])