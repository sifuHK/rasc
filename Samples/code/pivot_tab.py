# -*- coding: utf-8 -*-
"""
pandas DataFrame输出到EXCEL的美化演示
支持多个DataFrame排版输出到同一个Sheet
支持小数点后保留位数
支持指定列百分数显示
支持色阶显示（可设定单列色阶或跨多列色阶）
色阶可以设定最大值为红色或绿色
输出一个DataFrame或Series时可以使用 bfy_df_like_excel_one，使用方法与bfy_df_like_excel相似，可参看API：https://github.com/sifuHK/rasc/blob/main/2025.11.11/API.pdf
或 https://gitee.com/sifuHK/rasc/blob/master/2025.11.11/API_CN.pdf
"""
import pandas as pd
import numpy as np
from rascpy.Report import bfy_df_like_excel

df_grid=[]
df_rows1=[]#输出到第1行的表格放入此list中
df_rows2=[]#输出到第2行的表格放入此list中
df_rows3=[]#输出到第3行的表格放入此list中
df_rows4=[]#输出到第4行的表格放入此list中
#注：此处的行不是excel中“行”的概念
df_grid.append(df_rows1)
df_grid.append(df_rows2)
df_grid.append(df_rows3)
df_grid.append(df_rows4)

df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A1', 'B1', 'C1', 'D1'])
df1['SCORE_BIN']=['[0,100)','[100,200)','[200,300)','[300,400)','[400,500)','[500,600)','[600,700)','[700,800)','[800,900)','[900,1000]']
#将df1输出到第1行的第1个表格中
df_rows1.append({'df':df1[['SCORE_BIN','A1', 'B1', 'C1', 'D1']],'title':['notAnum'],'percent_cols':df1.columns,'color_gradient_sep':True})

df2 = pd.DataFrame(np.random.randn(2, 4), columns=['A2', 'B2', 'C2', 'D2'])
#将df2输出到第1行的第2个表格中
df_rows1.append({'df':df2,'color_gradient_cols':['A2','C2'],'title':['percent for BC','gradient_color for AC'],'percent_cols':['B2','C2']})

df3 = pd.DataFrame(np.random.randn(15, 4), columns=['A3', 'B3', 'C3', 'D3'])
#将df3输出到第2行的第1个表格中
df_rows2.append({'df':df3,'color_gradient_cols':['B3'],'title':['red_max==True'],'red_max':True})

df4 = pd.DataFrame(np.random.randn(4, 5), columns=['A4', 'B4', 'C4', 'D4', 'E4'])
#将df4输出到第2行的第2个表格中
df_rows2.append({'df':df4,'color_gradient_sep':False,'text_lum':0.4,'title':['text_lum=0.4']})

df5 = pd.DataFrame(np.random.randn(10, 15), columns=map(lambda x:'Col_%d'%x,range(1,16)))
#将df5输出到第3行的第1个表格中
df_rows3.append({'df':df5,'color_gradient_sep':True,'title':['align==False']})

df6 = pd.DataFrame({'A6':[1,2,3,4,None],'B6':[0.1,1.2,100.5,7.4,1]})
#将df6输出到第4行的第1个表格中
df_rows4.append({'df':df6,'color_gradient_sep':True,'percent_cols':['A6']})

df7 = pd.DataFrame({'A7':[1,2,3,4],'B7':[0.1,1.2,100.5,7.4]})
#将df7输出到第4行的第2个表格中
df_rows4.append({'df':df7,'not_color_gradient_cols':df7.columns,'title':['not_color_gradient_cols=df.columns']})

#两种输出方式，根据实际需要任选其一即可
#last_row_index,last_col_index分别为excel sheet中最后一行的行号和最后一列的列号（即整篇sheet的最右下角）。输出这两个位置，方便用户继续用自己的方式向sheet中添加表格
last_row_index,last_col_index = bfy_df_like_excel(df_grid,'pivot.xlsx',sheet_name='demo',default_red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2,align=True,ex_align=[2])
# or
with pd.ExcelWriter('pivot.xlsx') as writer:
    last_row_index,last_col_index = bfy_df_like_excel(df_grid,writer,sheet_name='demo',default_red_max=False,row_num=4,col_num=4,row_gap=2,col_gap=2,align=True,ex_align=[2])