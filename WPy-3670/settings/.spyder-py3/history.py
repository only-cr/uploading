# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Thu Jan  3 13:38:37 2019)---
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import math
path='E:/7 大数据业务部 数据分析组/001项目/1808 工作内容/申请模型/'
path1='E:/7 大数据业务部 数据分析组/001项目/1808 工作内容/申请模型/mxf_20180824/mxf_20180824/'
from sklearn.model_selection import train_test_split
path2='E:/7 大数据业务部 数据分析组/001项目/1808 工作内容/申请模型/7k样本建模墨菲数据/'
mf_mobile=pd.read_table(path2+'7k_mobile.txt',sep='\t',header=None)
mf_mobile.shape#[6678 rows x 3 columns]
mf_mobile.columns=['a.phone_number','gcr','name','身份证']
mf_mobile.head()

mf_mobile=mf_mobile[-(mf_mobile['身份证']=='null')]
mf_mobile.shape

mf_data1=pd.read_excel(path2+'MER14_1_IDPH_7K_蜜小蜂.xlsx', sheetname='多头信息可回溯')#{‘a’: np.float64, ‘b’: np.int32}
mf_data1.shape #(8307, 104)
mf_data1.head()
mf_data1.iloc[0,4]
mf_data1=mf_data1.replace(0,np.nan)#把0替换为空，进行饱和度统计
#bhd=np.array((-mf_data1.isnull()).astype(float).sum()/float(mf_data1.shape[0])).T
bhd=np.array((-mf_data1.isnull()).astype(float).sum()).T
pd.DataFrame(bhd).to_csv(path2+'mf_data1_饱和度.txt',header=False,index=False)
mf_data1.tail()
mf_data1.columns
mf_data1.drop_duplicates()

mf_data2=pd.read_excel(path2+'MER14_1_IDPH_7K_蜜小蜂.xlsx', sheetname='消费信息可回溯')
mf_data3=pd.read_excel(path2+'MER14_1_IDPH_7K_蜜小蜂.xlsx', sheetname='持卡信息不回溯')
mf_data2.shape #(6955, 30)
mf_data2.drop_duplicates()
mf_data2=mf_data2.replace(0,np.nan)#把0替换为空，进行饱和度统计
#bhd=np.array((-mf_data2.isnull()).astype(float).sum()/float(mf_data2.shape[0])).T
bhd=np.array((-mf_data2.isnull()).astype(float).sum()).T
pd.DataFrame(bhd).to_csv(path2+'mf_data2_饱和度.txt',header=False,index=False)
mf_data3.shape#(9356, 38)
mf_data3.columns
mf_data3.drop_duplicates()
mf_data3=mf_data3.replace(0,np.nan)#把0替换为空，进行饱和度统计
#bhd=np.array((-mf_data3.isnull()).astype(float).sum()/float(mf_data3.shape[0])).T
bhd=np.array((-mf_data3.isnull()).astype(float).sum()).T
pd.DataFrame(bhd).to_csv(path2+'mf_data3_饱和度.txt',header=False,index=False)
path2='E:/7 大数据业务部 数据分析组/001项目/1808 工作内容/申请模型/7k样本建模墨菲数据/'
mf_mobile=pd.read_table(path2+'7k_mobile.txt',sep='\t',header=None)