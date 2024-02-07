# %%
import pandas as pd
import numpy as np
import pymysql
import os

from datetime import datetime
from importlib import reload
from WindPy import w
import cvxpy as cp

def getCSI500Data(conn,start_dt,end_dt):
    query = f"""
    SELECT s_con_windcode, trade_dt, weight
    FROM aindexcsi500weight
    WHERE trade_dt >= '{str(start_dt)}' AND trade_dt <= '{str(end_dt)}'
    ORDER BY s_con_windcode ASC
    """
    data = pd.read_sql_query(query, conn)
    return data


def getCSI1000Data(conn,start_dt,end_dt):
    query = f"""
    SELECT s_con_windcode, trade_dt, weight
    FROM aindexcsi1000weight
    WHERE trade_dt >= '{str(start_dt)}' AND trade_dt <= '{str(end_dt)}'
    ORDER BY s_con_windcode ASC
    """
    data = pd.read_sql_query(query, conn)
    return data


def getHS300Data(conn, start_dt, end_dt):
    '''
       Function description : 提取沪深300成分股信息
       param : conn - 数据连接
       return :
       Usage : 

    '''
    query = f"""
    SELECT s_con_windcode, trade_dt, i_weight as weight
    FROM aindexhs300weight
    WHERE trade_dt >= '{str(start_dt)}' AND trade_dt <= '{str(end_dt)}'
    ORDER BY s_con_windcode ASC
    """
    data = pd.read_sql_query(query, conn)
    return data

# %%
conn = pymysql.connect(host="192.168.64.57", user="infoasadep01", password="tfyfInfo@1522", database="wind", port=3306, charset="utf8")
date = '20230105'
save_name = '500'

data1000 = getCSI1000Data(conn,date,date)
data500 = getCSI500Data(conn, date, date)
data300 = getHS300Data(conn, date, date)

data = data500
# %%
id_list = data.iloc[:, 0].tolist()
stock_id_list = ', '.join(id_list)
# 初始化Wind API
w.start()

# 获取风格因子数据
df = w.wss(stock_id_list, "mv_ref,pq_avgturn2,roe_ttm2,qfa_yoyprofit,pct_chg_per,industry_citic,val_lnmv,eqy_belongto_parcomsh","unit=1;tradeDate=20240105;startDate=20231204;endDate=20240105;rptDate=20230930;industryType=1;rptType=1")
df = pd.DataFrame(df.Data, columns=df.Codes, index=df.Fields).T

# %%
out_folder = 'wind_data_prj/'
os.makedirs(out_folder,exist_ok=True)
df.to_csv(out_folder + f'{save_name}风格因子.csv')

import pickle

def write_pickle(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
        return

def read_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

write_pickle(df,out_folder + f'{save_name}风格因子.pkl')
# %%
column_mapping = {
    'MV_REF': '总市值',
    'PQ_AVGTURN2': '换手率',
    'ROE_TTM2': 'ROETTM',
    'QFA_YOYPROFIT': '净利润增速',
    'PCT_CHG_PER': '反转',
    'INDUSTRY_CITIC': '所属行业',
    'VAL_LNMV': '市值对数',
    'EQY_BELONGTO_PARCOMSH': '净资产'
}


df.rename(columns=column_mapping, inplace=True)

df['BP'] = df['净资产']/df['总市值']
style_factors = df[['换手率', 'ROETTM', '净利润增速', '反转', '市值对数', 'BP']]

# 风格因子
print(style_factors)

# %%
# 使用get_dummies将"所属行业"列转化为独热编码
industry_dummies = pd.get_dummies(df['所属行业'])

# 行业因子矩阵
industry_factors = pd.DataFrame(industry_dummies)

# 因子数据
factor_scores = w.wss(stock_id_list,"qfa_roe","rptDate=20230930")
# HS300_factor = w.wss("000300.SH", "qfa_roe","rptDate=20230930")
CSI500_factor = w.wss('000905.SH',"qfa_roe","rptDate=20230930")
# %%
# 计算基准指数风格因子矩阵
merged_data = style_factors.merge(data[['s_con_windcode', 'weight']], left_index=True, right_on='s_con_windcode')
merged_data = merged_data.drop(columns='s_con_windcode')

merged_data = merged_data.set_index(style_factors.index)
selected_columns = merged_data.iloc[:, :6]
weighted_columns = selected_columns.mul(merged_data['weight']/100, axis=0)
weighted_sum = weighted_columns.sum()

# HS300_style = pd.DataFrame(weighted_sum.values.reshape(1,-1), index=["HS300"], columns=selected_columns.columns)
CSI500_style = pd.DataFrame(weighted_sum.values.reshape(1,-1), index=["CSI500"], columns=selected_columns.columns)
# %%
num_stocks = df.shape[0]
w_b = np.array(data['weight']/100)  
factor_values = np.array(factor_scores.Data[0])  
X = np.array(style_factors)  
H = np.array(industry_factors)  

# %%
# 初始化股票权重为优化变量
w = cp.Variable(num_stocks)

# 目标函数：最大化因子暴露
objective = cp.Maximize(factor_values.T @ w)

# %%
# 设置约束条件的参数

def indexData(style_data):
    # 市值对数这列数据保持不变，其他列都设为无穷
    style_sl = np.copy(style_data)
    style_sl[:, :4] = np.inf
    style_sl[:, 5:] = np.inf
    s_l = style_sl.reshape(-1)

    style_sh = np.copy(style_data)
    style_sh[:, :4] = -np.inf
    style_sh[:, 5:] = -np.inf
    s_h = style_sh.reshape(-1)
    return s_l, s_h
    

s_l, s_h = indexData(CSI500_style)

h_l = 0.5 * np.ones((H.shape[1],))  
h_h = 1.5 * np.ones((H.shape[1],))  

w_l = w_b - 0.01 * np.ones(int(save_name))
w_h = w_b + 0.01 * np.ones(int(save_name))

# b_l, b_h = 0, 1  # 成分股权重占比的上下限

l = 2  # 个股权重上限

# %%
# 定义约束
constraints = [
    s_l.T <= X.T @ (w - w_b),  # 风格暴露约束
    X.T @ (w - w_b) <= s_h.T,

    h_l <= H.T @ (w - w_b),  # 行业暴露约束
    H.T @ (w - w_b) <= h_h,

    w_l <= w - w_b,
    w - w_b <= w_h,

    # b_l <= w,  # 成分股权重占比约束
    # w <= b_h,

    0 <= w,
    # w <= l,

    cp.sum(w) == 1,  # 权重和为1
]

# %%
# 求解优化问题
prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.ECOS, verbose=True)

# %%
print(w.value)

# %%
result_df = pd.DataFrame({'Result': factor_values * w.value}, index=factor_scores.Codes)
# 降序排列
result_df1 = result_df.sort_values(by="Result", ascending=False)
# 选择前20%的数据并计算平均值
percentile = 0.2
num_rows_to_select = int(num_stocks * percentile)
average_value = result_df.head(num_rows_to_select)["Result"].mean()

# 计算超额收益
excess_returns = average_value - HS300_factor.Data
print("{:.2f}%".format(float(excess_returns[0]) * 100))
