#!/usr/bin/env python
# coding: utf-8

# # 数据分析

# ## 数据说明及清洗

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[2]:


sample_submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
store = pd.read_csv('rossmann-store-sales/store.csv')
train = pd.read_csv('rossmann-store-sales/train.csv')
test = pd.read_csv('rossmann-store-sales/test.csv')


# ### 提交数据集
# 
# **提交数据集**：包含两列，第一列为训练数据的 ID 编号，第二列 Sales 为该商店对应的销量 Sales。

# In[3]:


sample_submission.head()


# ### 训练数据集

# **训练数据集**：包括销售在内的历史数据，包含以下变量：
# 
# | 变量名 | 变量含义 |变量类型 | 取值范围 | 备注 |
# | ------ | ------ | ------ | ------ | ------ | 
# | **Store** | 商店编号 | 离散型 | [1,1115] |  | 
# | **DayOfWeek** | 星期 | 离散型 | [1,7] |  | 
# | **Date** | 销售日期 | 离散型 | [2013-01-01,2015-07-31] |  | 
# | **Sales** | 销售量 | 连续型 | [0,41551] |  | 
# | **Customers** | 顾客数 | 连续型 | [0,7388] |  | 
# | **Open** | 是否营业 | 离散型 | [0,1] |  | 
# | **Promo** | 当日是否有促销活动 | 离散型 | [0,1] |  | 
# | **StateHoliday** | 国定假日 | 离散型 | [0,a,b,c] | a=公共假日，b=复活节假日，c=圣诞节，0=无 | 
# | **SchoolHoliday** | 学校假日 | 离散型 | [0,1] | 是否受到公立学校关闭的影响 | 
# 
# > *通常除了少数例外，所有商店都在国定假日关门。请注意，所有学校都在公共假日和周末关闭。*

# In[4]:


train.head()


# #### 异常值处理
# 
# 可以看到，原数据中 StateHoliday 类数据有 ['0', 'a', 'b', 'c', 0] ，0 和 '0' 属于同一类，因此将 0 类数据转换为 '0'。

# In[5]:


pd.unique(train['StateHoliday'])


# In[6]:


index = train[train['StateHoliday'] == 0].index
train.loc[index,['StateHoliday']] = '0'


# In[7]:


pd.unique(train['StateHoliday'])


# In[8]:


train.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/data/train.csv')


# ### 测试数据集

# **训练数据集**：历史数据（不包括销售额），包含以下变量：
# 
# | 变量名 | 变量含义 |变量类型 | 取值范围 | 备注 |
# | ------ | ------ | ------ | ------ | ------ | 
# | **Id** | 记录编号 | 离散型 | [1,41088] |  | 
# | **Store** | 商店编号 | 离散型 | [1,1115] |  | 
# | **DayOfWeek** | 星期 | 离散型 | [1,7] |  | 
# | **Date** | 销售日期 | 离散型 | [2015-08-01,2015-09-17] |  | 
# | **Open** | 是否营业 | 离散型 | [0,1] | nan为未知 | 
# | **Promo** | 当日是否有促销活动 | 离散型 | [0,1] |  | 
# | **StateHoliday** | 国定假日 | 离散型 | [0,a,b,c] | a=公共假日，b=复活节假日，c=圣诞节，0=无 | 
# | **SchoolHoliday** | 学校假日 | 离散型 | [0,1] | 是否受到公立学校关闭的影响 | 
# 
# *通常所有商店，除了少数例外，都在国定假日关门。请注意，所有学校都在公共假日和周末关闭。*

# In[9]:


test.head()


# #### 异常值处理

# 可以看到，编号为 622 的商店在 '2015-09-05' 至 '2015-09-17' 中除了周日，其他日期的 **Open** 数据都为缺失值：

# In[10]:


test[test['Open'].isnull()].sort_values(by='Date')


# 通过对 622 号商店的数据观察，可以发现其正常的开业时间为周一至周六，而考虑到在这段时间内没有其他法定假期，因此将其缺失数据填充为1。

# In[11]:


test.loc[test[test['Open'].isnull()].index, 'Open'] = np.array(1)
test['Open'] = test['Open'].astype('int64')
test.head()


# In[12]:


test.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/data/test.csv')


# ### 商店数据集
# 
# **商店数据集**：每个商店的附加信息，包含以下变量：
# 
# | 变量名 | 变量含义 |变量类型 | 取值范围 | 备注 |
# | ------ | ------ | ------ | ------ | ------ | 
# | **Store** | 商店编号 | 离散型 | [1,,1115] | |
# | **StoreType** | 商店类型  | 离散型 | [a,b,c,d] |  |
# | **Assortment** | 产品组合  | 离散型 | [a,b,c] | 描述产品组合级别：a=基本，b=附加，c=扩展 |
# | **CompetitionDistance** | 竞争者距离  | 连续型 | [20,75860] | 单位：米 |
# | **CompetitionOpenSinceMonth** | 竞争者开业月份 | 离散型 | [1,12] | nan 为未知 |
# | **CompetitionOpenSinceYear** | 竞争者开业年份 | 离散型 | [1900,2015] | nan 为未知 |
# | **Promo2** | 商店的连续促销 | 离散型  | [0,1] | 0=商店未参与，1=商店正在参与 |
# | **Promo2SinceWeek** | 促销开始周数 | 离散型 | [1,50] | nan 为未知 |
# | **Promo2SinceYear** | 促销开始年份 | 离散型 | [2009,2015] | nan 为未知 |
# | **PromoInterval** | 促销期间 | 离散型 | [Jan,Dec] | nan 为未知 |
# 
# > **PromoInterval** 描述Promo2开始的连续间隔，并命名重新开始促销的月份。例如，“2月、5月、8月、11月”是指该商店在任何给定年份的2月、5月、8月、11月开始的每一轮

# In[13]:


store.head()


# #### 异常值处理
# 
# 从箱线图可以看出有两个明显对离群点，因此将其赋值为未知值。

# In[14]:


store['CompetitionOpenSinceYear'].plot.box()


# In[15]:


store[~store['CompetitionOpenSinceYear'].isnull()].sort_values(by='CompetitionOpenSinceYear',ascending=True).head()


# In[16]:


nan_index = store[~store['CompetitionOpenSinceYear'].isnull()].sort_values(by='CompetitionOpenSinceYear',ascending=True).head().index[:2]
store.loc[nan_index, ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear']] = np.nan
store['CompetitionOpenSinceYear'].plot.box()


# #### 数据转换
# 
# 将对手开业年月转换为绝对时间（月）

# In[17]:


store[~store['CompetitionOpenSinceMonth'].isnull()].head()


# In[18]:


store['CompetitionOpenTime'] = pd.Series(np.nan)


# In[19]:


store.loc[:,'CompetitionOpenTime'] = (2015 - store['CompetitionOpenSinceYear'])*12 + 13 - store['CompetitionOpenSinceMonth'] 
del store['CompetitionOpenSinceMonth']
del store['CompetitionOpenSinceYear']


# In[20]:


store.head()


# #### 缺失值处理
# 
# 对竞争者的距离和开业时间的缺失值，分三种情况处理：
# 
# 1. CompetitionDistance 缺失，CompetitionOpenTime 不缺失：使用 CompetitionDistance 均值填充
# 2. CompetitionDistance 不缺失，CompetitionOpenTime 缺失：使用 CompetitionOpenTime 均值填充
# 3. CompetitionDistance 缺失，CompetitionOpenTime 缺失：使用 CompetitionDistance 最大值填充 CompetitionDistance，使用 0 填充 CompetitionOpenTime

# In[21]:


Distance_Null = store['CompetitionDistance'].isnull()
OpenTime_Null = store['CompetitionOpenTime'].isnull()

store.loc[Distance_Null & ~OpenTime_Null, 'CompetitionDistance'] = store['CompetitionDistance'].mean()
store.loc[~Distance_Null & OpenTime_Null, 'CompetitionOpenTime'] = store['CompetitionOpenTime'].mean()
store.loc[Distance_Null & OpenTime_Null, 'CompetitionDistance'] = max(store['CompetitionDistance'])
store.loc[Distance_Null & OpenTime_Null, 'CompetitionOpenTime'] = np.array(0)

print(sum(store['CompetitionDistance'].isnull()))
print(sum(store['CompetitionOpenTime'].isnull()))


# 对促销策略 Promo2 当无促销策略时，将其 Promo2SinceYear 和Promo2SinceWeek
# 赋值为 0:

# In[22]:


store.loc[store['Promo2SinceYear'].isnull(), 'Promo2SinceYear'] = np.array(0)
store.loc[store['Promo2SinceWeek'].isnull(), 'Promo2SinceWeek'] = np.array(0)


# In[23]:


store.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/data/store.csv')


# ## 数据探索
# 
# ### 数据联合
# 
# 联合训练集 train 与 附加信息数据集 store，并将 Date 数据转换为年月季度：

# In[39]:


Date_to_Date = lambda x: x.day
Date_to_Month = lambda x: x.month
Date_to_Year = lambda x: x.year
Month_to_Season = dict(zip(np.arange(12)+1, np.repeat(np.arange(4)+1,3)))

train_extend = pd.merge(train,store,how='left')
train_extend['Date2'] = pd.to_datetime(train_extend['Date']) 
train_extend['Day'] = train_extend['Date2'].apply(Date_to_Date)
train_extend['Month'] = train_extend['Date2'].apply(Date_to_Month)
train_extend['Year'] = train_extend['Date2'].apply(Date_to_Year)
train_extend['Season'] = train_extend['Month'].map(Month_to_Season)
train_extend.drop(['Date','Date2'], axis=1, inplace=True)
train_extend = train_extend[['Store', 'Sales', 'Customers', 'Month', 'Day', 'Year', 'DayOfWeek',
 'Season', 'Open','StateHoliday', 'SchoolHoliday','CompetitionDistance',
 'CompetitionOpenTime','StoreType', 'Assortment', 'Promo',
 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear','PromoInterval']]
train_extend.head()


# ### 箱线图

# In[43]:


fig_num = 6
fig, axes = plt.subplots(2, 3, figsize=(12,6))
axes_x = [0,0,0,1,1,1]
axes_y = [0,1,2,0,1,2]
for i,key in enumerate(['Promo','StateHoliday','SchoolHoliday','StoreType','Assortment','Promo2']):
    sns.boxplot(x=key, y='Sales', data=train_extend, ax=axes[axes_x[i],axes_y[i]])
plt.show()


# 1. 可以看出当商店正处于促销期（Promo=1）时销量高于平时时期；
# 2. 当不处于法定假期（StateHoliday=0）时销量高于其他时期；
# 3. 学习是否放假对销量没有明显影响；
# 4. b 类商店的销量高于其他商店；
# 5. 销售 b 类商品组合的商店销量稍高于其他商店；
# 6. 是否有周期性的促销策略对销量没有明显提高

# ### 可视化不同时段的销量

# In[46]:


train_extend.groupby(by='DayOfWeek')['Sales'].mean().plot.bar()


# In[48]:


train_extend.groupby(by=['Month','Year'])['Sales'].mean().unstack().plot()

