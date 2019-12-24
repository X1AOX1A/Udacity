#!/usr/bin/env python
# coding: utf-8

# # 数据处理

# In[1]:


#导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
import xgboost as xgb
from time import time

#导入数据集
store=pd.read_csv(r'rossmann-store-sales/store.csv')
train=pd.read_csv(r'rossmann-store-sales/train.csv',dtype={'StateHoliday':pd.np.string_})
test=pd.read_csv(r'rossmann-store-sales/test.csv',dtype={'StateHoliday':pd.np.string_})

# In[2]:

train[train['Store']==622]
test.fillna(1,inplace=True)
store.fillna(0,inplace=True)

train=train[train['Sales']>0]
train=pd.merge(train,store,on='Store',how='left')
test=pd.merge(test,store,on='Store',how='left')


# ## 特征工程

# In[5]:


for data in [train,test]:
    #将时间特征进行拆分和转化
    data['year']=data['Date'].apply(lambda x:x.split('-')[0])
    data['year']=data['year'].astype(int)
    data['month']=data['Date'].apply(lambda x:x.split('-')[1])
    data['month']=data['month'].astype(int)
    data['day']=data['Date'].apply(lambda x:x.split('-')[2])
    data['day']=data['day'].astype(int)
    #将'PromoInterval'特征转化为'IsPromoMonth'特征，表示某天某店铺是否处于促销月，1表示是，0表示否
    #提示下：这里尽量不要用循环，用这种广播的形式，会快很多。循环可能会让你等的想哭
    month2str={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    data['monthstr']=data['month'].map(month2str)
    data['IsPromoMonth']=data.apply(lambda x:0 if x['PromoInterval']==0 else 1 if x['monthstr'] in x['PromoInterval'] else 0,axis=1)
    #将存在其它字符表示分类的特征转化为数字
    mappings={'0':0,'a':1,'b':2,'c':3,'d':4}
    data['StoreType'].replace(mappings,inplace=True)
    data['Assortment'].replace(mappings,inplace=True)
    data['StateHoliday'].replace(mappings,inplace=True)
    data['StoreType'] = data['StoreType'].astype('int')
    data['Assortment'] = data['Assortment'].astype('int')
    data['StateHoliday'] = data['StateHoliday'].astype('int')


# ## 划分数据集

# In[6]:


#删掉训练和测试数据集中不需要的特征
df_train=train.drop(['Date','Customers','Open','PromoInterval','monthstr'],axis=1)
df_test=test.drop(['Id','Date','Open','PromoInterval','monthstr'],axis=1)
#如上所述，保留训练集中最近六周的数据用于后续模型的测试
Xtrain=df_train[6*7*1115:]
Xtest=df_train[:6*7*1115]


# ## 对数变换

# In[8]:


#拆分特征与标签，并将标签取对数处理
ytrain=np.log1p(Xtrain['Sales'])
ytest=np.log1p(Xtest['Sales'])

Xtrain=Xtrain.drop(['Sales'],axis=1)
Xtest=Xtest.drop(['Sales'],axis=1)


# # 模型构建

# ## 定义评价函数

# In[9]:


#定义评价函数，可以传入后面模型中替代模型本身的损失函数
def rmspe(y,yhat):
    return np.sqrt(np.mean((yhat/y-1)**2))

def rmspe_xg(yhat,y):
    y=np.expm1(y.get_label())
    yhat=np.expm1(yhat)
    return 'rmspe',rmspe(y,yhat)


# ## 创建初始模型

# In[10]:


#初始模型构建
#参数设定
params={'objective':'reg:linear',
       'booster':'gbtree',
       'eta':0.03,
       'max_depth':10,
       'subsample':0.9,
       'colsample_bytree':0.7,
       'silent':1,
       'seed':10}
num_boost_round=6000
dtrain=xgb.DMatrix(Xtrain,ytrain)
dvalid=xgb.DMatrix(Xtest,ytest)
watchlist=[(dtrain,'train'),(dvalid,'eval')]

#模型训练
print('Train a XGBoost model')
start=time()
gbm=xgb.train(params,dtrain,num_boost_round,evals=watchlist,
             early_stopping_rounds=100,feval=rmspe_xg,verbose_eval=True)
end=time()
print('Train time is {:.2f} s.'.format(end-start))


# In[13]:


gbm.save_model('gbm.model')


# In[13]:


gbm = xgb.Booster(model_file='gbm.model')


# ## 结果分析

# In[16]:


#采用保留数据集进行检测
print('validating')
Xtest.sort_index(inplace=True)
ytest.sort_index(inplace=True)
yhat=gbm.predict(xgb.DMatrix(Xtest))
error=rmspe(np.expm1(ytest),np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

#构建保留数据集预测结果
res=pd.DataFrame(data=ytest)
res['Predicition']=yhat
res=pd.merge(Xtest,res,left_index=True,right_index=True)
res['Ratio']=res['Predicition']/res['Sales']
res['Error']=abs(res['Ratio']-1)
res['Weight']=res['Sales']/res['Predicition']
res.head()

#分析保留数据集中任意三个店铺的预测结果
col_1=['Sales','Predicition']
col_2=['Ratio']
L = np.array([510,437])
# L= np.random.randint(low=1,high=1115,size=3)
print('Mean Ratio of predition and real sales data is {}:store all'.format(res['Ratio'].mean()))
for i in L:
    s1=pd.DataFrame(res[res['Store']==i],columns=col_1)
    s2=pd.DataFrame(res[res['Store']==i],columns=col_2)
    s1.plot(title='Comparation of predition and real sales data:store {}'.format(i),figsize=(12,4))
    s2.plot(title='Ratio of predition and real sales data: store {}'.format(i),figsize=(12,4))
    print('Mean Ratio of predition and real sales data is {}:store {}'.format(s2['Ratio'].mean(),i))

# ## 细致优化

# In[17]:


#细致校正：以不同的店铺分组进行细致校正，每个店铺分别计算可以取得最佳RMSPE得分的校正系数
L=range(1115)
W_ho=[]
W_test=[]
for i in L:
    s1=pd.DataFrame(res[res['Store']==i+1],columns=col_1)
    s2=pd.DataFrame(df_test[df_test['Store']==i+1])
    W1=[(0.990+(i/1000)) for i in range(20)]
    S=[]
    for w in W1:
        error=rmspe(np.expm1(s1['Sales']),np.expm1(s1['Predicition']*w))
        S.append(error)
    Score=pd.Series(S,index=W1)
    BS=Score[Score.values==Score.values.min()]
    a=np.array(BS.index.values)
    b_ho=a.repeat(len(s1))
    b_test=a.repeat(len(s2))
    W_ho.extend(b_ho.tolist())
    W_test.extend(b_test.tolist())
#调整校正系数的排序
Xtest=Xtest.sort_values(by='Store')
Xtest['W_ho']=W_ho
Xtest=Xtest.sort_index()
W_ho=list(Xtest['W_ho'].values)
Xtest.drop(['W_ho'],axis=1,inplace=True)

df_test=df_test.sort_values(by='Store')
df_test['W_test']=W_test
df_test=df_test.sort_index()
W_test=list(df_test['W_test'].values)
df_test.drop(['W_test'],axis=1,inplace=True)

#计算校正后整体数据的RMSPE得分
yhat_new=yhat*W_ho
error=rmspe(np.expm1(ytest),np.expm1(yhat_new))
print('RMSPE for weight corretion {:.6f}'.format(error))


# In[20]:


#构建保留数据集预测结果
res=pd.DataFrame(data=ytest)
res['Predicition']=yhat_new
res=pd.merge(Xtest,res,left_index=True,right_index=True)
res['Ratio']=res['Predicition']/res['Sales']
res['Error']=abs(res['Ratio']-1)
res['Weight']=res['Sales']/res['Predicition']
res.head()

#分析保留数据集中任意两个店铺的预测结果
col_1=['Sales','Predicition']
col_2=['Ratio']
L = np.array([510,437])
# L= np.random.randint(low=1,high=1115,size=3)
print('Mean Ratio of predition and real sales data is {}:store all'.format(res['Ratio'].mean()))
for i in L:
    s1=pd.DataFrame(res[res['Store']==i],columns=col_1)
    s2=pd.DataFrame(res[res['Store']==i],columns=col_2)
    s1.plot(title='Comparation of predition and real sales data:store {}'.format(i),figsize=(12,4))
    s2.plot(title='Ratio of predition and real sales data: store {}'.format(i),figsize=(12,4))
    print('Mean Ratio of predition and real sales data is {}:store {}'.format(s2['Ratio'].mean(),i))


# ## 导出结果

# In[21]:


#用初始和校正后的模型对训练数据集进行预测
print('Make predictions on the test set')
dtest=xgb.DMatrix(df_test)
test_probs=gbm.predict(dtest)

# #初始模型
# result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs)})
# result.to_csv(r'result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*W_test)})
# result.to_csv(r'/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission/submission1.csv',index=False)
# #整体校正模型
# result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*0.996)})
# result.to_csv(r'result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*W_test)})
# result.to_csv(r'/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission/submission2.csv',index=False)
#细致校正模型
result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*W_test)})
result.to_csv(r'/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission/submission.csv',index=False)

