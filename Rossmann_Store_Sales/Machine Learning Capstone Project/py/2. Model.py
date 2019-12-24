#!/usr/bin/env python
# coding: utf-8

# # 数据准备
# 
# ## 数据预处理

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[2]:


from help import read_data, data_preprocess
train, test, store = read_data()
X_reduce, y, X_predict_reduce = data_preprocess(train, test, store)

del train, test, store


# ## 数据划分

# In[3]:


from sklearn.model_selection import train_test_split
X_train_reduce, X_test_reduce, y_train, y_test = train_test_split(X_reduce, y, random_state=520)


# # 模型
# 
# [定义评分函数：](https://www.cnblogs.com/harvey888/p/6964741.html)

# In[8]:


from help import rmspe
from sklearn.metrics import make_scorer
respe_scorer = make_scorer(rmspe, greater_is_better=False)


# ## Tree

# ### 决策树

# In[81]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

Tree = DecisionTreeRegressor(min_samples_leaf=4)
parameters = {'max_depth':[25,30,35,40,50],
              'min_samples_split':[7,10,15,20,30]}

Tree_grid_search = GridSearchCV(Tree, parameters, scoring=respe_scorer, 
                                cv=5, n_jobs=-1, return_train_score=True)

import datetime
starttime = datetime.datetime.now()
print('Fitting model...')Z
Tree_grid_search.fit(X_train, y_train)
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

from help import save_model
Model = Tree_grid_search
FileName = 'Tree_grid_search'
save_model(Model,FileName)

from help import train_test_score
train_test_score(Model,X=X,y=y)

from help import visual_result
visual_result(Model, ['param_'+i for i in list(parameters.keys())],same_axis=False)


# ### 随机森林

# In[40]:


from sklearn.ensemble import RandomForestRegressor
Random_Forest_reduce = RandomForestRegressor(n_estimators=100, n_jobs=-1)

import datetime
starttime = datetime.datetime.now()
print('Fitting model...')
Random_Forest_reduce.fit(X_train_reduce, y_train)
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

Model = Random_Forest_reduce
FileName = 'Random_Forest_reduce'
from help import save_model
save_model(Model, FileName, Best_Model=False)

from help import train_test_score
train_test_score(Model, X=X_reduce, y=y)


# ### 极端随机树

# In[41]:


from sklearn.ensemble import ExtraTreesRegressor
Extra_Forest_reduce = ExtraTreesRegressor(n_estimators=100, bootstrap=True, n_jobs=-1)

import datetime
starttime = datetime.datetime.now()
print('Fitting model...')
Extra_Forest_reduce.fit(X_train_reduce, y_train)
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

Model = Extra_Forest_reduce
FileName = 'Extra_Forest_reduce'
from help import save_model
save_model(Model, FileName, Best_Model=False)

from help import train_test_score
train_test_score(Model, X=X_reduce, y=y)


# # Ensemble

# ## Voting

# In[3]:


DT = joblib.load('Model_Parameter2/Tree_grid_search_reduce.pkl')
RT = joblib.load('Model_Parameter2/Random_Forest_reduce.pkl')
ET = joblib.load('Model_Parameter2/Extra_Forest_reduce.pkl')


# In[4]:


from help import Voting, rmspe
rmspe(Voting([DT, RT, ET], data = X_train_reduce), y_train)


# In[3]:


from help import Voting, rmspe
rmspe(Voting([DT, RT, ET], data = X_test_reduce), y_test)


# In[7]:


submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
submission['Sales'] = Voting([DT, RT, ET], data = X_predict_reduce)
submission.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission.csv',
                      index=False)


# ## Bagging

# In[4]:


from sklearn.ensemble import BaggingRegressor
Base_Model = joblib.load('Model_Parameter2/Tree_grid_search_reduce.pkl').best_estimator_
Bagging_reg_Decision_Tree_reduce = BaggingRegressor(Base_Model, n_estimators=30)

import datetime
starttime = datetime.datetime.now()
print('Fitting model...')
Bagging_reg_Decision_Tree_reduce.fit(X_train_reduce, y_train.values.ravel())
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

Model = Bagging_reg_Decision_Tree_reduce
FileName = 'Bagging_reg_Decision_Tree_reduce'
from help import save_model
save_model(Model, FileName, Best_Model=False)

from help import train_test_score
train_test_score(Model, X=X_reduce, y=y)

from help import submission
submission(Model, Save_File=True)


# ## Adaboost

# In[53]:


from sklearn.ensemble import AdaBoostRegressor
Base_Model = joblib.load('Model_Parameter2/Tree_grid_search_reduce.pkl').best_estimator_
Ada_reg_Decision_Tree_reduce = AdaBoostRegressor(Base_Model, n_estimators=25,
                                          learning_rate=1)
import datetime
starttime = datetime.datetime.now()
print('Fitting model...')
Ada_reg_Decision_Tree_reduce.fit(X_train_reduce, y_train.values.ravel())
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

Model = Ada_reg_Decision_Tree_reduce
FileName = 'Ada_reg_Decision_Tree_reduce'
from help import save_model
save_model(Model, FileName, Best_Model=False)

from help import train_test_score
train_test_score(Model, X=X_reduce, y=y)

from help import submission
submission(Model, Save_File=True)


# ## GBRT

# In[6]:


from sklearn.ensemble import GradientBoostingRegressor
GBRT_reg_Decision_Tree_reduce = GradientBoostingRegressor(n_estimators=40,
    criterion='mse', max_depth=60, max_features=None,
    max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, min_samples_leaf=4,
    min_samples_split=20, min_weight_fraction_leaf=0.0,
    presort=False, random_state=None)

import datetime
starttime = datetime.datetime.now()
print('Fitting model...')
GBRT_reg_Decision_Tree_reduce.fit(X_train_reduce, y_train.values.ravel())
endtime = datetime.datetime.now()
print('  Done!  \n  Using time:',(endtime - starttime).seconds,'sec\n')

Model = GBRT_reg_Decision_Tree_reduce
FileName = 'GBRT_reg_Decision_Tree_reduce'
from help import save_model
save_model(Model, FileName, Best_Model=False)

from help import train_test_score
train_test_score(Model, X=X_reduce, y=y)

from help import submission
submission(Model, Save_File=True)


