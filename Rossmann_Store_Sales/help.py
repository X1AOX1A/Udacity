import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------read data-----------------#
def read_data(path='data'):
    print('Loading train data...')
    train = pd.read_csv(path+'/train.csv', index_col = 0, low_memory=False)

    print('Loading test data...')
    test = pd.read_csv(path+'/test.csv', index_col = 0)
    
    print('Loading store data...')
    store = pd.read_csv(path+'/store.csv', index_col = 0)
    print('  Successfully read data as: train, test, store.\n')
    return train, test, store


#----------------data preprocess-----------------#
from sklearn.feature_extraction.text import CountVectorizer
def BOW(Data):
    # transform 'PromoInterval' into independent columns
    from sklearn.feature_extraction.text import CountVectorizer
    count_vector = CountVectorizer()

    doc_array = count_vector.fit_transform(Data['PromoInterval'].fillna('NaN')).toarray()
    count_vector.get_feature_names()

    def normallize(item):
        return 'PromoInterval_'+item.capitalize()
    columns = list(map(normallize,count_vector.get_feature_names()))

    Bow_Matrix = pd.DataFrame(doc_array, columns=columns)

    Data[Bow_Matrix.columns] = Bow_Matrix
    return Data.drop('PromoInterval', axis=1), Bow_Matrix

def extend(Data,extend_data,BOW_key):
    Date_to_Date = lambda x: x.day
    Date_to_Month = lambda x: x.month
    Date_to_Year = lambda x: x.year
#     Month_to_Season = dict(zip(np.arange(12)+1, np.repeat(np.arange(4)+1,3)))
    # merge data
    Data = pd.merge(Data,extend_data,how='left')    
    # transform time data('Data') to ['Day','Month','Year']
    Data['Date2'] = pd.to_datetime(Data['Date']) 
    Data['Day'] = Data['Date2'].apply(Date_to_Date)
    Data['Month'] = Data['Date2'].apply(Date_to_Month)
    Data['Year'] = Data['Date2'].apply(Date_to_Year)
#     Data['Season'] = Data['Month'].map(Month_to_Season)
    Data.drop(['Date','Date2'], axis=1, inplace=True)
    Data = Data[['Store', 'Month', 'Day', 'Year', 'DayOfWeek',
     'Open','StateHoliday', 'SchoolHoliday','CompetitionDistance',
     'CompetitionOpenTime','StoreType', 'Assortment', 'Promo',
     'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']+BOW_key.to_list()]
#     Data = Data[['Store', 'Month', 'Day', 'Year', 'DayOfWeek',
#      'Season', 'Open','StateHoliday', 'SchoolHoliday','CompetitionDistance',
#      'CompetitionOpenTime','StoreType', 'Assortment', 'Promo',
#      'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']+BOW_key.to_list()]
    return Data

def one_hot(Data):
    # One-Hot on ['StateHoliday', 'Assortment', 'StateHoliday']
    index = Data[Data['StateHoliday'] == 0].index
    Data.loc[index,['StateHoliday']] = '0'

    One_Hot_Matrix = pd.get_dummies(Data[['StoreType','Assortment','StateHoliday']]) 
    Data[One_Hot_Matrix.columns] = One_Hot_Matrix
    Data.drop(['StoreType','Assortment','StateHoliday'], axis=1, inplace=True)
    return Data

def drop_columns(data):
    # 删除 ['PromoInterval_Nan', 'StoreType_d', 'Assortment_c', 'StateHoliday_0'] 防止共线性
    data.drop(['PromoInterval_Nan', 'StoreType_d', 'Assortment_c', 'StateHoliday_0'], axis=1, inplace=True)
    return data

from sklearn.preprocessing import StandardScaler
def scale_data1(data):
    scaler = StandardScaler()
    col = ['Store','Month','Day','Year','DayOfWeek','CompetitionDistance',
             'CompetitionOpenTime','Promo2SinceWeek','Promo2SinceYear']
    data[col] = scaler.fit_transform(data[col])
    return data, scaler

def scale_data2(data,scaler):
    col = ['Store','Month','Day','Year','DayOfWeek','CompetitionDistance',
             'CompetitionOpenTime','Promo2SinceWeek','Promo2SinceYear']
    data[col] = scaler.transform(data[col])
    return data
    
    
def train_preprocess(train_data, extend_data):
    store_pre, Bow_Matrix = BOW(Data=extend_data)
    train_data = extend(train_data.drop(['Sales','Customers'], axis=1), store_pre, Bow_Matrix.columns)
    train_data = one_hot(train_data)
    train_data = drop_columns(train_data)
    train_data,scaler = scale_data1(train_data)
    print('  Successfully preprocess training data.\n')
    return train_data,scaler
    
def test_preprocess(test_data, extend_data, scaler):
    store_pre, Bow_Matrix = BOW(Data=extend_data)
    test_data = extend(test_data.drop('Id', axis=1), store_pre, Bow_Matrix.columns)
    test_data = one_hot(test_data)
    test_data[['StateHoliday_b', 'StateHoliday_c']] = pd.DataFrame(np.zeros((len(test_data),2)), 
                                                                   columns=
                                                                   ['StateHoliday_b','StateHoliday_c'], 
                                                                   dtype='uint8')
    test_data = drop_columns(test_data)
    test_data = scale_data2(test_data, scaler)
    print('  Successfully preprocess testing data.\n')
    return test_data

def data_preprocess(train_data, test_data, extend_data):
    print('Preprocessing training data...')
    X1,scaler = train_preprocess(train_data, extend_data)
    print('Preprocessing testing data...')
    X2 = test_preprocess(test_data, extend_data, scaler)
    y1 = train_data['Sales']
    return X1, y1, X2


#----------------define score function-----------------#
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w
 
def rmspe(yhat, y):
    y = np.array(y).reshape( (len(y),) )
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_log(yhat, y):
    yhat = np.expm1(yhat)
    y = np.expm1(y)
    y = np.array(y).reshape( (len(y),) )
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


#----------------visual the train result-----------------#
def visual_result(model,list_X_y, same_axis=False):
    X = list_X_y[0]
    y = list_X_y[1]
    df = pd.DataFrame(model.cv_results_)
    df.loc[:,['mean_train_score','mean_test_score']] = -df[['mean_train_score','mean_test_score']]

    # visualize the score
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    if same_axis:
        vmin=df[['mean_test_score','mean_train_score']].min().min()
        vmax=df[['mean_test_score','mean_train_score']].max().max()
        sns.heatmap(df.groupby([X,y])['mean_train_score'].mean().unstack(), 
                    ax=axes[0], vmin=vmin, vmax=vmax, annot=True, fmt='.3f')

        sns.heatmap(df.groupby([X,y])['mean_test_score'].mean().unstack(), 
                    ax=axes[1], vmin=vmin, vmax=vmax, annot=True, fmt='.3f')
    else:
        sns.heatmap(df.groupby([X,y])['mean_train_score'].mean().unstack(), 
                    ax=axes[0], annot=True, fmt='.3f')

        sns.heatmap(df.groupby([X,y])['mean_test_score'].mean().unstack(), 
                    ax=axes[1], annot=True, fmt='.3f')
        
    axes[0].set_title('Mean Train Respe Score')
    axes[1].set_title('Mean Test Respe Score')
    
    
#----------------print the train and test score-----------------#
from sklearn.model_selection import train_test_split
import datetime
def train_test_score(Model=None, train_size=762906, test_size=254303, X=None, y=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=520)
    
    print('Calculating train score...')
    starttime = datetime.datetime.now()
    y_train_hat = Model.predict(X_train[:train_size])
    print('  Train rmspe score:',round(rmspe(y_train_hat, y_train[:train_size]),5) )
    endtime = datetime.datetime.now()
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    print('Calculating test score...')
    starttime = datetime.datetime.now()
    y_test_hat = Model.predict(X_test[:test_size])
    print('  Test rmspe score:',round(rmspe(y_test_hat, y_test[:test_size]),5) )
    endtime = datetime.datetime.now()
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
def train_test_score_log(Model=None, train_size=762906, test_size=254303, X=None, y=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=520)
    
    print('Calculating train score...')
    starttime = datetime.datetime.now()
    y_train_hat = Model.predict(X_train[:train_size])
    print('  Train rmspe score:',round(rmspe_log(y_train_hat, y_train[:train_size]),5) )
    endtime = datetime.datetime.now()
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    print('Calculating test score...')
    starttime = datetime.datetime.now()
    y_test_hat = Model.predict(X_test[:test_size])
    print('  Test rmspe score:',round(rmspe_log(y_test_hat, y_test[:test_size]),5) )
    endtime = datetime.datetime.now()
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    
#----------------save model and print the best estimator-----------------#
import joblib
import datetime
def save_model(Model, FileName, Best_Model=True):
    print('Saving model...')
    starttime = datetime.datetime.now()
    FileName2 = 'Model_Parameter2/'+FileName+'.pkl'
    joblib.dump(Model, FileName2) 
    endtime = datetime.datetime.now()
    print('  The Model have been save in ','[\''+FileName2+'\']')
    print('    PS: To load model, use command: \'Model = joblib.load(FileName)\'.')
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    if Best_Model:
        print(' ',FileName+'.best_estimator_:')
        print('   ',Model.best_estimator_,'\n')

        
#----------------predict and save submission-----------------#
def submission(Model, Save_File=True):
    starttime = datetime.datetime.now()
    print('Predicting submission...')
    submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
    X_predict_reduce = pd.read_csv('data/X_predict_reduce.csv', index_col = 0)    
#     submission['Sales'] = np.exp(Model.predict(X_predict_reduce))
    submission['Sales'] = Model.predict(X_predict_reduce)
    endtime = datetime.datetime.now()
    print('  Done!')
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    if Save_File:
        print('Saving submission...')
        submission.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission.csv',
                      index=False)
        print('  Done!')
        print('  PS: To submit file, use command:')
        print('    cd /Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales')
        print('    kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "Message"')
        
def submission_log(Model, Save_File=True):
    starttime = datetime.datetime.now()
    print('Predicting submission...')
    submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
    X_predict_reduce = pd.read_csv('data/X_predict_reduce.csv', index_col = 0)    
    submission['Sales'] = np.expm1(Model.predict(X_predict_reduce))
    endtime = datetime.datetime.now()
    print('  Done!')
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    if Save_File:
        print('Saving submission...')
        submission.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission.csv',
                      index=False)
        print('  Done!')
        print('  PS: To submit file, use command:')
        print('    cd /Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales')
        print('    kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "Message"')
        
def submission_log2(Model, Save_File=True):
    starttime = datetime.datetime.now()
    print('Predicting submission...')
    submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
    X_predict_reduce = pd.read_csv('data/X_predict_process.csv', index_col = 0)    
    submission['Sales'] = np.expm1(Model.predict(X_predict_reduce))
    endtime = datetime.datetime.now()
    print('  Done!')
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    if Save_File:
        print('Saving submission...')
        submission.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission.csv',
                      index=False)
        print('  Done!')
        print('  PS: To submit file, use command:')
        print('    cd /Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales')
        print('    kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "Message"')
        
def submission_log_adjust(Model, Weight, Save_File=True):
    starttime = datetime.datetime.now()
    print('Predicting submission...')
    submission = pd.read_csv('rossmann-store-sales/sample_submission.csv')
    X_predict_reduce = pd.read_csv('data/X_predict_reduce.csv', index_col = 0) 
    df_submission = pd.read_csv('data/X_predict_process.csv', index_col = 0) 

    df_submission = pd.merge(df_submission, Weight, on='Store')
    submission['Sales'] = np.expm1(Model.predict(X_predict_reduce)) * df_submission['Adjust_Weight']
    
    endtime = datetime.datetime.now()
    print('  Done!')
    print('  Using time:', (endtime - starttime).seconds, 'sec\n')
    
    if Save_File:
        print('Saving submission...')
        submission.to_csv('/Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales/submission.csv',
                      index=False)
        print('  Done!')
        print('  PS: To submit file, use command:')
        print('    cd /Users/apple/Documents/Jupyter/Udacity/Rossmann_Store_Sales')
        print('    kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "Message"')
        
#----------------Voting-----------------#
def Voting(estimators, data):
    predict_mean = np.zeros(len(data), dtype=float)
    for estimator in estimators:
        predict_mean = predict_mean + estimator.predict(data)
    return predict_mean/len(estimators)

