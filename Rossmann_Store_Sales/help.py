import numpy as np
import pandas as pd

#----------------read data-----------------#
def read_data(path='data'):
    print('Loading train data...')
    train = pd.read_csv(path+'/train.csv', index_col = 0)

    print('Loading test data...')
    test = pd.read_csv(path+'/test.csv', index_col = 0)
    
    print('Loading test data...')
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
    
def scale_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    col = ['Store','Month','Day','Year','DayOfWeek','CompetitionDistance',
             'CompetitionOpenTime','Promo2SinceWeek','Promo2SinceYear']
    data[col] = scaler.fit_transform(data[col])
    return data
    
    
def train_preprocess(train_data, extend_data):
    store_pre, Bow_Matrix = BOW(Data=extend_data)
    train_data = extend(train_data.drop(['Sales','Customers'], axis=1), store_pre, Bow_Matrix.columns)
    train_data = one_hot(train_data)
    train_data = drop_columns(train_data)
    train_data = scale_data(train_data)
    print('  Successfully preprocess training data.\n')
    return train_data
    
def test_preprocess(test_data, extend_data):
    store_pre, Bow_Matrix = BOW(Data=extend_data)
    test_data = extend(test_data.drop('Id', axis=1), store_pre, Bow_Matrix.columns)
    test_data = one_hot(test_data)
    test_data[['StateHoliday_b', 'StateHoliday_c']] = pd.DataFrame(np.zeros((len(test_data),2)), 
                                                               columns=['StateHoliday_b', 'StateHoliday_c'], 
                                                               dtype='uint8')
    test_data = drop_columns(test_data)
    test_data = scale_data(test_data)
    print('  Successfully preprocess testing data.\n')
    return test_data

def data_preprocess(train_data, test_data, extend_data):
    print('Preprocessing training data...')
    X1 = train_preprocess(train_data, extend_data)
    print('Preprocessing testing data...')
    X2 = test_preprocess(test_data, extend_data)
    y1 = train_data['Sales']
    return X1, y1, X2


#----------------define score function-----------------#
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w
 
def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe
