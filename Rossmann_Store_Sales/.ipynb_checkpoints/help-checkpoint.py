from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

def BOW(Data):

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

def data_preprocess(Data, extend_data, BOW_key):
    
    Date_to_Date = lambda x: x.day
    Date_to_Month = lambda x: x.month
    Date_to_Year = lambda x: x.year
    Month_to_Season = dict(zip(np.arange(12)+1, np.repeat(np.arange(4)+1,3)))

    # merge data
    Data = pd.merge(Data,extend_data,how='left')
    
    # transform time data('Data') to ['Day','Month','Year','Season']
    Data['Date2'] = pd.to_datetime(Data['Date']) 
    Data['Day'] = Data['Date2'].apply(Date_to_Date)
    Data['Month'] = Data['Date2'].apply(Date_to_Month)
    Data['Year'] = Data['Date2'].apply(Date_to_Year)
    Data['Season'] = Data['Month'].map(Month_to_Season)
    Data.drop(['Date','Date2'], axis=1, inplace=True)
    Data = Data[['Store', 'Month', 'Day', 'Year', 'DayOfWeek',
     'Season', 'Open','StateHoliday', 'SchoolHoliday','CompetitionDistance',
     'CompetitionOpenTime','StoreType', 'Assortment', 'Promo',
     'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']+BOW_key.to_list()]
    
    # One-Hot on ['StateHoliday', 'StateHoliday']
    index = Data[Data['StateHoliday'] == 0].index
    Data.loc[index,['StateHoliday']] = '0'

    One_Hot_Matrix = pd.get_dummies(Data[['StoreType','Assortment','StateHoliday']]) 
    Data[One_Hot_Matrix.columns] = One_Hot_Matrix
    Data.drop(['StoreType','Assortment','StateHoliday'], axis=1, inplace=True)
    
    return Data