# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:02:49 2020

@author: Charlotte
"""


from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_data():
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
    dataset = read_csv(url, compression='gzip')
    dataset['date_time']= pd.to_datetime(dataset['date_time'])
    dataset['dayofweek']=dataset['date_time'].dt.dayofweek
    dataset['hour']=dataset['date_time'].dt.hour
    return dataset

#one hot encoding
#input: column of dataset
def onehot_encoding(data):
    values = np.array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

if __name__ == "__main__":
    dataset=load_data()
    one_holiday = onehot_encoding(dataset['holiday'])
    one_wmain = onehot_encoding(dataset['weather_main'])
    one_wdesc = onehot_encoding(dataset['weather_description'])
    one_day = onehot_encoding(dataset['dayofweek'])
    one_hour = onehot_encoding(dataset['hour'])

    #Splitting the dataset into Training set and Test Set
    X=dataset.iloc[:, 1:5].values
    X=np.concatenate((one_holiday,X,one_wmain,one_wdesc,one_day,one_hour),axis=1)
    y=dataset.iloc[:, 8].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

    #Standardisation
    std= StandardScaler()
    x_train= std.fit_transform(x_train)
    x_test=std.transform(x_test)