# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:41:26 2020

@author: Charlotte
"""


from pandas import read_csv
from matplotlib import pyplot
import preprocessing_traffic as traff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv"
names = ['IDLink','title', 'headline', 'source', 'topic', 'publishDate', 'SentimentTitle', 'SentimentHeadline', 'Facebook', 'GooglePlus','LinkedIn']
type_dict = {'IDLink': 'float64', 'title':'str','headline':'str','source':'str','topic':'str'
             ,'SentimentTitle':'float16', 'SentimentHeadline':'float16'
             ,'Facebook':'int','GooglePlus':'int16','LinkedIn':'int16'}
dataset = read_csv(url, names=names, dtype=type_dict, header=0)

#missing values?
dataset.columns[dataset.isnull().any()]

dataset = dataset.dropna().sort_values(by="IDLink").reset_index()

onehot_topic = traff.onehot_encoding(dataset['topic'])
#onehot_source = traff.onehot_encoding(dataset['source'])


X=dataset.iloc[:, 8:10].values
X=np.concatenate((onehot_topic, X),axis=1)
y=dataset.iloc[:, 9:12].sum(axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

print("Normalizing values")
#Standardisation
std= StandardScaler()
x_train= std.fit_transform(x_train)
x_test=std.transform(x_test)

from src.regressors.factories import Hosts

host_list = [
        Hosts.linear_svr_host_C,
        Hosts.kernel_ridge_host_alpha,
        Hosts.kernel_ridge_host_gamma,
        Hosts.decision_tree_host,
        Hosts.gaussian_svr_host_C,
        Hosts.poly_svr_host_C,
        Hosts.poly_svr_host_d
    ]

def execute_host_search(host):
        host.do_search(x_train, y_train)
        host.do_test(x_test, y_test)
        host.plot_search("Accuracy per parameters for news popularity")

for host in host_list:
        try:
            execute_host_search(host)
        except Exception as e:
            print("HOST FAILED")
            print(host.regressor_factory)
            print(e)
