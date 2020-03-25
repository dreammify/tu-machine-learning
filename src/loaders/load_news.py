# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:27:49 2020

@author: Charlotte
"""

from pandas import read_csv
from matplotlib import pyplot

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv"
names = ['IDLink','title', 'headline', 'source', 'topic', 'publishDate', 'SentimentTitle', 'SentimentHeadline', 'Facebook', 'GooglePlus','LinkedIn']
type_dict = {'IDLink': 'float64', 'title':'str','headline':'str','source':'str','topic':'str'
             ,'SentimentTitle':'float16', 'SentimentHeadline':'float16'
             ,'Facebook':'int','GooglePlus':'int16','LinkedIn':'int16'}
dataset = read_csv(url, names=names, dtype=type_dict, header=0)

#missing values?
dataset.columns[dataset.isnull().any()]

print(dataset.describe())

#plots
dataset['SentimentHeadline'].hist()
pyplot.show()
dataset['SentimentTitle'].hist()
pyplot.show()

dataset['Facebook'].hist()
pyplot.show()

dataset['GooglePlus'].hist()
pyplot.show()

dataset['LinkedIn'].hist()
pyplot.show()

#Topic:
fracs = dataset.groupby('topic').size().tolist()

labels = 'economy', 'Microsoft', 'Obama', 'Palestine'

fig = pyplot.figure()
ax1 = fig.add_subplot(212)
ax1.axis('equal')
ax1.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)



        