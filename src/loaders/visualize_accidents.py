import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from pandas import DataFrame
from sklearn import preprocessing

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

def visualize_day_severity(frame: DataFrame):
    temp_frame = frame[['Day_of_Week', 'Accident_Severity']]
    temp_frame['Accident_Severity'] = temp_frame['Accident_Severity'].astype("int32")
    temp_frame['Day_of_Week'] = temp_frame['Day_of_Week'].astype("int32")
    temp_frame = temp_frame.groupby(['Day_of_Week', 'Accident_Severity'])\
        .size().reset_index().rename(columns={0:'count'})

    # Scaling, comment this out to get raw results
    x = temp_frame[['count']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    temp_frame['count'] = x_scaled

    temp_frame = temp_frame.pivot("Accident_Severity", "Day_of_Week", "count")
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(temp_frame, annot=True,  linewidths=.5, ax=ax)
    plt.show()


def visualize_dist(frame: DataFrame):
    temp_frame = frame[['Day_of_Week', 'Accident_Severity']]
    temp_frame['Accident_Severity'] = temp_frame['Accident_Severity'].astype("int32")
    temp_frame['Day_of_Week'] = temp_frame['Day_of_Week'].astype("int32")
    print(temp_frame.head(10))

    ax = sns.distplot(temp_frame.Day_of_Week.map(int).values, kde=False, bins=[0,1,2,3,4,5,6,7,8])
    ax.set(xlabel='Day of the Week', ylabel='Distribution')
    plt.show()
    plt.clf()

    ax = sns.distplot(temp_frame.Accident_Severity.map(int).values, kde=False, bins=[0,1,2,3,4])
    ax.set(xlabel='Accident Severity', ylabel='Distribution')
    plt.show()
    plt.clf()

def load_accidents_frame(filename: str):
    with open(filename, encoding='utf-8') as json_file:
        data: DataFrame = pandas.read_csv(json_file)[['Day_of_Week', 'Accident_Severity']] # Use dataframe indexing to get specific columns pd.read_csv(json_file)[[column, ...]]
        return data

if __name__ == "__main__":
    data = load_accidents_frame("resources/accidents_2005_to_2007.csv")
    # visualize_dist(data)
    visualize_day_severity(data)
