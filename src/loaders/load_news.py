import numpy
import pandas
from typing import List
from pandas import DataFrame

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

def load_news_frame_pop(filename: str, popfilenames: List[str]):
    with open(filename, encoding='utf-8') as json_file:
        data: DataFrame = pandas.read_csv(json_file)[[
            'IDLink',
            'Title',
            'Headline',
            'Source',
            'Topic',
            'PublishDate'
        ]]

        data = data.astype({"IDLink": int})
        basecolumns = data.columns

        for filename in popfilenames:
            popdata = pandas.read_csv(filename)
            popdata = popdata.astype({"IDLink": int})
            if popfilenames.index(filename) == 0:
                data = pandas.merge(data, popdata, on=["IDLink"], how="left")
            else:
                merged = pandas.merge(data[basecolumns], popdata, on=["IDLink"], how="left")
                data.fillna(merged)

        data = data.dropna().sort_values(by="IDLink")
        return data

if __name__ == "__main__":
    load_news_frame_pop(filename = "resources/news.csv", popfilenames=["resources/news-fb-timeseries-economy.csv", "resources/news-fb-timeseries-microsoft.csv", "resources/news-fb-timeseries-obama.csv", "resources/news-fb-timeseries-palestine.csv"])