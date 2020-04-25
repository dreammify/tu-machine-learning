import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def visualize_class(frame: DataFrame):
    value_counts : pd.Series = frame["recclass"].value_counts()
    value_counts = value_counts[value_counts > 100]
    value_counts.plot.bar()

    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.show()

def visualize_years(frame: DataFrame):
    bins = np.arange(1750, 2025, 3)
    temp_frame = frame[["year"]]
    temp_frame = temp_frame[temp_frame["year"].map(np.isreal)]

    sns.distplot(temp_frame["year"].values, bins= bins, kde_kws={'bw': 1.1}, hist_kws=dict(ec="k"))
    plt.xlim(1750,2025)
    plt.show()
    plt.clf()

def visualize_mass(frame: DataFrame):
    temp_frame = frame[["mass"]]
    temp_frame["mass"] = temp_frame["mass"].map(np.log)
    temp_frame = temp_frame[temp_frame["mass"].map(np.isfinite)]

    sns.distplot(temp_frame["mass"].values, kde_kws={'bw': 1.1}, hist_kws=dict(ec="k"))
    plt.show()
    plt.clf()

def load_meteorite_frame(filename: str):
    with open(filename, encoding='utf-8') as csv_file:
        data: DataFrame = pd.read_csv(csv_file)

        print(data.head(3))
        return data

if __name__ == "__main__":
    data = load_meteorite_frame("resources/meteorite-landings.csv")
    visualize_mass(data)
    visualize_years(data)
    visualize_class(data)