import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from wordcloud import WordCloud


def visualize_word_cloud(frame: DataFrame):
    text = data.text.values
    wordcloud = WordCloud(
        width=1920,
        height=1080,
        background_color='black',
        collocations=False
    ).generate(' '.join(text))

    plt.figure(
        figsize=(19, 11),
        facecolor='k',
        edgecolor='k'
    )

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def visualize_distributions(frame: DataFrame):
    temp_frame = frame[['power', 'toughness', 'textCount', 'convertedManaCost']]
    print(temp_frame.head(3))

    ax = sns.distplot(temp_frame.convertedManaCost.map(int).values, bins=np.arange(0, 16), kde_kws={'bw': 1},
                 hist_kws=dict(ec="k"))
    ax.set(xlabel='Total mana cost', ylabel='Distribution')
    plt.show()
    plt.clf()
    ax = sns.distplot(temp_frame.textCount.map(int).values, kde_kws={'bw': 1}, hist_kws=dict(ec="k"))
    ax.set(xlabel='Total words in text', ylabel='Distribution')
    plt.show()
    plt.clf()


def visualize_text_cost(frame: DataFrame):
    temp_frame = frame[['convertedManaCost', 'textCount']]
    ax = sns.boxplot(x="convertedManaCost", y="textCount", data=temp_frame)
    ax.set(xlabel='Total mana cost', ylabel='Total words in text')
    plt.show()
    plt.clf()


def load_mtg_frame(filename: str):
    with open(filename, encoding='utf-8') as json_file:
        data: DataFrame = pd.read_json(json_file, orient='index')[
            ['name', 'text', 'manaCost', 'convertedManaCost', 'colors', 'types', 'supertypes', 'subtypes', 'power',
             'toughness', 'legalities']]

        # Normalize the indices to numerical
        act_index = 0
        index_dict = {}
        for index, _ in data.iterrows():
            index_dict[index] = act_index
            act_index = act_index + 1

        data.rename(index=index_dict, inplace=True)

        # Replace linebreaks with spaces
        def replace_linebreaks(text: str):
            if pd.isna(text):
                return ""
            else:
                return text.replace("\n", " ")

        data['text'] = data.text.apply(replace_linebreaks)

        # Add text word count column
        def count_words(text: str):
            return len(text.split(" "))

        data["textCount"] = data.text.apply(count_words)

        # Remove cards that are not actually legal
        def numkeys(d: dict):
            return len(d.keys())

        data = data[data['legalities'].map(numkeys) > 0]

        print(len(data))

        return data


if __name__ == "__main__":
    data = load_mtg_frame("resources/all-cards.json")
    visualize_text_cost(data)
    visualize_distributions(data)
    visualize_word_cloud(data)
