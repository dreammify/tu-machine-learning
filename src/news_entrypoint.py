import gc

import numpy
import pandas
import matplotlib.pyplot as plt

from src.data.idf_dataset import generate_idf_vectors
from src.loaders.load_news import load_news_frame_pop
from src.regressors.support_vector_machine import SVMWrapper

def generate_targets(dataframe : pandas.DataFrame, mask, column: str):
    train_frame = dataframe[mask]
    test_frame = dataframe[~mask]

    train_targets = train_frame[column]
    test_targets = test_frame[column]
    return train_targets, test_targets

def train_and_score(svm, train_data, test_data, train_labels, test_labels):
    svm.train(train_data, train_labels)

    train_score = svm.score(train_data, train_labels)
    test_score = svm.score(test_data, test_labels)
    return train_score, test_score

def svr_hint_headline_progressive(data, mask, hint_time, classifier):
    print("Performing TF-IDF on dataset")
    train_vectors, test_vectors = generate_idf_vectors(data, mask, 'Title')
    train_targets, test_targets = generate_targets(data, mask, 'TS75')


    hint_train = []
    hint_test = []

    both_train = []
    both_test = []

    for i in hint_time:
        # Clean some garbage from the previous run
        gc.collect()

        print("Generating hint " + str(i))
        temp_trainv = train_vectors.toarray()
        temp_testv = test_vectors.toarray()

        train_hint, test_hint = generate_targets(data, mask, 'TS' + str(i))
        train_hint = train_hint.to_numpy().reshape(-1, 1)
        test_hint = test_hint.to_numpy().reshape(-1, 1)

        temp_trainv = numpy.concatenate((temp_trainv, train_hint), axis=1)
        temp_testv = numpy.concatenate((temp_testv, test_hint), axis=1)

        # Training classifier
        svm = SVMWrapper(c=1, e=0.0, loss="epsilon_insensitive", dual=True, max_iter=20000)
        rtrain, rtest = train_and_score(classifier, temp_trainv, temp_testv, train_targets, test_targets)
        both_train.append(rtrain)
        both_test.append(rtest)

        rtrain, rtest = train_and_score(classifier, train_hint, test_hint, train_targets, test_targets)
        hint_train.append(rtrain)
        hint_test.append(rtest)

    plt.plot(hint_train, 'ro')
    plt.plot(hint_test, 'r^')
    plt.plot(both_train, 'go')
    plt.plot(both_test, 'g^')
    plt.show()




if __name__ == "__main__":
    # Load data
    print("Loading data")
    data = load_news_frame_pop(
        filename="loaders/resources/news.csv",
        popfilenames=["loaders/resources/news-fb-timeseries-economy.csv",
                      "loaders/resources/news-fb-timeseries-microsoft.csv",
                      "loaders/resources/news-fb-timeseries-obama.csv",
                      "loaders/resources/news-fb-timeseries-palestine.csv"]).head(15000)

    # Generate train-test split
    print("Generating train/test mask")
    mask = numpy.random.rand(len(data)) < 0.8

    svr_hint_headline_progressive(
        data=data,
        mask=mask,
        hint_time=range(1, 74, 2),
        classifier = SVMWrapper(c=1, e=0.0, loss="epsilon_insensitive", dual=True, max_iter=10000)
    )