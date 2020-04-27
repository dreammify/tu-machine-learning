# Start with simple regression for some time series point
import numpy
import pandas

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
    print("Training SVR")
    svm.train(train_data, train_labels)

    print("Train Score")
    print(svm.score(train_data, train_labels))

    print("Test Score")
    print(svm.score(test_data, test_labels))

def svr_hint_only(data, mask, hint_time):
    print("Performing TF-IDF on dataset")
    train_hint, test_hint = generate_targets(data, mask, 'TS' + str(70 - hint_time))
    train_hint = train_hint.to_numpy().reshape(-1, 1)
    test_hint = test_hint.to_numpy().reshape(-1, 1)
    train_targets, test_targets = generate_targets(data, mask, 'TS70')

    svm = SVMWrapper(c=1, e=0.0, loss="epsilon_insensitive", dual=True, max_iter=10000)
    train_and_score(svm, train_hint, test_hint, train_targets, test_targets)


def svr_headline_only(data, mask):
    print("Performing TF-IDF on dataset")
    train_vectors, test_vectors = generate_idf_vectors(data, mask, 'Title')
    train_targets, test_targets = generate_targets(data, mask, 'TS70')

    svm = SVMWrapper(c=1, e=0.0, loss="epsilon_insensitive", dual=True, max_iter=10000)
    train_and_score(svm, train_vectors, test_vectors, train_targets, test_targets)

if __name__ == "__main__":
    print("Loading data")
    data = load_news_frame_pop(
        filename="loaders/resources/news.csv",
        popfilenames=["loaders/resources/news-fb-timeseries-economy.csv",
                      "loaders/resources/news-fb-timeseries-microsoft.csv",
                      "loaders/resources/news-fb-timeseries-obama.csv",
                      "loaders/resources/news-fb-timeseries-palestine.csv"])

    print("Generating train/test mask")
    mask = numpy.random.rand(len(data)) < 0.8
    svr_headline_only(data, mask)