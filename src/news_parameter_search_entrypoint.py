import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.loaders.load_news import load_news_frame_pop
from src.regressors.factories import Hosts

def get_train_test_split_with_average_popularity(data):
    ts_col_names = []
    for i in range(1, 145):
        ts_col_names.append("TS" + str(i))

    avg_cols = data[ts_col_names].mean(axis=1)
    data['avg'] = avg_cols

    train_frame = data[mask]
    test_frame = data[~mask]

    tokenizer = TfidfVectorizer(strip_accents='unicode', min_df=10)
    tokenizer.fit(train_frame["Headline"])

    train_tokens = tokenizer.transform(train_frame["Headline"]).toarray()
    test_tokens = tokenizer.transform(test_frame["Headline"]).toarray()

    train_labels = train_frame["avg"].to_numpy()
    test_labels = test_frame["avg"].to_numpy()

    return train_tokens, test_tokens, train_labels, test_labels


if __name__ == "__main__":
    # Load data
    print("Loading data")
    data = load_news_frame_pop(
        filename="loaders/resources/news.csv",
        popfilenames=["loaders/resources/news-fb-timeseries-economy.csv",
                      "loaders/resources/news-fb-timeseries-microsoft.csv",
                      "loaders/resources/news-fb-timeseries-obama.csv",
                      "loaders/resources/news-fb-timeseries-palestine.csv"]).head(10000)

    # Generate train-test split
    print("Generating train/test mask")
    mask = numpy.random.rand(len(data)) < 0.8

    x_train, x_test, y_train, y_test = get_train_test_split_with_average_popularity(data)

    host_list = [
        Hosts.decision_tree_host,
        Hosts.kernel_ridge_host_alpha,
        Hosts.kernel_ridge_host_gamma,
        Hosts.linear_svr_host_C,
        Hosts.gaussian_svr_host_C,
        Hosts.poly_svr_host_C,
        Hosts.poly_svr_host_d
    ]

    def execute_host_search(host):
        host.do_search(x_train, y_train)
        host.do_test(x_test, y_test)
        host.plot_search("Accuracy per parameters for Traffic Volume")

    for host in host_list:
        try:
            execute_host_search(host)
        except Exception as e:
            print("HOST FAILED")
            print(host.regressor_factory)
            print(e)

