import time
from sklearn.svm import LinearSVC, LinearSVR


class SVMWrapper:
    def __init__(self, c=1.0, e=0.0, loss="epsilon_insensitive", dual=True, max_iter=1000):
        self.regressor = LinearSVR(C=c, epsilon=e, loss=loss, dual=dual, max_iter=max_iter)
        self.training_time = None

    def train(self, x_train, y_train):
        start = time.perf_counter()
        self.regressor.fit(x_train, y_train)
        self.training_time = time.perf_counter() - start

    def score(self, x_test, y_test):
        return self.regressor.score(x_test, y_test)

    def predict(self, x_test):
        return self.regressor.predict(x_test)

    def predict_one(self, x_single):
        return self.regressor.predict(x_single)

    def get_training_time(self):
        if self.training_time is None:
            raise ValueError()
        else:
            return self.training_time
