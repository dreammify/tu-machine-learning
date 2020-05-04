import time
from sklearn.svm import SVR

class LinearWrapper:
    def __init__(self, degree=3, gamma='scale', C=1.0, max_iter=-1):
        self.regressor = SVR(degree=degree, gamma=gamma, C=C, max_iter=max_iter)
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
