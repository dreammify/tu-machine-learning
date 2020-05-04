from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ParameterSearchHost:
    def __init__(self, parameters, regressor_factory, scale=True, cv=10):
        # Keep trained models here
        self.regressor_list = []
        # Make models with the regressor factory provided, this should be a
        # a function with a single parameter that returns a classifier
        self.regressor_factory = regressor_factory
        # List of parameters to use in search
        self.parameters_to_search = parameters
        # Whether to scale the data
        self.scale = scale
        # How many crossvalidation folds to do
        self.cv = cv

        # Whether this host has already been trained
        self._trained = False

        # Output data
        self.cv_scores_list = []
        self.cv_scores_std = []
        self.cv_scores_mean = []

        self.train_scores = []
        self.test_scores = []

        self.cv_mean = None
        self.cv_std = None

    def do_search(self, x_train, y_train):
        if self._trained:
            raise Exception("Host already trained")

        for p in self.parameters_to_search:
            print("Training model for parameter: " + str(p))
            model = self.regressor_factory(p)

            if self.scale:
                clf = make_pipeline(StandardScaler(), model)
            else:
                clf = make_pipeline(model)

            cv_scores = cross_val_score(clf, x_train, y_train, cv=self.cv)

            self.cv_scores_list.append(cv_scores)
            self.cv_scores_mean.append(cv_scores.mean())
            self.cv_scores_std.append(cv_scores.std())
            self.train_scores.append(model.fit(x_train, y_train).score(x_train, y_train))

        self.cv_scores_mean = np.array(self.cv_scores_mean)
        self.cv_scores_std = np.array(self.cv_scores_std)
        self.train_scores = np.array(self.train_scores)
        return self.cv_scores_mean, self.cv_scores_std, self.train_scores

    def do_test(self, x_train, y_train):
        pass

    def plot_search(self, title):
        fig, ax = plt.subplots(1,1, figsize=(15,5))
        ax.plot(self.parameters_to_search, self.cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
        ax.fill_between(self.parameters_to_search, self.cv_scores_mean-2*self.cv_scores_std, self.cv_scores_mean+2*self.cv_scores_std, alpha=0.2)
        ylim = plt.ylim()
        ax.plot(self.parameters_to_search, self.train_scores, '-*', label='train accuracy', alpha=0.9)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Tree depth', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_ylim(ylim)
        ax.set_xticks(self.parameters_to_search)
        ax.legend()
        plt.show()