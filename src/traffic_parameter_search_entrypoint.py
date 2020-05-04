from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
import src.loaders.preprocessing_traffic as traffic
import src.regressors.factories as factories
from src.utilities.plot_param import ParameterSearchHost

if __name__ == "__main__":
    dataset = traffic.load_data()
    x_train, x_test, y_train, y_test = traffic.preprocess_data(dataset)

    print("Preparing decision trees")
    sm_tree_depths = range(1,20)
    host = ParameterSearchHost(
        parameters=range(1,50),
        regressor_factory=factories.decision_tree_factory,
        scale=True,
        cv=10
    )

    host.do_search(x_train, y_train)
    host.do_test(x_test, y_test)

    host.plot_search("Accuracy per decision tree depth for Traffic Volume")
