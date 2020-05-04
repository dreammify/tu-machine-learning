from concurrent.futures.thread import ThreadPoolExecutor

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
import src.loaders.preprocessing_traffic as traffic
import src.regressors.factories as factories
from src.regressors.factories import Hosts
from src.utilities.plot_param import ParameterSearchHost

if __name__ == "__main__":
    dataset = traffic.load_data()
    x_train, x_test, y_train, y_test = traffic.preprocess_data(dataset)

    host_list = [
        Hosts.decision_tree_host,
        Hosts.logistic_regression_host_C,
        Hosts.logistic_regression_host_penalty,
        Hosts.linear_svr_host_C,
        Hosts.gaussian_svr_host_C,
        Hosts.gaussian_svr_host_d
    ]

    def execute_host_search(host):
        host.do_search(x_train, y_train)
        host.do_test(x_test, y_test)
        host.plot_search("Accuracy per parameters for Traffic Volume")

    with ThreadPoolExecutor(max_workers=2) as executor:
        for host in host_list:
            print(host.regressor_factory)
            try:
                executor.submit(execute_host_search, host)
            except Exception:
                pass

