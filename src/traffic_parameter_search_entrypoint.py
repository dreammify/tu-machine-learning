from concurrent.futures.thread import ThreadPoolExecutor

import src.loaders.preprocessing_traffic as traffic
from src.regressors.factories import Hosts

if __name__ == "__main__":
    dataset = traffic.load_data().head(5000)
    x_train, x_test, y_train, y_test = traffic.preprocess_data(dataset)

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
        host.plot_search("Accuracy per parameters for traffic volume")

    for host in host_list:
        try:
            execute_host_search(host)
        except Exception as e:
            print("HOST FAILED")
            print(host.regressor_factory)
            print(e)

