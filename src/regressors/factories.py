from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from src.regressors.parameters import ParameterSearchHost

def decision_tree_depth(depth):
    return DecisionTreeRegressor(max_depth=depth)

def kernel_ridge_alpha(alpha):
    return KernelRidge(kernel='rbf', gamma=0.1, alpha=alpha)

def kernel_ridge_gamma(gamma):
    return KernelRidge(kernel='rbf', gamma=gamma, alpha=0.001)

def linear_svm_C(c_value):
    return LinearSVR(C=c_value, max_iter=100000)

def gaussian_svm_C(c_value):
    return SVR(C=c_value)

def gaussian_svm_Degree(degree):
    return SVR(degree=degree)

class Hosts:
    decision_tree_host = ParameterSearchHost(
        parameters=range(1,50),
        regressor_factory=decision_tree_depth
    )

    kernel_ridge_host_alpha = ParameterSearchHost(
        parameters=[10, 1e0, 0.1, 0.01, 0.001, 0.0001],
        regressor_factory=kernel_ridge_alpha,
        plot_type="log"
    )

    kernel_ridge_host_gamma = ParameterSearchHost(
        parameters=[10, 1e0, 0.1, 0.01, 0.001, 0.0001],
        regressor_factory=kernel_ridge_gamma,
        plot_type="log"
    )

    linear_svr_host_C = ParameterSearchHost(
        parameters=[0.01, 0.1, 1, 10, 100],
        regressor_factory=linear_svm_C,
        plot_type="log"
    )

    gaussian_svr_host_C = ParameterSearchHost(
        parameters=[0.01, 0.1, 1, 10, 100],
        regressor_factory=gaussian_svm_C,
        plot_type="log"
    )

    gaussian_svr_host_d = ParameterSearchHost(
        parameters=range(1,20),
        regressor_factory=gaussian_svm_Degree
    )