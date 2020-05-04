from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from src.regressors.parameters import ParameterSearchHost

def decision_tree_depth(depth):
    return DecisionTreeRegressor(max_depth=depth)

def logistic_regression_penalty(penalty):
    return LogisticRegression(penalty=penalty)

def logistic_regression_C(c_value):
    return LogisticRegression(C=c_value)

def linear_svm_C(c_value):
    return LinearSVR(C=c_value)

def gaussian_svm_C(c_value):
    return SVR(C=c_value)

def gaussian_svm_Degree(degree):
    return SVR(degree=degree)

class Hosts:
    decision_tree_host = ParameterSearchHost(
        parameters=range(1,50),
        regressor_factory=decision_tree_depth
    )

    logistic_regression_host_penalty = ParameterSearchHost(
        parameters=['l1', 'l2', 'elasticnet', 'none'],
        regressor_factory=logistic_regression_penalty
    )

    logistic_regression_host_C = ParameterSearchHost(
        parameters=[0.01, 0.1, 1, 10, 100],
        regressor_factory=logistic_regression_C
    )

    linear_svr_host_C = ParameterSearchHost(
        parameters=[0.01, 0.1, 1, 10, 100],
        regressor_factory=linear_svm_C
    )

    gaussian_svr_host_C = ParameterSearchHost(
        parameters=[0.01, 0.1, 1, 10, 100],
        regressor_factory=gaussian_svm_C
    )

    gaussian_svr_host_d = ParameterSearchHost(
        parameters=range(1,20),
        regressor_factory=gaussian_svm_Degree
    )