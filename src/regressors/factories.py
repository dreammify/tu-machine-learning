from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

def decision_tree_factory(depth):
    return DecisionTreeRegressor(max_depth=depth)

def logistic_regression_factory_penalty(penalty):
    return LogisticRegression(penalty=penalty)

def logistic_regression_factory_C(c_value):
    return LogisticRegression(C=c_value)

def linear_svr_factory_C(c_value):
    return LinearSVR(C=c_value)

def gaussian_svr_factory_C(c_value):
    return SVR(C=c_value)

def gaussian_svr_factory_d(degree):
    return SVR(degree=degree)