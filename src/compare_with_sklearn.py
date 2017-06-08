"""
This module performs comparison between my implementation and 
scikit-learns ElasticNetCV on a real world dataset(Hitters)
"""

import Elastic_Net
from sklearn.linear_model import ElasticNet, ElasticNetCV
import numpy as np

# Set the global variables
MAX_ITER = 100
ALPHA = 0.9

def sklearn_elastic_net(X, y, lamda, alpha):
    """
    Method to convert lambda and alpha and get beta from sklearn
    :param X: Feature matrix 
    :param y: Response vector
    :param lamda: regularization parameter
    :param alpha: Constant that multiplies the penalty terms
    :return: betas(learning parameters)
    """
    l1ratio_sklearn = alpha / (2 - alpha)
    alpha_sklearn = lamda * (1 - 1 / 2 * alpha)
    elastic_net = ElasticNet(alpha=alpha_sklearn, l1_ratio=l1ratio_sklearn, fit_intercept=False, max_iter=10000,
                             tol=0.000001).fit(X, y)
    beta_star = elastic_net.coef_
    return beta_star

def sklearn_elastic_net_cv(X, y, alpha):
    """
    Method to convert lambda and alpha and get optimal lambda from sklearn
    :param X: Feature matrix 
    :param y: Response vector
    :param alpha: Constant that multiplies the penalty terms
    :return: optimal lambda
    """
    l1ratio_sklearn = alpha / (2 - alpha)
    elastic_net = ElasticNetCV(l1_ratio=l1ratio_sklearn, fit_intercept=False, max_iter=10000, tol=0.000001).fit(X, y)
    lambda_opt = elastic_net.alpha_ / (1 - 1 / 2 * alpha)
    return lambda_opt

#Call method from Elastic_Net to get real world data
X_train, X_test, y_train, y_test = Elastic_Net.get_real_data()

#Initialize beta
beta_init = np.zeros(np.size(X_train, 1))

#Get optimal lambda from sklearn
lambda_opt_s = sklearn_elastic_net_cv(X_train,y_train,ALPHA)

#Get optimal lambda from my ElasticNet cross validation
lambda_opt_t = Elastic_Net.cross_validate(X_train, y_train, X_test, y_test, MAX_ITER, ALPHA)

#Print both the lambdas
print('Optimal lamba from sklearn:',lambda_opt_s )
print('Optimal lamba from my implementation:',lambda_opt_t )

#Get betas from sklearn
beta_sklearn = sklearn_elastic_net(X_train, y_train, lambda_opt_s, ALPHA)

#Get betas from my ElasticNet cyclic coordinate descent
beta_myalgo = Elastic_Net.cycliccoorddescent(beta_init, X_train, y_train, lambda_opt_t, MAX_ITER, ALPHA)

#Get mean squared error for betas from sklearn
mse_sklearn = Elastic_Net.compute_mse(X_test, y_test, beta_sklearn)

#Get mean squared error for betas from my ElasticNet cyclic coordinate descent
mse_myalgo = Elastic_Net.compute_mse(X_test, y_test, beta_myalgo[-1])

#Print both the MSEs
print('MSE with sklearn:', mse_sklearn)
print('MSE with my implementation:', mse_myalgo)