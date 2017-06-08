"""
The elastic net is a regularized regression method that linearly combines 
the L1 and L2 penalties of the lasso and ridge methods.
In this module, I have implemented in Python my own coordinate descent algorithm to solve
least-squares regression with elastic net regularization.

Implementation by Deepa Agrawal
deepa15@uw.edu
June 2017

References:    
https://github.com/cjones6/cubic_reg
Mid term homework and lab solutions
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import cross_validation

def min_coord(j,X,y,lambduh,beta,alpha):
    """
    Method to compute the solution of the partial minimization problem with respect to beta 
    :param j: Coordinate to optimize over
    :param X: Feature matrix
    :param y: Response vector
    :param lambduh: regularization parameter
    :param beta: learning parameter
    :param alpha: Constant that multiplies the penalty terms
    :return: updated value of beta
    """
    n = np.size(X,0)
    a = 2*np.sum(X**2, axis=0)
    aj = a[j]+2*lambduh*(1-alpha)
    xy = np.dot(X.T, y)
    cj = (2/n)*(xy[j] - np.sum(X[:,j]*(np.inner(X, beta) - X[:, j]*beta[j])))
    if cj < -lambduh*alpha:
        beta_j = (cj+lambduh*alpha)/aj
    elif cj > lambduh*alpha:
        beta_j = (cj-lambduh*alpha)/aj
    else:
        beta_j = 0
    return beta_j

def computeobj(beta, X, y, lambduh, alpha):
    """
    Method to calculate obejective value
    :param beta: learning parameter 
    :param X: Feature matrix
    :param y: Response matrix
    :param lambduh: regularization parameter
    :param alpha: Constant that multiplies the penalty terms
    :return: objective value for the elastic net function
    """
    obj_val = 1/np.size(X, 0)*sum((y-np.dot(X, beta))**2) + lambduh*alpha*sum(abs(beta)) + lambduh*(1-alpha)* np.linalg.norm(beta)**2
    return obj_val

def cycliccoorddescent(beta, X, y, lambduh, max_iter, alpha):
    """
    Method to implement cyclic coordinate descent algorithm
    :param beta: learning parameter
    :param X: feature matrix
    :param y: response matrix
    :param lambduh: regularization parameter
    :param max_iter: maximum number of iterations
    :param alpha: Constant that multiplies the penalty terms
    :return: betas for all coordinates
    """
    iter = 0
    d = np.size(X, 1)
    all_betas = beta
    while iter<max_iter:
        j = np.remainder(iter, d)
        beta[j] = min_coord(j, X,y, lambduh, beta, alpha)
        all_betas = np.vstack((all_betas, beta))
        iter = iter+1
    return all_betas

def cross_validate(X_train, y_train, X_test, y_test, max_iter, alpha):
    """
    Method to perform cross validation 
    :param X_train: train feature matrix
    :param y_train: train response matrix
    :param X_test: test feature matrix
    :param y_test: test response matrix
    :param max_iter: maximum number of iterations
    :param alpha: Constant that multiplies the penalty terms 
    :return: smallest optimal lambda
    """
    my_lams = (10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4)
    my_mse_all = []
    beta_init = np.zeros(X_train.shape[1])

    for i in my_lams:
        betas_t = cycliccoorddescent(beta_init, X_train, y_train, i, max_iter, alpha)
        my_mse = np.mean(np.square(y_test - X_test.dot(betas_t[-1])))
        my_mse_all = np.append(my_mse_all, my_mse)

    plt.plot(np.log(my_lams), my_mse_all, color='pink', label='MSE from my coordinate decsent')
    plt.xlabel('log of lambda values')
    plt.ylabel('MSE')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    print('Smallest MSE value:', min(my_mse_all), 'at lambda =', my_lams[np.argmin(my_mse_all)])
    return my_lams[np.argmin(my_mse_all)]

def convergence_plot(X_train, y_train, max_iter, alpha, lambda_opt_t):
    """
    Method to plot objective values vs number of iterations
    :param X_train: train feature matrix
    :param y_train: train response matrix
    :param max_iter: maximum number of iterations
    :param alpha: Constant that multiplies the penalty terms
    :param lambda_opt_t: optimal lambda acquired from cross validation 
    :return: none
    """
    beta_init = np.zeros(X_train.shape[1])
    betas_t = cycliccoorddescent(beta_init, X_train, y_train, lambda_opt_t, max_iter, alpha)
    my_vals = [computeobj(z, X_train, y_train, lambda_opt_t, alpha) for z in betas_t]
    plt.plot(range(len(my_vals)), my_vals, label='Cyclic coordinate descent', color='purple')
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective values')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    return

def compute_mse(X_test, y_test, betas_t):
    """
    Method to compute mean squared error
    :param X_test: Test feature matrix
    :param y_test: Test response matrix
    :param betas_t: learning parameter
    :return: mean squared error on test set
    """
    mse = np.mean((X_test.dot(betas_t) - y_test) ** 2)
    return mse

def get_real_data():
    """
    Method to get real world data
    :return: return the train and test data 
    """
    hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)
    hitters = hitters.dropna()
    X = hitters.drop('Salary', axis=1)
    Y = hitters.Salary
    X.head()
    X = pd.get_dummies(X, drop_first=True)
    my_scaleX  = sklearn.preprocessing.StandardScaler().fit(X)
    X_scaled = my_scaleX.transform(X)
    Y_scaled = (Y - np.mean(Y))/np.std(Y)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled,Y_scaled,test_size=0.25,random_state=42)
    return X_train, X_test, y_train, y_test

def get_simulated_data():
    """
    Method to get simulated data
    :return: returns the train and test data 
    """
    np.random.seed(25)
    x = np.random.normal(size=10)
    X = np.vstack((x, np.power(x, 2), np.power(x, 3)))
    beta3 = 0.01
    beta5 = 0.02
    Y = beta3 * (np.power(x, 3)) + beta5 * (np.power(x, 5))
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X.T,Y,test_size=0.25,random_state=42)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    return X_train, X_test, y_train, y_test