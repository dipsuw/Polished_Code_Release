import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def min_coord(j,X,y,lambduh,beta,alpha):
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

# Testing
d=np.size(X_scaled, 1)
n=np.size(X_scaled, 0)
beta_init = np.ones(d)
#min_coord(0, X_scaled, Y_scaled, 0.1, beta_init ,alpha=0.9)

def get_data():
    hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)
    hitters = hitters.dropna()
    X = hitters.drop('Salary', axis=1)
    Y = hitters.Salary
    X.head()
    X = pd.get_dummies(X, drop_first=True)
    my_scaleX  = sklearn.preprocessing.StandardScaler().fit(X)
    X_scaled = my_scaleX.transform(X)
    Y_scaled = (Y - np.mean(Y))/np.std(Y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_scaled, Y_scaled, random_state=0)
    return X_train, X_test, y_train, y_test

def computeobj(beta, X, y, lambduh, alpha):
    obj_val = 1/np.size(X, 0)*sum((y-np.dot(X, beta))**2) + lambduh*alpha*sum(abs(beta)) + lambduh*(1-alpha)* np.linalg.norm(beta)**2
    return obj_val

#computeobj(beta_init, X_scaled, Y_scaled, lambduh=1, alpha=0.9)

def cycliccoorddescent(beta, X, y, lambduh, max_iter, alpha):
    iter = 0
    d = np.size(X, 1)
    all_betas = beta
    while iter<max_iter:
        j = np.remainder(iter, d)
        beta[j] = min_coord(j, X,y, lambduh, beta, alpha)
        all_betas = np.vstack((all_betas, beta))
        iter = iter+1
    return all_betas

#cycliccoorddescent(beta_init, X_scaled, Y_scaled, lambduh=0.1, max_iter=100, alpha=0.9)

def pickcoord(X):
    n = X.shape[1]
    coord = np.random.randint(0,n-1)
    return coord

def randcoorddescent(beta, X, y, lambduh, max_iter, alpha):
    d = X.shape[1]
    all_betas = beta
    for i in range(max_iter*d):
        j = pickcoord(X)
        beta[j] = min_coord(j, X, y, lambduh, beta, alpha)
        if i % d == 0:
            all_betas = np.vstack((all_betas, beta))
    return all_betas

#randcoorddescent(beta_init, X_scaled, Y_scaled, lambduh=0.1, max_iter=10, alpha=0.9)

def cross_validate():
    my_lams = (10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4)
    my_mse_all = []

    for i in my_lams:
        betas_t = cycliccoorddescent(beta_init, X_train, y_train, i, max_iter=100, alpha=0.9)
        my_mse = np.mean(np.square(y_test - X_test.dot(betas_t[-1])))
        my_mse_all = np.append(my_mse_all, my_mse)

    plt.plot(np.log(my_lams), my_mse_all, color='pink', label='MSE from my coordinate decsent')
    plt.xlabel('log of lambda values')
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    plt.show()

    print('Smallest MSE value:', min(my_mse_all), 'at lambda =', my_lams[np.argmin(my_mse_all)])
