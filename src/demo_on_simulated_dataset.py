"""
This is a demo file to launch my implementation of ElasticNet CV
on SIMULATED data(Hitters)
Output:
Prints mean squared error and optimal lambda from cross validation
2 plots: cross validation plot(MSE vs lambda) and 
         convergence plot(objective values vs iterations)
"""

# import the python file where ElasticNet is implemented
import Elastic_Net

# Set the variables for function call
max_iter = 100
ALPHA = 0.9

#Call method from Elastic_Net to get simulated data
X_train, X_test, y_train, y_test = Elastic_Net.get_simulated_data()

#Call method from Elastic_Net to plot and print MSE and optimal lambda
lamda_opt = Elastic_Net.cross_validate(X_train, y_train, X_test, y_test, max_iter, ALPHA)

#Call method from Elastic_Net to plot objective values vs iterations
Elastic_Net.convergence_plot(X_train, y_train, max_iter, ALPHA, lamda_opt)

