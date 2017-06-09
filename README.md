# Elastic Net CV
The elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods. The loss function is given as below. I used cyclic coordinate descent algorithm to implement this regression method.
![alt text][logo]

[logo]: https://github.com/dipsuw/Polished_Code_Release/blob/master/elasticnet_eq.PNG

In this repo, I have added following in 'src' directory:
### Elastic_Net.py: 
This is the source code file where I have implemented my own coordinate descent algorithm to solve least-squares regression with elastic net regularization.
### demo_on_real_dataset.py:
This is the demo file that allows the user to launch ElasticNet method on a real world dataset(Hitters). 
### demo_on_simulated_dataset.py:
This is the demo file that allows the user to launch ElasticNet method on a real world dataset(Hitters).
### compare_with_sklearn.py
This is the file that allows user to perform comparison between my implementation and scikit-learns ElasticNetCV on a real world dataset(Hitters)


