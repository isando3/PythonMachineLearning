# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:47:47 2016

@author: isando3
"""

import numpy as np
'''
eta : float (learning rate)
n_iter : int ( passes over training dataset)
w_ : array (of weights after fitting)
errors : array (misclassifications)
X : 2darray (shape = n_samples, n_features)
y : array (shape = n_samples, contains labels)

'''

class AdalineGD(object):
    def __init__(self, eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter 
        
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
        
    def activation(self, X):
        return self.net_input(X)
        
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def predict(self,X):
        return np.where(self.activation(X)>=0.0, 1,-1)        
        
    