# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 17:48:41 2016

@author: isando3
"""

import numpy as np
'''
eta:        learning rate
w_ :        1-d array with weights after fitting
errors_ :   1-d array with number of misclassifications in every epoch
'''


class Perceptron(object):
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta =eta
        self.n_iter= n_iter
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]+self.w_[0])
        
    def predict(self, X):
        return np.where(self.net_input(X) >=0.0, 1, -1)
    
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for i in range(self.n_iter):
            errors = 0 
            for xi, target in zip(X,y):
                #print xi, target
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            
        return self
        