# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 18:05:58 2016

@author: isando3
"""

import pandas as pd 
import perceptron 
import matplotlib.pyplot as plt
import numpy as np

#read from CSV file and save data into pandas data frame

df = pd.read_csv('iris.data.txt', header=None)
#print df
#make the labels (y) 1 if setosa ,  -1 otherwise and the data (X) first and second variables of the dataset

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
#print y 
X = df.iloc[0:100, [0,2]].values
print'X:', X


#Decleare perceptron 

ppn  = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1) , ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
