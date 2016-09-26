# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 18:05:58 2016

@author: isando3
"""

import pandas as pd 
import perceptron 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


##definitions##
def plot_decision_regions(X,y, classifier, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap =ListedColormap(colors[:len(np.unique(y))])
    x1_min , x1_max = X[:,0].min() -1 , X[:,0].max()+1
    x2_min , x2_max = X[:,1].min() -1 , X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min,x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],alpha=0.8, c=cmap(idx),marker=markers[idx],label=cl)

#read from CSV file and save data into pandas data frame

df = pd.read_csv('iris.data.txt', header=None)
#print df
#make the labels (y) 1 if setosa ,  -1 otherwise and the data (X) first and second variables of the dataset

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
#print y 
X = df.iloc[0:100, [0,2]].values
print'X:', X


#visualize the dataset
plt.scatter(X[:50,0],X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100,0],X[50:100,1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
plt.savefig('dataset_visualization.png')
plt.close()


#Decleare perceptron 
ppn  = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
#Check how many iterations are needed
plt.plot(range(1,len(ppn.errors_)+1) , ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
plt.savefig('epochs_vs_misclassif.png')
plt.close()

#Check the decision surfaces of the classifier
plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()
plt.savefig('decision_surface.png')
plt.close()


