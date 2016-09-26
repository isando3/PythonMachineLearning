# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 22:11:30 2016

@author: isando3
"""

import pandas as pd
import adaline_perceptron
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


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

df = pd.read_csv('iris.data.txt', header=None)
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
#print y
X = df.iloc[0:100, [0,2]].values
print X.shape

#checking which learning rate is better
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
ada1 = adaline_perceptron.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-sq-error)')
ax[0].set_title('Adaline-Learning rate 0.01')
ada2 = adaline_perceptron.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1), np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-sq-error)')
ax[1].set_title('Adaline-Learning rate 0.0001')
plt.show()
plt.savefig("CompareLearningRate.png")
plt.close()
# standarizing the data

X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

#after standarization we check again the convergence 
ada = adaline_perceptron.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ada.fit(X_std,y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline- Gradient Descent')
plt.xlabel('sepal length [standard]')
plt.ylabel('petal length [standard]')
plt.legend(loc='upper left')
plt.savefig('DecisionSurface_standarized.png')
plt.show()
plt.close()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Sum-sq-error')
plt.show()
plt.savefig('StandarizedLearningRate.png')
