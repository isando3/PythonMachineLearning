# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:39:34 2016

@author: isando3
"""

import pandas as pd
import adaline_sgd
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
X = df.iloc[0:100, [0,2]].values

# standarizing the data
X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
ada = adaline_sgd.AdalineSGD(n_iter=15,eta=0.01, random_state=1)
ada.fit(X_std,y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline- Stochastic Gradient Descent')
plt.xlabel('sepal length [standard]')
plt.ylabel('petal length [standard]')
plt.legend(loc='upper left')
plt.savefig('DecisionSurface_Standarized_SGDAdaline.png')
plt.show()
plt.close()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Sum-sq-error')
plt.show()
plt.savefig('StandarizedLearningRate.png')



        