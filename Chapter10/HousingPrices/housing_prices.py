# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:41:50 2016

@author: isando3
"""

import pandas as pd 
import numpy as np

#grab data
df = pd.read_csv('housing.data',header=None, sep='\s+')
df.columns = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print df.head()
import matplotlib.pyplot as plt
import seaborn as sns
#Exploratory data analysis (EDA) using seaborn library 
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()
plt.savefig('eda_housing.png')
plt.close()
#look for correlation between variables, using correlation matrix
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
#visualize with heatmap
hm = sns.heatmap(cm,cbar=True,annot= True, fmt='.2f', annot_kws={'size': 15},yticklabels=cols, xticklabels=cols)
#, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},yticklabels=cols,xticklables=cols)
plt.show()
plt.savefig('correlation_heatmap_housing.png')
plt.close()


