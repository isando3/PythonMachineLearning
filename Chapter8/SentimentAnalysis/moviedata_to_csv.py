# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:29:29 2016

@author: isando3
"""

import pyprind 
import pandas as pd
import numpy as np
import os 


pbar   = pyprind.ProgBar(50000)#visualize progress

#get data file in df and shuffle entries then save it in CSV file 
labels = {'pos':1,'neg':0}
df     = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' %(s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r') as infile:
                txt =infile.read()
            df = df.append([[txt,labels[l]]],ignore_index=True)
    pbar.update()
df.columns = ['review','sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
                