# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:46:29 2016

@author: isando3
"""

import pickle 
import re
import os 
from vectorizer import vect
import numpy as np

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'movieclassifier','pkl_objects','classifier.pkl'),'rb'))
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print 'Prediction:', label[clf.predict(X)[0]]
print 'Probability', np.max(clf.predict_proba(X))

example2 = ['This movie was very bad, awful, terrible. It sucks!']
X2 = vect.transform(example2)
print 'Prediction:', label[clf.predict(X2)[0]]
print 'Probability', np.max(clf.predict_proba(X2))