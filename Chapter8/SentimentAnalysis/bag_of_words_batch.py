# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 18:03:34 2016

@author: isando3
"""

import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
import pyprind
pbar = pyprind.progbar.ProgBar(45)


def tokenizer(text):
    text = re.sub('<[^>]*','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path,'r') as csv:
        next(csv)
        for line in csv:
            text,label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream,size):
    docs, y = [],[]
    try:
        for _ in range(size):
            text,label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs,y
    
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None, tokenizer=tokenizer)
doc_stream = stream_docs(path='./movie_data.csv')
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream,size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream,size=5000)
X_test = vect.transform(X_test)
print ('Accuracy: %3f' % clf.score(X_test,y_test))
clf=clf.partial_fit(X_test,y_test)
print ('Accuracy: %3f' % clf.score(X_test,y_test))

# serializing the estimator
import pickle
import os
dest = os.path.join('movieclassifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=2)
pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=2)
 

    