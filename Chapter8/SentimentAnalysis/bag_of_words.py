# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:19:48 2016

@author: isando3
"""

#import numpy as np 
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def preprocessor(text):
    text= re.sub('<[^>]*>','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

def tokenizer(text):
    return text.split()
    
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
    
from nltk.corpus import stopwords
stop = stopwords.words('english')    

df = pd.DataFrame()
df = pd.read_csv('./movie_data.csv')
print df.head(3) 
X_train = df.loc[:2500, 'review'].values
X_test = df.loc[2500:, 'review'].values
y_train = df.loc[:2500,'sentiment'].values
y_test = df.loc[2500:, 'sentiment'].values


from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid =[{'vect__ngram_range':[(1,1)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenizer_porter],
              'clf__penalty': ['l1','l2'],
              'clf__C':[1.0,10.0,100.0]},
             {'vect__ngram_range':[(1,1)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenizer_porter],
              'vect__norm':[None],
              'clf__penalty': ['l1','l2'],
              'clf__C':[1.0,10.0,100.0]},
            ]
lr_tfidf = Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=1)
gs_lr_tfidf.fit(X_train,y_train)
print ('Best parameter set: %s' % gs_lr_tfidf.best_params_)
print ('CV Accuracy: %3f' % gs_lr_tfidf.best_score_)




