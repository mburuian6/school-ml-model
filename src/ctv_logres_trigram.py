#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:16:44 2020

@author: ian
"""

import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    df = pd.read_csv("../input/merged.csv")
    
    df["kfold"] = -1 
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.Insult.values
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
        
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        
        count_vec = CountVectorizer(tokenizer=word_tokenize,token_pattern=None,ngram_range=(1,3))
        
        count_vec.fit(train_df.Comment)
        
        xtrain = count_vec.transform(train_df.Comment)
        xtest = count_vec.transform(test_df.Comment)
        
        model = linear_model.LogisticRegression()
        
        model.fit(xtrain,train_df.Insult)
        
        preds = model.predict(xtest)
        
        accuracy = metrics.roc_auc_score(test_df.Insult,preds)
        
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
        
        
        
        
        
        
        
        
        
        
        
        
