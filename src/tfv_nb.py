#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:02:56 2020

@author: ian
"""
import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

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
        
        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize,token_pattern=None)
        
        tfidf_vec.fit(train_df.Comment)
        
        xtrain = tfidf_vec.transform(train_df.Comment)
        xtest = tfidf_vec.transform(test_df.Comment)
        
        model = naive_bayes.MultinomialNB()
        
        model.fit(xtrain,train_df.Insult)
        
        preds = model.predict(xtest)
        
        accuracy = metrics.roc_auc_score(test_df.Insult,preds)
        
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
        
        
        
        
        
        
        
        
        
        
        
        
