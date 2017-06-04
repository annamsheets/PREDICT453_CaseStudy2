# -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:46:39 2017

@author: matth
"""
import pandas as pd
import numpy as np
import os
import nltk
from sklearn.metrics import roc_curve, auc
import re
from sklearn import tree
from sklearn.model_selection import train_test_split
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import confusion_matrix
import pydotplus
import matplotlib.pyplot as plt
from IPython.display import Image
import datetime as dt
import seaborn as sns


## folder with data
os.chdir("C:/Users/matth/Desktop/MSPA/453")
## Read data from EMC
df = pd.read_pickle('data')
## Manually creaed ECs
ec = pd.read_csv('ec.csv')

## colums used to validate the data
del ec['count']
del ec['updated']
ec_dict = ec.set_index('token').T.to_dict('list')

## load nltk's SnowballStemmer as variabled 'stemmer'
lemma = nltk.wordnet.WordNetLemmatizer()

## Code below lemmatizes the terms in the next steps to reduct unique terms
## Leveraging code from http://brandonrose.org/clustering
def tokenize_and_lemmatize(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1: ## must be longer than 1
            filtered_tokens.append(token)
    lemmas = [lemma.lemmatize(t) for t in filtered_tokens]
    return lemmas

## This is used to create the ECs later on 
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1: ## must be longer than 1
            filtered_tokens.append(token)
    return filtered_tokens  

## This code applies functions and removes system formatting
data_samples = []
for n in df['nxt_stps']:
    try:
        match = re.search(r'\d{4}-\d{2}-\d{2}', n) 
        n = n[match.span()[1] + 1:n.find('\n')]
        n = n.replace('\r',"")
        n = re.sub('[^a-zA-Z\s]',"",n)
        words = n.split()
        n = ' '.join(str(ec_dict.get(str.lower(word), word)[0]) for word in words)
        n = re.sub('EC_ZZZ',"",n)
        n = " ".join(tokenize_and_lemmatize(n))
        data_samples.append(n)
    except:
        n = n[0:n.find('\n')]
        n = n.replace('\r',"")
        words = n.split()
        n = ' '.join(str(ec_dict.get(str.lower(word), word)[0]) for word in words)
        n = re.sub('EC_ZZZ',"",n)
        n = " ".join(tokenize_and_lemmatize(n))
        data_samples.append(n)     

auc_list = [] ## capture AUC from each decison tree
model_list = [] ## capture each model
tpr_list = [] ## save tpr for analysis
fpr_list = [] ## save fpr for analysis
NUM_TOPICS_TO_TRY = 15
for NUM_TOPICS in range(3,NUM_TOPICS_TO_TRY+1): # +1 to get to as many as specified 
    ## LDA
    n_features = 1000
    n_topics = NUM_TOPICS
    n_top_words = 20
    
    ## leveraging code from SKLearn documentation with some modification
    # Author: Olivier Grisel <olivier.grisel@ensta.org>
    #         Lars Buitinck
    #         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
    # License: BSD 3 clause
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()
    
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA with {0} topcics...".format(NUM_TOPICS))
    tf_vectorizer = CountVectorizer(max_df=0.95 ## Exclude if used in 95% docs
                                    , min_df=3  ## Exclude if 3 or few docs
                                    ##, max_features=n_features
                                    , stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    
    
    print("Fitting LDA models with {0} topics".format(NUM_TOPICS))
    lda = LatentDirichletAllocation(n_topics=n_topics
                                    , max_iter=25
                                    , learning_method='online'
                                    , learning_offset=50.
                                    , random_state=42)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))
    
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)    
    
    ## This fits the LDA model on to my data
    df_r = pd.DataFrame(lda.transform(tf)) 
    ## This joins the LDA topic weights for each next step onto my data with outcomes
    df_r = pd.concat([df,df_r],axis = 1)
    ## filter to week 12 to avoid seeing the same deal twice
    df_r = df_r[df_r.wk_of_qtr_cy == 12]
    ## Create training data for model. These columns are the topic numbers
    df_data = df_r[list(range(NUM_TOPICS))]
    ## create target data for model. This is the deal outcome
    df_target = df_r[['event_status']]
    ## recode to binary outcome
    df_target = df_target['event_status'].replace(['CLOSED','PUSHED'], 'NOT BOOKED')
    ## create train/test split
    X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, random_state=0) 
    ## fit DT
    clf = tree.DecisionTreeClassifier(max_depth = 3)
    clf = clf.fit(X_train, y_train)
    model_list.append(clf)
    ## predict on my testing data
    Z = clf.predict(X_test)
    Z_prob  = clf.predict_proba(X_test)
    
    ## view results as confusion matrix
    cm = confusion_matrix(y_test, Z)
    print(cm)
    ## calculate accuracy
    acc = (np.sum(cm[0,0]) + np.sum(cm[1,1])) / np.sum(cm)
    print("Accuracy:",acc)
    
    ## calculate and plot AUC 
    Y_ROC = np.array(y_test)
    Y_SCORE_ROC = np.array(Z_prob)
    
    fpr, tpr, thresholds = roc_curve(Y_ROC, Y_SCORE_ROC[:,0], pos_label="BOOKED")
    fpr_list.append(fpr) 
    tpr_list.append(tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot - {0} Topics'.format(NUM_TOPICS))
    plt.legend(loc="lower right")
    plt.show()
    ## retain AUC for analysis
    auc_list.append(auc(fpr,tpr))

## This plots all the ROC Curves on one plot with the max in orange
plt.figure()
lw = 2
MAX_AUC = np.argmax(auc_list)
BEST_MODEL_NUM = list(range(3,len(auc_list)+3))[MAX_AUC]
for i in range(0,len(auc_list)):
    if i != MAX_AUC:
        plt.plot(fpr_list[i], tpr_list[i], color='darkblue',
                     lw=lw)
## plotting this last so it is on top
plt.plot(fpr_list[MAX_AUC], tpr_list[MAX_AUC], color='darkorange',
             lw=lw, label='Max AUC = {0} from {1} Topics'.format(round(auc_list[MAX_AUC],2), BEST_MODEL_NUM))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Plot: 3 - {0} Topics'.format(NUM_TOPICS))
plt.legend(loc="lower right")
plt.show()       
    
## This plots the AUC from each model for inspection
auc_series = pd.Series(auc_list, index = list(range(3,11))
ax = sns.barplot(x = auc_series.index.values.tolist()
            , y = auc_series
            , color = 'darkorange')
ax.set(xlabel='Number of Topics'
       , ylabel='AUC'
       , title = 'ROC Curve Area 3 - {0} Topics'.format(NUM_TOPICS) )

## create decision tree chart for 8 topic model
dot_data = tree.export_graphviz(model_list[5], out_file=None
                                , filled = True
                                , rounded = True
                                , class_names = ['Booked','Not Booked']) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())  

## This was used to create the ECs
## Leveraging code from http://brandonrose.org/clustering
docs_for_ec = []
for n in df['nxt_stps']:
    try:
        match = re.search(r'\d{4}-\d{2}-\d{2}', n) 
        n = n[match.span()[1] + 1:n.find('\n')]
        n = n.replace('\r',"")
        n = re.sub('[^a-zA-Z\s]',"",n)
        n = " ".join(tokenize_only(n))
        docs_for_ec.append(n)
    except:
        n = n[0:n.find('\n')]
        n = n.replace('\r',"")
        n = " ".join(tokenize_only(n))
        docs_for_ec.append(n)

totalvocab_tokenized = []
for i in range(len(docs_for_ec)):
    allwords_tokenized = docs_for_ec[i]
    totalvocab_tokenized.extend(allwords_tokenized)
 
df_export = pd.DataFrame(totalvocab_tokenized, columns = ['token'])
## Compute term frequency to prioritize time with ECs
counts = df_export.groupby('token').size()
## save to file for analysis
counts.to_csv('output_{0}.csv'.format(dt.datetime.today().strftime("%m_%d_%Y")))


        