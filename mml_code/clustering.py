# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:06:04 2017

@author: matth
"""

#import numpy as np
import pandas as pd
import nltk
import re
import os
#import codecs
#from sklearn import feature_extraction
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import mpld3

stopwords = nltk.corpus.stopwords.words('english')

## this data structure holds info about each DSI
## thank you to Marek Blat for the FileData Class
class FileData(object):
    name = ""
    content = ""

    def __init__(self, name, content):
        self.name = name
        self.content = content
    
    ## holds the names of each DSI
    def get_name(self):
        return self.name
    
    ## holds the raw content
    def get_content(self):
        return self.content
    
    ## returns content without stopwords. Probably does not scale well.
    def get_content_filtered(self):
        return ' '.join([word for word in self.content.split() if word not in stopwords])
         

## extract all file names from a path
def get_files(path):
    os.chdir(path)
    files = [f for f in os.listdir()]
    return files

## keep only fileNames with DSI
def get_txt_files(files):
    txt_files = []
    for fileName in files:
        if "DSI" in fileName and ".txt" in fileName:
            txt_files.append(fileName)
    return txt_files    

## opens all the files that meet the criteria above
def get_txt_file_data(path):
    files = get_files(path)
    txt_files = get_txt_files(files)
    documents = []
    for fileName in txt_files:
        ## I added the ignore as some docs had chars that gave me an error
        file = open(fileName, 'r',encoding='utf-8', errors = 'ignore')
        content = file.read()
        ## This uses the function defined above
        fileData = FileData(fileName, content)
        documents.append(fileData)
    ## sort documents by DSI number
    documents = sorted(documents, key=lambda x: int(re.sub(r'(\D)', "", x.get_name())[0]))
    return documents

path = "C:/Users/matth/PREDICT453_CaseStudy2/DSI"
docs = get_txt_file_data(path)

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1: ## must be longer than 1
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1: ## must be longer than 1
            filtered_tokens.append(token)
    return filtered_tokens    

#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in range(len(docs)):
    allwords_stemmed = tokenize_and_stem(docs[i].get_content_filtered()) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(docs[i].get_content_filtered())
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

## removes stopwords 
## filter_df = vocab_frame.index.isin(stopwords)
## vocab_frame = vocab_frame[~filter_df]

## Create tf-idf matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(stop_words='english'                                 
                                   #,max_features=200000
                                   #,min_df=0.2
                                   ,max_df=0.8
                                   ,use_idf=True
                                   ,tokenizer=tokenize_and_stem
                                   ,ngram_range=(1,3)
                                   )

## need to extract content and names from data structure
doc_content = []
for doc in docs:
    doc_content.append(doc.get_content_filtered())

doc_name = []
for doc in docs:
    doc_name.append(doc.get_name())

tfidf_matrix = tfidf_vectorizer.fit_transform(doc_content) #fit the vectorizer to content
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
num_clusters = 4
## runs cluster
km = KMeans(n_clusters=num_clusters
            ,random_state=42
            ,max_iter = 1000) 
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

doc_dict = { 'DSI': doc_name, 'content': doc_content, 'cluster': clusters}
frame = pd.DataFrame(doc_dict, index = [clusters] )

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    
    print("Cluster %d DSIs:" % i, end='')
    print()
    if type(frame.ix[i]['DSI']) is not str:
        for dsi in list(frame.ix[i]['DSI']):
            print(dsi)
    else:
        print(str(frame.ix[i]['DSI']))

print()
print()

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix
                ,orientation="right"
                ,labels=doc_name)

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
##plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters