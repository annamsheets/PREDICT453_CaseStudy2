mport nltk
from textblob import TextBlob as tb
import math
import io
import codecs
from __future__ import division
import numpy
import pandas as pd
import matplotlib
%matplotlib inline 

import sys
import re
import os
import string
import re
os.chdir('/users/asheets/Documents/Work_Computer_new/Work_Computer/Grad_School/PREDICT_453/Notebooks/DSI/')

all_docs = []
blob_list = []
d = 24

cachedStopWords = nltk.corpus.stopwords.words('english')
pattern = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*')

for i in range(16):
    doc_name = 'DSI' + str(d) + '.txt'
    try:
        with open(doc_name, 'r') as f:
            sample = f.read()
        sample = sample.decode('utf-8')
       #sample = sample.decode('ascii')
        sample = sample.lower()
        sample = re.sub(r'[^\w]', ' ', sample)
        sample = ''.join([i for i in sample if not i.isdigit()])
        sample = pattern.sub('', sample)
        sample = "".join(l for l in sample if l not in string.punctuation)
        sample2 = " ".join(k for k in tb(sample).noun_phrases)
        all_docs.append({'DSInum': d, 'raw_text': sample, 'noun_phrases_only':
        sample2})
        blob_list.append(tb(sample2))
        d = d + 1
    except IOError:
        d = d + 1
        pass
    
all_documents = pd.DataFrame(all_docs)
    

#http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/

#computes "term frequency" which is the number of times a word appears in a document blob, 
#normalized by dividing by the total number of words in blob. 
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

#returns the number of documents containing word. A generator expression is passed to the sum() function.
def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

#computes "inverse document frequency" which measures how common a word is among all documents in bloblist. 
#The more common a word is, the lower its idf. 
#We take the ratio of the total number of documents to the number of documents containing word, 
#then take the log of that. Add 1 to the divisor to prevent division by zero.
def idf(word, bloblist):
     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

#computes the TF-IDF score. It is simply the product of  tf and idf.
def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

#compare all DSIs using all pre-defined functions
DSI_list = all_documents["DSInum"]
for i, blob in enumerate(blob_list):
    print("Top words in document {}".format(DSI_list[i]))
    scores = {word: tfidf(word, blob, blob_list) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        
##Read in RTV and check frequencies
RTV = pd.read_csv('RTV.txt',header=None)
RTV.columns = ['word']

RTVblob = tb(str(tuple(RTV.word.tolist())).replace("'", ""))

tf_list = []
for i, blob in enumerate(blob_list):
    for word in RTVblob.words:
        tf_list.append({'DSInum': DSI_list[i], 'word': word, 'term_freq': blob.words.count(word)})

#print pd.DataFrame(tf_list)
tf_df = pd.DataFrame(tf_list)
tf_df.to_csv("RTV_frequencies.txt", sep='\t')
