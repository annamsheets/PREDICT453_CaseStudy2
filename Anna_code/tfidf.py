
# coding: utf-8

# In[2]:

import nltk
from textblob import TextBlob as tb
import math
import io
import codecs
from __future__ import division
import numpy
import pandas as pd
import matplotlib
get_ipython().magic(u'matplotlib inline')


# In[3]:

#http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/

#computes "term frequency" which is the number of times a word appears in a document blob, 
#normalized by dividing by the total number of words in blob. 
def tf1(word, blob):
    return blob.words.count(word)

def tf2(word, blob):
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
    return tf2(word, blob) * idf(word, bloblist)


# In[4]:

import sys
import re
import os
import string
import re
os.chdir('/users/asheets/Documents/Work_Computer_new/Work_Computer/Grad_School/PREDICT_453/Notebooks/DSI/')

all_docs = []
all_docs2 = ""
blob_list = []
blob_list_raw = []
blob_list_bigrams = []
blob_list_trigrams = []
blob_list_fourgrams = []
blob_list_fivegrams = []
d = 1

cachedStopWords = nltk.corpus.stopwords.words('english')
pattern = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*')

for i in range(40):
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
        all_docs2 = all_docs2 + sample2
        blob = tb(sample)
        blob_list.append(blob)
        blob_list_raw.append(blob)
        blob_list_bigrams.append(blob.ngrams(n=2))
        blob_list_trigrams.append(blob.ngrams(n=3))
        blob_list_fourgrams.append(blob.ngrams(n=4))
        blob_list_fivegrams.append(blob.ngrams(n=5))
        d = d + 1
    except IOError:
        d = d + 1
        pass

all_documents = pd.DataFrame(all_docs)
DSI_list = all_documents['DSInum']

#all_documents.head(n=5)


# In[ ]:

#compare my article
blob1 = tb(all_documents['noun_phrases_only'][23])
allblob = tb(all_docs2)
print 'there are' , len(blob1.words) , 'words in DSI 24'
print 'there are' , len(allblob.words) , 'words across the entire Corpus'
term_freq = [tf1(word,blob1) for word in blob1.words]
term_rel_freq = [tf2(word,blob1) for word in blob1.words]
all_docs_term_freq = [tf1(word,allblob) for word in blob1.words]
all_docs_term_rel_freq = [tf2(word,allblob) for word in blob1.words]
tf_df1 = pd.DataFrame({'word': blob1.words, 'all_docs_term_freq': all_docs_term_freq, 'doc1_term_freq': term_freq, 'all_docs_term_rel_freq': all_docs_term_rel_freq, 'doc1_term_rel_freq': term_rel_freq})

docs_containing1 = pd.DataFrame({'word': blob1.words, 'doc_freq': [n_containing(word,blob_list) for word in blob1.words]})

df = pd.merge(tf_df1,docs_containing1,on='word',how='outer').drop_duplicates()

df['intermediate_calc'] = (len(blob_list) / df['doc_freq']).astype(float)
df['idf'] = df['intermediate_calc'].apply(math.log)
df = df.sort_values("idf",ascending=True)

tfidf_df = pd.DataFrame({'word': blob1.words, 'tf_idf_doc1': [tfidf(word, blob1, blob_list) for word in blob1.words]})
tfidf_df_all = pd.DataFrame({'word': blob1.words, 'tf_idf_all_docs': [tfidf(word, allblob, blob_list) for word in blob1.words]})
tf_idf = pd.merge(tfidf_df,tfidf_df_all,on='word',how='outer').drop_duplicates()

df = pd.merge(df,tf_idf,on='word',how='outer').drop_duplicates()

df_final = df[["word","doc1_term_freq","all_docs_term_freq","all_docs_term_rel_freq","doc1_term_rel_freq","doc_freq","idf","tf_idf_all_docs","tf_idf_doc1"]]
df_final = df_final.sort_values(by=['tf_idf_doc1'], ascending=[False])
df_final.to_csv("/users/asheets/Documents/Work_Computer_new/Work_Computer/Grad_School/PREDICT_453/Notebooks/DSI24_tfidf.txt", sep='\t')
df_final = df_final.round(4)
df_final.sort_values(by=['tf_idf_doc1'], ascending=[False]).head(n=10)


# In[ ]:

##Compare just the two articles we know to be similar
blob2 = tb(all_documents['noun_phrases_only'][27])
term_freq = [tf1(word,blob2) for word in blob2.words]
term_rel_freq = [tf2(word,blob2) for word in blob2.words]
tf_df2 = pd.DataFrame({'word': blob2.words, 'doc2_term_freq': term_freq, 'doc2_term_rel_freq': term_rel_freq})
two_tf = pd.merge(tf_df1,tf_df2,on='word',how='outer').drop_duplicates()

docs_containing1 = pd.DataFrame({'word': blob1.words, 'doc_freq': [n_containing(word,blob_list) for word in blob1.words]})
docs_containing2 = pd.DataFrame({'word': blob2.words, 'doc_freq': [n_containing(word,blob_list) for word in blob2.words]})

doc_freq = pd.concat([docs_containing1,docs_containing2]).drop_duplicates().reset_index(drop=True)
df = pd.merge(doc_freq,two_tf,on='word',how='inner')
df = df.sort_values("doc_freq",ascending=False)

df['intermediate_calc'] = (len(blob_list) / df['doc_freq']).astype(float)
df['idf'] = df['intermediate_calc'].apply(math.log)
df = df.sort_values("idf",ascending=False)

scores1 = pd.DataFrame({'word': blob1.words, 'tf_idf_doc1': [tfidf(word, blob1, blob_list) for word in blob1.words]})
scores2 = pd.DataFrame({'word': blob2.words, 'tf_idf_doc2': [tfidf(word, blob2, blob_list) for word in blob2.words]})

two_tfidf= pd.merge(scores1,scores2,on='word',how='outer').drop_duplicates()
two_tfidf = two_tfidf[['word', 'tf_idf_doc1', 'tf_idf_doc2']]
two_tfidf = two_tfidf.sort_values(by=['tf_idf_doc1', 'tf_idf_doc2'], ascending=[False,False])
two_tfidf.head(n=15)


# In[ ]:

#compare all DSIs sing all pre-defined functions
for i, blob in enumerate(blob_list):
    print("Top words in document {}".format(DSI_list[i]))
    scores = {word: tfidf(word, blob, blob_list_raw) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:10]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
             


# In[16]:

RTV = pd.read_table('RTV_new.txt',sep='\t')
unigrams = RTV.loc[RTV['Word_Count'] == 1]

RTV_final = list(unigrams['Term'])

#RTVblob = tb(str(tuple(RTV.Term.tolist())).replace("'", ""))
#print RTV_final

tf_list = []

for i, blob in enumerate(blob_list_raw):
    for item in RTV_final:
        tf_list.append({'DSInum': DSI_list[i], 'Term': item, 'term_freq': blob.words.count(item.lower())})

#print pd.DataFrame(tf_list)
tf_df = pd.DataFrame(tf_list)


# In[17]:

#for the bigrams, trigrams, fourgrams and fivegrams
bigrams_tmp = RTV.loc[RTV['Word_Count'] == 2]
RTV_final = list(bigrams_tmp['Term'])

tf_list2 = []
for item in RTV_final:
    for i in range(len(blob_list_bigrams)):
        term_freq = 0
        for k in range(len(blob_list_bigrams[i])):
            bigram = ' '.join(str(elem.lower()) for elem in blob_list_bigrams[i][k]).lstrip()
            if item.lower() in bigram: 
                term_freq = term_freq + 1
        tf_list2.append({'DSInum': DSI_list[i], 'Term': item.lower(), 'term_freq': term_freq})


tf_df2 = pd.DataFrame(tf_list2)


# In[18]:

#for the bigrams, trigrams, fourgrams and fivegrams
trigrams_tmp = RTV.loc[RTV['Word_Count'] == 3]
RTV_final = list(trigrams_tmp['Term'])

tf_list3 = []
for item in RTV_final:
    for i in range(len(blob_list_trigrams)):
        term_freq = 0
        for k in range(len(blob_list_trigrams[i])):
            trigram = ' '.join(str(elem.lower()) for elem in blob_list_trigrams[i][k]).lstrip()
            if item.lower() in trigram: 
                term_freq = term_freq + 1
        tf_list3.append({'DSInum': DSI_list[i], 'Term': item.lower(), 'term_freq': term_freq})

tf_df3 = pd.DataFrame(tf_list3)


# In[19]:

#for the bigrams, trigrams, fourgrams and fivegrams
fourgrams_tmp = RTV.loc[RTV['Word_Count'] == 4]
RTV_final = list(fourgrams_tmp['Term'])

tf_list4 = []
for item in RTV_final:
    for i in range(len(blob_list_fourgrams)):
        term_freq = 0
        for k in range(len(blob_list_fourgrams[i])):
            fourgram = ' '.join(str(elem.lower()) for elem in blob_list_fourgrams[i][k]).lstrip()
            if item.lower() in fourgram: 
                term_freq = term_freq + 1
        tf_list4.append({'DSInum': DSI_list[i], 'Term': item.lower(), 'term_freq': term_freq})

tf_df4 = pd.DataFrame(tf_list4)


# In[25]:

tf_df_final = tf_df.append([tf_df2, tf_df3,tf_df4])

RTV['Term'] = RTV['Term'].str.lower()
tf_df_final['Term'] = tf_df_final['Term'].str.lower()

tf_df_final2 = pd.merge(tf_df_final,RTV,on='Term',how='inner')
tf_df_agg = tf_df_final2.groupby(['DSInum', 'EC']).sum()
tf_df_agg.to_csv("RTV_frequencies.txt", sep='\t')


# In[26]:

tf_df3 = pd.read_table('RTV_frequencies.txt',sep='\t')
new_tf = tf_df3[tf_df3['DSInum'] == 1]
new_tf = new_tf[["EC","term_freq"]]

for i in range(1,39):
    try:
        tmp = tf_df3[tf_df3['DSInum'] == DSI_list[i]]
        tmp = tmp[["EC","term_freq"]]
        new_tf = pd.merge(new_tf, tmp, on='EC', how='outer')
    except IOError:
        pass  


my_columns = ["EC"]
for i in DSI_list:
    my_columns.append('DSI' + str(i))
new_tf.columns = my_columns
new_tf.to_csv("RTV_frequencies_final.txt", sep='\t')
new_tf


# In[ ]:



