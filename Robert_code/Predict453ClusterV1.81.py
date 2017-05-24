# get our packages
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import string
import math
import sys
import csv

from sklearn import metrics
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.hierarchy import ward, dendrogram
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from sklearn import feature_extraction
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans        
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# pull in ontology rtv
def getBaseRTV3():
    # Note, same file format as in getBaseRTV2
    #
    # IMPORTANT, SEE IMPORTANT BELOW ABOUT 15 lines down
    #
    with open("/Users/Robert/Documents/Predict453/rtvfrequenciesV1.0.csv", "rt" ) as theFile:
        rX = csv.reader( theFile )
        rowData = [row for row in rX]
    
    vocab2 = []
    for rowX in range(1,len(rowData)):
        if int(rowData[rowX][2]) > 0:
            vocab2.append(rowData[rowX][1])
    
    rtvMatrix2 = []
    rtvMatrixXsub2 = []
    for colX in range(3,43):
        rtvMatrixSub2 = []
        # IMPORTANT
        # update this to number of rows in csv file
        for rowX in range(1,25):
            if int(rowData[rowX][2]) > 0:    
                rtvMatrixSub2.append(int(rowData[rowX][colX]))
            
        rtvMatrix2.append(rtvMatrixSub2)
    return vocab2, rtvMatrix2
        
    
# pull in basic rtv files
def getBaseRTV2():
    # nothing done here to automate/validate input
    #
    # IMPORTANT, SEE IMPORTANT BELOW ABOUT 15 lines down
    #
    # expect a csv file with:
    #   Header Row with 'ECID', 'EC', 'Total', 'DSI1', 'DSI2', ... 'DSIn'
    #   Column 1: ECID (starts at 0 and increases by 1 for each row/term
    #   Column 2: RC (rtv/ec term)
    #   Column 3: Total (sum of columns for 'DSI's
    #             Note, reason for this column is to exclude terms not occuring
    #             which shouldn't happen, but during the process, nothings perfect
    #   Column 4-.Term counts for DSI1
    #             Note, for all rows in columns for DSIs, expect 0's so do a replace
    #             in excel before saving as csv and replace all cells which match
    #             '' with '0'
    #   Column 5: Term counts for DSI2
    #   etc
    #
    #   Keep your versions separate and you can always go back
    #
    # with open("/Users/Robert/Documents/Predict453/rtv3_051117V2.csv", "rt" ) as theFile:
    # with open("/Users/Robert/Documents/Predict453/rtv3_052117V3.csv", "rt" ) as theFile:
    # with open("/Users/Robert/Documents/Predict453/rtv4_052117V1.csv", "rt" ) as theFile:
    with open("/Users/Robert/Documents/Predict453/Predict453ClusterRTV052317.csv", "rt" ) as theFile:
        rX = csv.reader( theFile )
        rowData = [row for row in rX]
    
    vocab2 = []
    for rowX in range(1,len(rowData)):
        if int(rowData[rowX][2]) > 0:
            vocab2.append(rowData[rowX][1])
    
    rtvMatrix2 = []
    rtvMatrixXsub2 = []
    for colX in range(3,43):
        rtvMatrixSub2 = []
        # IMPORTANT
        # update this based on rows in csv
        for rowX in range(1,116):
            if int(rowData[rowX][2]) > 0:    
                rtvMatrixSub2.append(int(rowData[rowX][colX]))
            
        rtvMatrix2.append(rtvMatrixSub2)
    return vocab2, rtvMatrix2
        
    
def getWordList():
    wordList = ["president",	"energy",	"jobs",	"executive order",	"russia",	"government",	"administration",	"regulation",	"campaign",	"secretary",	"vote",	"power",	"mexico",	"climate change",	"environment",	"bill",	"election",	"wall",	"business",	"border",	"republican",	"russian",	"congress",	"washington",	"work",	"domesticSecurity",	"bank",	"reform",	"conservative",	"American Health Care Act",	"immigration",	"cost",	"party",	"economy",	"Ryan",	"leader",	"United States",	"construction",	"money",	"Small business confidence",	"media",	"twitter",	"Healthcare",	"gorsuch",	"house Republican",	"EPA",	"moscow",	"certain Middle-Eastern countries",	"tweet",	"Navarro",	"budget cuts",	"Japan",	"war",	"voter",	"democrat",	"public safety",	"FCC",	"democracy",	"Department of Homeland Security",	"Internet User",	"immigrant",	"refugees",	"legislation",	"deductibles",	"filibuster",	"travel",	"hack",	"tariff",	"Protectionism",	"premiums",	"Gerrymander",	"ISP",	"Small business confidence index",	"VPN",	"uncertainty"]
    return wordList

def loadDSIs():
    # download the DSI's (see link from Marek), and put in a 
    #   folder with only the DSI's, all name as is
    # then put that path to those DSIs in the path statement
    os.chdir('/Users/Robert/Documents/Predict453/DSIs/TXT/')
    path = '/Users/Robert/Documents/Predict453/DSIs/TXT/*.txt'
    files = glob.glob(path)

    # put number for dsi's and text into lists 
    dsiNums = []
    synopses = []

    for cnt in range(len(files)):
        dsiName = files[cnt].split('/')[7]
        dsiNum = (files[cnt].split('/')[7]).split('.')[0]
        dsiNums.append(dsiNum)
        allLines = ''
        with open(dsiName, "rb") as f:
            s = f.read().replace('\r\n', '\n').replace('\r', '\n')
            line = s.split('\n')
            # lets get rid of some workds/characters we do not want
            for l in line:
                l = l.lower()
                l = l.replace("'","")
                l = l.replace('"','')
                l = l.replace(',',' ')
                l = l.replace('?',' ')
                l = l.replace('said',' ')
                l = l.replace('mr. ',' ')
                l = l.replace('could ',' ')
                l = l.replace('also ',' ')
                l = l.replace('could ',' ')
                l = l.replace('like ',' ')
                l = l.replace('take ',' ')               
                l = l.replace('new ',' ')
                l = l.replace('one ',' ')               
                l = l.replace('make ',' ')
                               
                l = re.sub(r'[^\x00-\x7F]+','', l)
                allLines = allLines + l
            
        synopses.append(allLines)
    
    # load nltk's English stopwords as variable called 'stopwords'
    stopwords = nltk.corpus.stopwords.words('english')
    
    return synopses, dsiNums


def tokenize_and_stem(text):
    # mechanism to identify base of words
    stemmer = SnowballStemmer("english")
    # here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
    
    
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist, wordfreq))

    
def gettfidf(synopses):    
    #not super pythonic, no, not at all.
    #use extend so it's a big flat list of vocab
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in synopses:
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)  
    
    # create a data frame with the stemmed vocabulary as the index
    #   and the tokenized words as the column    
                
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'  
    #print vocab_frame
    #print vocab_frame.words.value_counts()
    
    # create tf-idf
    # 1) get term frequency matrix - word occurences by document
    # 2) apply term frequency-inverse document frequency weighting
    
    #define vectorizer parameters
    min_df = 0.2 # min number of documents the term must be included in
    max_df = 0.8 # max frequency within the doucments a given feature
                #   can be used in the tf-idf matrix 
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=200000,
                                    min_df=min_df, stop_words='english',
                                    use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
    tfidf_matrixD = tfidf_vectorizer.fit_transform(synopses).todense() #fit the vectorizer to synopses
    
    # list of terms in the tf-idf matrix
    terms = tfidf_vectorizer.get_feature_names()
    
    termsX = []
    for indx in range(len(terms)):
        termsX.append(terms[indx].encode('ascii','ignore'))
    my_count = sorted(termsX)  
    wordListFreq = wordListToFreqDict(terms)

    return tfidf_matrix, tfidf_matrixD,  vocab_frame, terms


def getCosineSimilarity(tfidf_matrix, dsiNums):
    dist = 1 - cosine_similarity(tfidf_matrix)
    gl = []
    gls = []
    for docNum in range(len(dist)):
        cs_doc = cosine_similarity(tfidf_matrix[docNum:(docNum+1)], tfidf_matrix)
        gls = []
        for indx in range(len(dist)):
            cos_sim = cs_doc[0][indx]
            #print 'cs',indx,cs_doc[0][indx],cos_sim
            if cos_sim < -1.0:
                cos_sim = -1.0
            if cos_sim > 1.0:
                cos_sim = 1.0
            angle_in_radians = math.acos(cos_sim)
            gls.append(cos_sim)
        gl.append(gls)
    g = np.array(gl)
    gdf = pd.DataFrame(gl)
    gdf.columns = dsiNums
    gdf.index = dsiNums
    printCosineSimilarityDetail = 0
    if printCosineSimilarityDetail == 1:
        print "Cosine Similarity Matrix Detail"
    
        for indxC in range(39):
            for indxR in range(39):
                print indxC, indxR, gdf.iloc[indxC][indxR]

    import seaborn as sns; sns.set()
    
    sns.plt.figure()
    fig, ax = plt.subplots(figsize=(8,6))   
    sns.set(font_scale=1.0)
    ax = sns.heatmap(gdf, linewidths=0.25, vmin=0, vmax=1, 
    annot=False,
    annot_kws={"size":6},cmap='Greens',linecolor='darkgreen')
    ax.tick_params(labelsize=6)
    plt.yticks(rotation=0) 
    ax.set_title('Cosine Similarity - Corpus/DSIs')
    
    ax.text(0, -6, 'Notes on the chart', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    
    sns.plt.show()
    plt.savefig('/Users/Robert/Documents/Predict453/plotCosineSimilarity.png',  
                edgecolor='white', transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig()
    return dist


def pcaXKMeans(doc_term_matrix_tfidf_l2, dsiNums):

    for num_clusters in range(2,10):    
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=3425)
        kmeans_model.fit(doc_term_matrix_tfidf_l2)
        labels = kmeans_model.predict(doc_term_matrix_tfidf_l2)
        print labels
        centroids = kmeans_model.cluster_centers_

        fig, ax = plt.subplots(figsize=(8,6))

        plt.grid()
        plt.scatter(data2D[:,0], data2D[:,1],marker='o', c="green", alpha=0.75, s=200)
    
        for indx in range(len(data2D)):
            ax.annotate(dsiNums[indx], (data2D[indx][0]-0.012, data2D[indx][1]-0.017),
            color="white",fontsize=8)
        
        plt.title('PCA Plot Normalized TF-IDF')
        plt.grid()
        plt.show()
        plt.savefig('/Users/Robert/Documents/Predict453/plotPCATFIDFNormalizedPlot.png',  
                    edgecolor='white', transparent=False,
                    orientation='portrait',
                    pad_inches=0.1,
                    frameon=False)
        pdfdoc.savefig() 
    
                
                    
                            
def pcaX(doc_term_matrix_tfidf_l2, dsiNums):
    pca = PCA(n_components=2).fit(doc_term_matrix_tfidf_l2)
    data2D = pca.transform(doc_term_matrix_tfidf_l2)
    fig, ax = plt.subplots(figsize=(8,6))

    plt.grid()
    plt.scatter(data2D[:,0], data2D[:,1],marker='o', c="green", alpha=0.75, s=200)
    
    for indx in range(len(data2D)):
        ax.annotate(dsiNums[indx], (data2D[indx][0]-0.012, data2D[indx][1]-0.017),
        color="white",fontsize=8)
    
    plt.title('PCA Plot Normalized TF-IDF')
    plt.grid()
    plt.show()
    plt.savefig('/Users/Robert/Documents/Predict453/plotPCATFIDFNormalizedPlot.png',  
                edgecolor='white', transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig() 

def pcaXNN(doc_term_matrix_tfidf_l2, dsiNums, numRows):
    pltArray = []
    x=[]
    for numComponents in range(2,numRows):
        pca = PCA(n_components=numComponents).fit(doc_term_matrix_tfidf_l2)
        var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    for indx in range(len(var1)):
        pltArray.append(var1[indx])
        newX = indx+2
        x.append(newX)
        
    print "PCA Variance Explained"
    print var1
     
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(x,pltArray,marker='o', c="green")
    plt.grid()
    plt.title("Cumulative Percent Variance Explained by PCA Components")
    plt.show()

def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
    
def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns   
    
def lda(topicRange):
    from gensim import corpora, models, similarities 
    stopwords = nltk.corpus.stopwords.words('english')
    #remove proper names
    preprocess = [strip_proppers(doc) for doc in synopses]

    #tokenize
    tokenized_text = [tokenize_and_stem(text) for text in preprocess]

    #remove stop words
    texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

    #create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(texts)
    
    #remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    
    #convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in texts]

    for number_of_topics in topicRange:
        lda = models.LdaModel(corpus, num_topics=number_of_topics, 
                                id2word=dictionary, 
                                update_every=5, 
                                chunksize=10000, 
                                passes=100)

        topics_matrix = lda.show_topics(formatted=False, num_words=20)
    
        print " "
        print "LDA - Number of Topics = " + str(number_of_topics) + " - Top 5 terms per topic"
        print " "
        print "Topic  Terms/Relative Importance"
        print "-----  --------------------------------------------------------------------------------"
        for indx1 in range(len(topics_matrix)):
            for indx2 in range(len(topics_matrix[0])):
                if indx2 > 0:
                    print "%3d   " % indx1,
                    for indx3 in range(5):
                        print "%-15s" % topics_matrix[indx1][indx2][indx3][0] ,
                    print
                    print "      " ,
                    for indx3 in range(5):
                        print "%8.6f       " % topics_matrix[indx1][indx2][indx3][1] ,
                    print
            print
            print
                
                
def clusterSDev(numRows, numCols, startCluster, numClusters, dsiNums):
    from sklearn.decomposition import PCA
        
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    clusterSilhouetteScores = []
    
    n_clusters = startCluster
    
    for n_clusters in range(2,38):
        clusterer = KMeans(n_clusters=n_clusters, random_state=3425)
        cluster_labels = clusterer.fit_predict(doc_term_matrix_tfidf_l2)
            
        silhouette_avg = silhouette_score(doc_term_matrix_tfidf_l2, cluster_labels)
        clusterSilhouetteScores.append(silhouette_avg)
            
        print "For number of clusters          : " + str(n_clusters)
        print "The average silhouette_score is : " + '{:8.5f}'.format(silhouette_avg)
        print "The clusters generated are      : "
        for indx in range(n_clusters):
            print "Cluster: " + '{:02d}'.format(indx) + "  DSIs: " , 
            for indx2 in range(len(cluster_labels)):
                if cluster_labels[indx2] == indx:
                    print dsiNums[indx2] ,
            print
        print
            
        n_clusters += 1
            
    x = range(2,38)
    plt.figure(figsize=(8,6))
    plt.plot(x,clusterSilhouetteScores, 'ro-')
    plt.title('Silhouette Scores by Number of Clusters')
    plt.show()
    plt.savefig('/Users/Robert/Documents/Predict453/plotSilhouetteSummaryLinePlot.png',  
                edgecolor='white',facecolor='white', transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig(tite="Silhouette Scores by Number of Clusters",dpi=600)

def clusterS(numRows, numCols, startCluster, numClusters, dsiNums):
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2).fit(doc_term_matrix_tfidf_l2)
    data2D = pca.transform(doc_term_matrix_tfidf_l2)
    # reduce to two dimensions
    
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    plt.figure(facecolor="white")
    #f, axarr = plt.subplots(numRows, 2, figsize=(8,20))
    #f.set_size_inches(numRows*3,numCols*6, forward=True)
    #f.set_size_inches(8,20)
    
    clusterSilhouetteScores = []
    
    n_clusters = startCluster
    
    for indxR in range(numRows):
        for indxC in range(numCols):
            f, axarr = plt.subplots(nrows=1,ncols=2, figsize=(8,6))
                
            f.patch.set_facecolor('white')
            f.patch.set_alpha(1.0)
            indxRStatic = 0
            firstChart = 0
            secondChart = 1
            axarr[firstChart].set_xlim([-0.1, 1])
            axarr[firstChart].set_ylim([0, len(data2D) + (n_clusters + 1) * 10])
        
            clusterer = KMeans(n_clusters=n_clusters, random_state=3425)
            cluster_labels = clusterer.fit_predict(data2D)
            silhouette_avg = silhouette_score(data2D, cluster_labels)
            clusterSilhouetteScores.append(silhouette_avg)
            print "For number of clusters = " + str(n_clusters)
            print "The average Silhouette score is :" + str(silhouette_avg)
            for indx in range(n_clusters):
                print "Cluster: ", indx, "  DSIs: " , 
                for indx2 in range(len(cluster_labels)):
                    if cluster_labels[indx2] == indx:
                        print dsiNums[indx2] ,
                print
            print
            
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data2D, cluster_labels)
        
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
        
                ith_cluster_silhouette_values.sort()
        
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
        
                color = cm.spectral(float(i) / n_clusters)
                axarr[firstChart].fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                axarr[firstChart].set_axis_bgcolor("white")
                axarr[firstChart].patch.set_facecolor('white')
            
                # Label the silhouette plots with their cluster numbers at the middle
                axarr[firstChart].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
        
            axarr[firstChart].set_title("Silhouette Plot", fontsize='10')
            axarr[firstChart].set_xlabel("Silhouette Coefficients", fontsize='10')
            axarr[firstChart].set_ylabel("Cluster", fontsize='10')
        
            # The vertical line for average silhouette score of all the values
            axarr[firstChart].axvline(x=silhouette_avg, color="red", linestyle="--")
        
            axarr[firstChart].set_yticks([])  # Clear the yaxis labels / ticks
            axarr[firstChart].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.setp(axarr[firstChart].get_xticklabels(), fontsize=8)
    
            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            
            #print doc_term_matrix_tfidf_l2
            axarr[secondChart].scatter(data2D[:, 0], data2D[:, 1], marker='.', s=80, lw=0, 
            c=colors, alpha=0.7)
            plt.setp(axarr[secondChart].get_xticklabels(), fontsize=8)
            plt.setp(axarr[secondChart].get_yticklabels(), fontsize=8)
        
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            axarr[secondChart].scatter(centers[:,0], centers[:, 1],
                    marker='o', c="white", alpha=0.25, s=200)
            axarr[secondChart].set_axis_bgcolor("white")
            axarr[secondChart].patch.set_facecolor('white')
            for i, c in enumerate(centers):
                axarr[secondChart].scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=80)
        
            axarr[secondChart].set_title("PCA Reduction", fontsize='8')
            axarr[secondChart].set_xlabel("", fontsize='8')
            axarr[secondChart].set_ylabel("", fontsize='8')
        
            plt.suptitle(("Silhouette analysis for KMeans/PCA clustering "
                        "with Number of Clusters = {:4d}".format( n_clusters)),
                        fontsize=10, fontweight='bold')
            plt.show()
            
            
            plt.savefig('/Users/Robert/Documents/Predict453/Plot_' + str(n_clusters) + '_Clusters.png',  
                edgecolor='white',facecolor=f.get_facecolor(), transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
            pdfdoc.savefig()
    
            n_clusters += 1
            
    x = range(2,numRows+2)
    print "Silhouette Scores"
    print clusterSilhouetteScores
    plt.figure(figsize=(8,6))
    plt.plot(x,clusterSilhouetteScores, 'ro-')
    plt.grid()
    plt.title('Silhouette Scores by Number of Clusters - First 2 PCA')
    plt.show()
    plt.savefig('/Users/Robert/Documents/Predict453/plotSilhouetteSummaryLinePlot.png',  
                edgecolor='white',facecolor=f.get_facecolor(), transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig()
    

def kmeansCalcs(numClusters, terms, dist):

    num_clusters = numClusters  
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=3425)    
    kmeans_model.fit(doc_term_matrix_tfidf_l2)
    labels = kmeans_model.predict(doc_term_matrix_tfidf_l2)
    centroids = kmeans_model.cluster_centers_
    print
    print
    print "Number of Clusters: ", numClusters
    print "Cluster DSIs"
    for indx in range(numClusters):
        print '  {:2d}   '.format(indx) ,
        for indx2 in range(len(dsiNums)):
            if labels[indx2] == indx:
                print dsiNums[indx2] ,
        print
    print ""
    
    mss = metrics.silhouette_score(doc_term_matrix_tfidf_l2, labels)
            
    clusters = kmeans_model.labels_.tolist()
    # print clusters
        
    datC = { 'title': dsiNums, 'synopsis': mydoclist, 'cluster': clusters}
    
    frame = pd.DataFrame(datC, index = [clusters] , columns = ['title', 'cluster'])
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1] 
    
    cluster_names = {}
    
    for i in range(num_clusters):
        cluster_words = ''
        for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
            cluster_words = cluster_words + "/" + vocabularyRTV[ind]
        print i, cluster_words
        cluster_names.update({i: cluster_words})
    
    MDS()
    
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    
    xs, ys = pos[:, 0], pos[:, 1]
        
    #set up colors per clusters using a dict
    cluster_colors = {0: 'red', 1: 'lawngreen', 2: 'gold', 3: 'green', 4: 'orange', 5: 'purple',
    6: 'black', 7: 'magenta', 8: 'brown', 9: 'blue', 10: 'aqua',
    11: 'black', 12: 'magenta', 13: 'brown', 14: 'blue', 15: 'aqua'}
    
    
    #create data frame that has the result of the MDS plus the cluster numbers and dsiNums
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=dsiNums)) 
    
    #group by cluster
    groups = df.groupby('label')
            
    # set up plot
    plt.rcParams['axes.linewidth'] = 0.1
    fig, (ax,lax) = plt.subplots(ncols=2, figsize=(8,6), gridspec_kw={"width_ratios":[4,4]})
    plt.tight_layout()     
    fig.patch.set_facecolor('white')
    ax.margins(0.20) # Optional, just adds 5% padding to the autoscaling
    ax.patch.set_facecolor('white')
    for name, group in groups:
        xP=0
        yP=0
        xP = np.mean(group.x)
        yP = np.mean(group.y)
        fig.subplots_adjust(bottom=0.2)
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=15, mew=1, 
                label=cluster_names[name], markerfacecolor=cluster_colors[name],
                alpha=0.25,
        markeredgecolor=cluster_colors[name],
        mec='none',
        markeredgewidth=2.0)
        ax.title.set_text("KMeans PCA Plot with " + str(numClusters) + ' Clusters')
        ax.title.set_position([.5, 1.0])
        ax.set_aspect('auto')
        ax.set_axis_bgcolor='cream'
        ax.tick_params(\
            axis= 'x',         # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',        # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
        
    for i in range(len(df)):
        xAdj=-0.04
        yAdj=-0.04
        ax.text(df.ix[i]['x']+xAdj, df.ix[i]['y']+yAdj, df.ix[i]['title'], size=10, 
            fontweight='bold', color='black')  

    
    h,l = ax.get_legend_handles_labels()
    lax.title.set_text("Legend")
    lax.title.set_position([.25, 1.0])
        
    lax.legend(h,l, loc=6, borderaxespad=0.2,prop={'size':8},labelspacing=1.5)
    lax.axis("off")

    plt.show()
    plt.savefig("/Users/Robert/Documents/Predict453/KMeansPCAWithTerms" + str(numClusters) + "Plot.png",  
                edgecolor='white',facecolor=fig.get_facecolor(), transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig()
                                                                            
    return ax, mss   
    
    
def plotPCA(tfidf_matrix, dsiNums):
    
    for numClusters in [2,3,4,5,6,7,8,9,10]:
        model7 = KMeans(n_clusters=numClusters)
        model7.fit(tfidf_matrix)
        clusassign = model7.predict(tfidf_matrix)
        cluster_colors = {0: 'red', 1: 'lightgreen', 2: 'yellow', 3: 'green', 4: 'blue', 5: 'purple',
            6: 'orange', 7: 'brown', 8: 'pink', 9: 'aqua', 10: 'black'}
 
        pca_2 = PCA(2)
        fig = plt.subplots(figsize=(10, 6))
        plt.rcParams['axes.facecolor']='white'
  
        plot_columns = pca_2.fit_transform(tfidf_matrix)
        plt.grid
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model7.labels_,
        s=100,marker='o', cmap='Set1')
        xOffset=-0.03
        yOffset=-0.03
        labelPoints = 1
        if labelPoints == 1:
            for indx in range(len(tfidf_matrix)):  
                labelPt = str(dsiNums[indx] )                                
                plt.annotate(labelPt, (plot_columns[indx,0]+xOffset, plot_columns[indx,1]+yOffset),
                fontsize=8)
        plt.legend()
        
        plt.xlabel=('Canonical variable 1')  
        plt.ylabel=('Canonical variable 2')
        plt.title("KMeans with " + str(numClusters) + ' Clusters' + ' with 2 PCA', fontsize=12)
        plt.tick_params(axis='both', left='on', top='off', right='off', bottom='on', labelleft='on', labeltop='off', labelright='off', labelbottom='on')
 
        plt.show()
        plt.savefig("/Users/Robert/Documents/Predict453/PCAClusters" + str(numClusters) + "Plot.png",  
                edgecolor='white', transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
        pdfdoc.savefig()
            

def denodrogram(lmatrix, orient, title, colorThreshold):
    # hierarchical clustering            

    linkage_matrix = ward(lmatrix) #define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(10, 8)) # set size
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    plt.figure(facecolor="white")
    fig.patch.set_facecolor('white')
    ax = dendrogram(linkage_matrix, orientation=orient, 
    labels=dsiNums, leaf_font_size=8,
    color_threshold=colorThreshold);
    plt.title(title)
    plt.tick_params(\
        axis= 'x',        # changes apply to the x-axis
        which='both',     # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='on',         # ticks along the top edge are off
        labelbottom='on')
    
    plt.show()
    plt.savefig("/Users/Robert/Documents/Predict453/Deno" + title + "Plot.png",  
                edgecolor='white',facecolor=fig.get_facecolor(), transparent=False,
                orientation='portrait',
                pad_inches=0.1,
                frameon=False)
    pdfdoc.savefig()


def kmeans(terms, dist):
    for row in range(numberRows):
        for col in range(numberColumns):
            clusters = startCluster + row*numberColumns + col
            ax1, mss = kmeansCalcs(clusters, terms, dist)
            print 'clusters {:2d}'.format(clusters),'  silhouette {:06.4f}'.format(mss)

    plt.show()


def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

    
def build_lexicon(corpus):
    lexicon = set()
    cnt=1
    totalWordCount = 0
    for doc in corpus:
        print "DSI" + str(cnt)
        cnt += 1
        totalWordCount += len(doc.split())
        print "  Length " + str(len(doc.split()))
        lexicon.update([word for word in doc.split()])
    print
    print "Total Word Count in Corpus is " + str(totalWordCount)
    print "Average Word Count per DSI is " + str(totalWordCount/40.0)
    print "Unique Words in Corpus is     " + str(len(lexicon))
    print 
    print "sorted lexicon"
    print sorted(lexicon)
    return lexicon

def tf(term, document):
  return freq(term, document)

def freq(term, document):
  return document.split().count(term)

def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat


# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# start execution here
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
writeStandardOutToFile = 0
if writeStandardOutToFile == 1:
    sys.stdout = open('/Users/Robert/Documents/Predict453/ClusterConsole_052117_rtv.txt', 'w')
else:
    sys.stdout = sys.__stdout__

# close any open plots
plt.close('all')

# also put plots to pdf file
# note: haven't figured out why the pdf plots lose axes and titles...
pdfdoc = PdfPages("/Users/Robert/Documents/Predict453/ClusterConsole_052117_rtv.pdf","a")

# ######################################
# get dsis
# ######################################
print "Loading DSI content"
mydoclist, dsiNums = loadDSIs()
print type(mydoclist)
print "We have loaded ", len(mydoclist), " DSIs"

dsiCount = 1
for doc in mydoclist:
    print
    print "==================="
    print "= Terms in DSI " + str(dsiCount) + " ="
    print "==================="
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    print tf.items()
    dsiCount += 1
    
vocabularyBuildLexicon = build_lexicon(mydoclist)
print 
print "vocabularyBuildLexicon"
print len(vocabularyBuildLexicon)
print vocabularyBuildLexicon
# vocabularyRTV = getWordList()
# print 
# print "vocabularyRTV"
# print vocabularyRTV

# run old or new processing
process = 'New'
# data = 2 # new rtv frequencies based on ontologies
data = 1 # new rtv data

if process == "New":
    if data == 1:
        vocabularyRTV, doc_term_matrix = getBaseRTV2()
    else:
        vocabularyRTV, doc_term_matrix = getBaseRTV3()        

print "vocabularyRTV from getBaseRTV2"
print vocabularyRTV
# print "doc_term_matrix"
# print doc_term_matrix
print
print "======================="
print "= Running " + process + " process ="
print "======================="
print 
print "=============="
print "= vocabulary ="
print "=============="
print vocabularyRTV

print
print "========================================="
print "= There are " + str(len(vocabularyRTV)) + " terms in the vocabulary ="
print "= There are " + str(len(doc_term_matrix)) + " DSIs loaded              ="            
print "========================================="

doc_term_matrix_l2 = []

print
print "========================================="
print "= Generating normalized doc_term_matrix ="
print "========================================="
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))

print
print "=============================================================="
print "= Document term matrix with row-wise L2 norms of 1 generated ="
print "=============================================================="
# print doc_term_matrix_l2

print
print "============================"
print "= Generating my_idf_vector ="
print "============================"
    
my_idf_vector = [idf(word, mydoclist) for word in vocabularyRTV]

print
print "============================"
print "= Generating my_idf_matrix ="
print "============================"

my_idf_matrix = build_idf_matrix(my_idf_vector)
print "================="
print "= my_idf_matrix ="
print "================="
print my_idf_matrix

print
print "===================================="
print "= Generating doc_term_matrix_tfidf ="
print "===================================="

doc_term_matrix_tfidf = []

for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

print
print "========================="
print "= doc_term_matrix_tfidf ="
print "========================="
# print doc_term_matrix_tfidf

#normalizing
print
print "=================================================="
print "= Generating normalized doc_term_matrix_tfidf_l2 ="
print "=================================================="
doc_term_matrix_tfidf_l2 = []
for tf_vector in doc_term_matrix_tfidf:
    doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))

print
print "============================"
print "= doc_term_matrix_tfidf_l2 ="
print "============================"
print np.matrix(doc_term_matrix_tfidf_l2) 

# now, lets run some clustering techniques
numberColumns = 1
numberRows = 10
startCluster = 2
numClusters = 2

# generate silhouette cluster analysis

print
print "==================================================="
print "= Generating silhouette cluster analysis - No PCA ="
print "==================================================="
clusterSDev(numberRows, numberColumns, startCluster, numClusters, dsiNums)

print
print "========================================================="
print "= Generating PCA Plot with First 2 Principal Components ="
print "========================================================="
pcaX(doc_term_matrix_tfidf_l2, dsiNums)

print
print "=================================================================="
print "= Generating PCA Plot Variance Explained By Principal Components ="
print "=================================================================="
if data == 2:
    n = 15
else:
    n = 30
    
pcaXNN(doc_term_matrix_tfidf_l2, dsiNums, n)

print
print "=============================================================="
print "= Generating Silhouette Analysis with 2 Principal Components ="
print "=============================================================="
clusterS(numberRows, numberColumns, startCluster, numClusters, dsiNums)

# read in dsis
synopses, dsiNums = loadDSIs()
  
plotPCA(doc_term_matrix_tfidf_l2, dsiNums)

print
print "=============================================================="
print "= Generating Cosine Similarity                               ="
print "=============================================================="
dist = getCosineSimilarity(doc_term_matrix_tfidf_l2, dsiNums)

print
print "=============================================================="
print "= Generating kmeans with Cosine Similarity distance matrix   ="
print "=============================================================="
kmeans(vocabularyRTV, dist)

print
print "=============================================================="
print "= Generating Hierarchical Clustering with Distance Matrix    ="
print "=============================================================="
denCosine = denodrogram(dist, "left", "Distance Matrix", 2.5)

print
print "=============================================================="
print "= Generating Hierarchical Clustering with TF-IDF             ="
print "=============================================================="
denTFIDF = denodrogram(doc_term_matrix_tfidf_l2, "right", "TF-IDF", 1.5)

# lda
# note: this takes time to run
print
print "=============================================================="
print "= Generating LDA                                             ="
print "=============================================================="
numberOfTopicsLDA = [2,3,4,5,6,7,8,9,10]
lda(numberOfTopicsLDA)

# close the pdf document
pdfdoc.close()
# return standard out to terminal
sys.stdout = sys.__stdout__
