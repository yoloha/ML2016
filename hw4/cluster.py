from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from random import randint



import sys
import pandas as pd
import pickle
import nltk
import os
import numpy as np
import string
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

dir_doc = sys.argv[1] + 'docs.txt'
dir_titles = sys.argv[1] + 'title_StackOverflow.txt'
with open(dir_doc) as f:
    docs = f.read().splitlines()
with open(dir_titles) as f:
    titles = f.read().splitlines()
with open('stopwords.txt') as f:
    stopwords = f.read().splitlines()



print "Eliminating stopwords from docs and titles"
for i in range(len(docs)):
    docs[i] = docs[i].translate(replace_punctuation)
    docs[i] = ' '.join([''.join([c for c in word if not c.isdigit()]) for word in docs[i].split()])
    docs[i] = ' '.join([word.lower() for word in docs[i].split() if word.lower() not in stopwords])
    
for i in range(len(titles)):
    titles[i] = titles[i].translate(replace_punctuation)
    titles[i] = ' '.join([''.join([c for c in word if not c.isdigit()]) for word in titles[i].split()])
    titles[i] = ' '.join([word.lower() for word in titles[i].split() if word.lower() not in stopwords])
    

total = docs + titles


print "Extracting features from the training dataset using a sparse vectorizer"
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
vectorizer.fit(titles)
X = vectorizer.transform(titles)
print "n_samples: %d, n_features: %d" % X.shape

print "Performing dimensionality reduction using LSA"
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
r1 = 1#randint(0,10000)
r2 = 1#randint(0,10000)
true_k = 53
svd = TruncatedSVD(n_components=20, random_state=r1)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X) 

explained_variance = svd.explained_variance_ratio_.sum()
print "Explained variance of the SVD step: {}%".format(int(explained_variance * 100))


km = KMeans(n_clusters=true_k, init='k-means++', n_jobs=-1, max_iter=1000, n_init=100, verbose=False, random_state=r2)
print "Clustering sparse data with %s" % km
km.fit(X)

ids = range(len(titles))
clusters = km.labels_.tolist()
stack = { 'title': titles, 'indexes': ids, 'cluster': clusters }

frame = pd.DataFrame(stack, index = [clusters] , columns = ['title', 'indexes', 'cluster'])

#sort cluster centers by proximity to centroid
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print "Cluster %d words:" % i
    for ind in order_centroids[i, :5]: #replace 6 with n words per cluster
        print "\t\t%s" % terms[ind]
    
    print "Cluster %d titles:" % i
    for ind in range(5):
        print "\t\t[ %d" % frame.ix[i]['indexes'].values.tolist()[ind], "] %s" % frame.ix[i]['title'].values.tolist()[ind]
    
# Check clusters' distribution
a = frame['cluster'].value_counts() #number of titles per cluster
print a







id_cluster = np.array(frame['cluster'])
dir_check = sys.argv[1] + 'check_index.csv'
with open(dir_check) as f:
    check = f.read().splitlines()
check = check[1:]
output = np.zeros((len(check)))
for i in range(len(check)):
    word = check[i].split(',')
    id1 = int(word[1])
    id2 = int(word[2])
    output[i] = (id_cluster[id1] == id_cluster[id2])

f = open(sys.argv[2], 'w')
f.write("ID,Ans\n")
for i in range(len(check)):
    f.write(str(i) + "," + str(int(output[i])) + "\n")
