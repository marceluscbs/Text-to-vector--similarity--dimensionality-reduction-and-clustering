#!/usr/bin/env python
# coding: utf-8

# # In-Class Assignment 2 and 3

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Use 10 tweets 
text_1 = "INSERT YOUR INFO"
text_3 = "INSERT YOUR INFO"
text_2 = "INSERT YOUR INFO"
text_4 = "INSERT YOUR INFO"
text_5 = "INSERT YOUR INFO"
text_6 = "INSERT YOUR INFO"
text_7 = "INSERT YOUR INFO"
text_8 = "INSERT YOUR INFO"
text_9 = "INSERT YOUR INFO"
text_10 = "INSERT YOUR INFO"


# In[ ]:


#Getting vectors from text using TfIdf
def get_vectors(*str):
    text = [t for t in str]
    tfidf_vectorizer = TfidfVectorizer(text)
    tfidf_vectorizer.fit(text)

    return tfidf_vectorizer.transform(text).toarray()

#Getting cosine similarity between vectors
def get_cosine_sim(vectors):
    return cosine_similarity(vectors)


# In[ ]:


# Converts those texts to vectors
vectors = get_vectors(text_1,text_2,text_3,text_4,text_5,text_6,text_7,text_8,text_9,text_10)
vectors


# In[ ]:


# Calculate similarity amount the texts
cos_sim = get_cosine_sim(vectors)
cos_sim 


# In[ ]:


# Reduce dimensionality to 2-D with PCA
plt.rcParams['figure.figsize'] = (12,8)

pca = PCA(n_components=2)
cos_sim_pca = pca.fit_transform(cos_sim)

n = cos_sim_pca.shape[0]

for i in range(cos_sim_pca.shape[0]):
    label = "Text- " + str(i + 1)
    x = cos_sim_pca[i, 0]
    y = cos_sim_pca[i, 1]
    
    plt.scatter(x,y, s=400)
    plt.text(x+.03, y+.03, label, fontsize=9)
    
plt.show()


# In[ ]:


#Use a clustering algorithm to form clusters
X_new=cos_sim_pca

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X_new)

print(kmeans.cluster_centers_)
print(kmeans.labels_)


# In[ ]:


# Draw a scatter plot of the clusters
plt.rcParams['figure.figsize'] = (20,14)
plt.scatter(X_new[:,0], X_new[:,1], c=kmeans.labels_, s=200)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*', c="black", s=200)
plt.text(x+.03, y+.03, label, fontsize=12)

for i in range(cos_sim_pca.shape[0]):
    label = "Text- " + str(i + 1)
    x = cos_sim_pca[i, 0]
    y = cos_sim_pca[i, 1]
    
    plt.scatter(x,y, s=400)
    plt.text(x+.03, y+.03, label, fontsize=12)

plt.show()

