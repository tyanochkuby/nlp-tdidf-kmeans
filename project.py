from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


k = 4

df = pd.read_csv(r'J:\hoojnia\py\PJN\projekt\Emails.csv')
# df['text'] = df['text'].apply(
#     lambda x: x.split(' ', 1)[1]
# )
emails = df['text'].to_numpy()
print(emails[0])
tfidf = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words = 'english',
    max_features=2000
)
tfidf_matrix = tfidf.fit_transform(emails)

clustering_model = KMeans(n_clusters=k, random_state=0, n_init="auto")

labels = clustering_model.fit_predict(tfidf_matrix)

kilka = 5
for i in range(k):
    print('Kilka maili {} kategorii:\n'.format(i+1))
    j = 0
    for label_n in range(len(clustering_model.labels_)):
        if clustering_model.labels_[label_n] == i:
            print(emails[label_n])
            print('\n\n')
            j+=1
            if j == kilka:
                break
    print('=======================')

x = tfidf_matrix.toarray()
reduced_data = PCA(n_components=2).fit_transform(x)
fig, ax = plt.subplots()
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=clustering_model.labels_)

plt.show()