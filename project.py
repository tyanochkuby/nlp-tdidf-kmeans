from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from time import time

J = lambda X, labels: silhouette_score(X, labels, metric='euclidean', sample_size=None, random_state=None)



df = pd.read_csv(r'D:\hoojnia\py\PJN\projekt\nlp-tdidf-kmeans\Emails.csv')

emails = df['text'].to_numpy()
tfidf = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words = 'english',
    max_features=2000
)
tfidf_matrix = tfidf.fit_transform(emails)

best_k = 2
best_j = 100000
best_clustering_model = KMeans(n_clusters=1, random_state=0, n_init="auto")

print("k\t\ttime\tsilhouette")
formatter_result = ("{:9s}\t{:.3f}s\t{:.3f}")

for k in range(2, 10):
    next_kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")

    t0 = time()
    next_kmeans.fit_predict(tfidf_matrix)
    fit_time = time() - t0
    
    j = J(tfidf_matrix, next_kmeans.labels_)
    print(formatter_result.format(str(k), fit_time, j))

    if j < best_j:
        best_j = j
        best_k = k
        best_clustering_model = next_kmeans

print(f'\nBest k: {best_k}\n\n\n')

kilka = 4
best_clustering_model.fit_predict(tfidf_matrix)

for i in range(best_k):
    print(f'Kilka maili {i+1} kategorii:\n')
    j = 0
    for label_n in range(len(best_clustering_model.labels_)):
        if best_clustering_model.labels_[label_n] == i:
            print(emails[label_n])
            print('\n\n')
            j+=1
            if j == kilka:
                break
    print('=======================')

x = tfidf_matrix.toarray()
reduced_data = PCA(n_components=2).fit_transform(x)
fig, ax = plt.subplots()
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=best_clustering_model.labels_)

plt.show()