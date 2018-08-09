from __future__ import print_function
from kmodes.kmodes import KModes
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples

pd.set_option('display.max_colwidth', -1)
pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=np.nan)

cars = pd.read_csv('/home/saibharadwaj/Downloads/cars.csv', delimiter=',')

print(cars.columns.values)
print(cars.shape)

uniqueOrigins = cars['Origin'].unique()
print (uniqueOrigins)

for numClusters in range(2, 13):
    km = KModes(n_clusters=numClusters, init='Huang', n_init=10, verbose=0)     # Ideal is 4

    clusters = km.fit_predict(cars)

    # print('Centroids are')
    # print(km.cluster_centroids_)
    # print(type(km.labels_), '  ', len(km.labels_))

    labels = list(km.labels_)
    # print('labels are ', labels)

    silhouette_avg = silhouette_score(cars[['MPG', 'Displacement']], clusters)

    print('For ', km.n_clusters, 'avg silhouette score is ', silhouette_avg)

    # sample_silhouette_values = silhouette_samples(cars[['MPG', 'Displacement']], labels)
    # print(sample_silhouette_values)



# c0 = [i for i in range(0, len(labels)) if labels[i] == 0]
#
# print(cars.iloc[c0])

# Update the data with cluster IDs
# clusterMap = pd.DataFrame()
# cars['index'] = cars.index.values
# clusterMap['index'] = cars.index.values
# clusterMap['cluster'] = km.labels_
#
# cars = pd.merge(left=cars, right=clusterMap, on='index')
# print(cars[cars['cluster'] == 1])


