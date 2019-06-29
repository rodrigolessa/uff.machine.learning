from sklearn import metrics
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle as cp
import glob
import imutils
import os
import sys

imageMomentsFile = 'index.pkl'

# Read pickle file to a dict
#unpickled_df = pd.read_pickle(imageMomentsFile)
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

print(str(len(sparse_matrix)) + ' itens/imagens no total:')

# Original labels
labels_true = pd.factorize([k.split('_')[0] for k in sparse_matrix.keys()])[0]

# Convert the dict to a numpy array
x = np.array(list(sparse_matrix.values()))
#x = df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

# #Converting into Datafarme
# x = pd.DataFrame(features)
df = pd.DataFrame(x_scaled)

df.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24']

#print('Head')
print(df.head())

# O data set de imagemMPEG7 possui 69 grupos
dbscan = DBSCAN(eps=0.01, metric='cosine', min_samples=3).fit(df)

#print(dbscan.labels_[:50])

# Return sequencial labels
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)

print('')
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# The Rand Index computes a similarity measure between two clusterings by considering
# all pairs of samples and counting pairs that are assigned in the same
# or different clusters in the predicted and true clusterings.
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))

# TODO: Testar
#ax = fig.add_subplot(geo + 5, projection='3d', title='dbscan')

core = dbscan.core_sample_indices_
#print(repr(core))

#size = [5 if i not in core else 40 for i in range(len(x))]
#print(repr(size))

# #############################################################################
# Plot result

print('')

core_samples_mask = np.zeros_like(labels, dtype=bool)

core_samples_mask[dbscan.core_sample_indices_] = True

# Black removed and is used for noise instead.
unique_labels = set(labels)

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = x[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = x[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()