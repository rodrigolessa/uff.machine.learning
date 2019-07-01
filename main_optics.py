from sklearn import metrics
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pandas as pd
import cv2
import matplotlib.gridspec as gridspec
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

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

# #Converting into Datafarme
# x = pd.DataFrame(features)
df = pd.DataFrame(x_scaled)

df.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24']

# print('Head')
# print(x.head())

# Conhecemos o dataset que possui 10 categorias
clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

# Run the fit
clust.fit(x)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.06)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(x))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = x[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(x[clust.labels_ == -1, 0], x[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = x[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(x[labels_050 == -1, 0], x[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = x[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(x[labels_200 == -1, 0], x[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()

# # #############################################################################
labels_predict = clust.labels_

# # Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_predict)) - (1 if -1 in labels else 0)
if n_clusters_ > 1:

    n_noise_ = list(labels_predict).count(-1)

    print('')
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_predict))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_predict))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_predict))
    # # The Rand Index computes a similarity measure between two clusterings by considering 
    # # all pairs of samples and counting pairs that are assigned in the same 
    # # or different clusters in the predicted and true clusterings.
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels_predict))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels_predict, average_method='arithmetic'))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels_predict))

# # #############################################################################
# # Plot result

# print('')

# core_samples_mask = np.zeros_like(labels, dtype=bool)

# core_samples_mask[dbscan.core_sample_indices_] = True

# # Black removed and is used for noise instead.
# unique_labels = set(labels)

# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]

# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = x[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = x[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()