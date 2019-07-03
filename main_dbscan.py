from sklearn import metrics
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from time import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as cp
import glob
import imutils
import os
import sys

def ami_score(U, V):
    return metrics.adjusted_mutual_info_score(U, V, average_method='arithmetic')


def uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                             fixed_n_classes=None, n_runs=5, seed=42):
    """Compute score for 2 random uniform cluster labelings.

    Both random labelings have the same number of clusters for each value
    possible value in ``n_clusters_range``.

    When fixed_n_classes is not None the first labeling is considered a ground
    truth class assignment with fixed number of classes.
    """
    random_labels = np.random.RandomState(seed).randint
    scores = np.zeros((len(n_clusters_range), n_runs))

    if fixed_n_classes is not None:
        labels_a = random_labels(low=0, high=fixed_n_classes, size=n_samples)

    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            if fixed_n_classes is None:
                labels_a = random_labels(low=0, high=k, size=n_samples)
            labels_b = random_labels(low=0, high=k, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


imageMomentsFile = 'index.pkl'

# Read pickle file to a dict
#unpickled_df = pd.read_pickle(imageMomentsFile)
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

#print(str(len(sparse_matrix)) + ' itens/imagens no total:')

# Original labels
labels_true = pd.factorize([k.split('_')[0] for k in sparse_matrix.keys()])[0]

#print(labels_true)

# Convert the dict to a numpy array
x = np.array(list(sparse_matrix.values()))
#x = df.values #returns a numpy array

#print(x)

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

# #Converting into Datafarme
# x = pd.DataFrame(features)
df = pd.DataFrame(x_scaled)

#df.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24']
df.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24','z25','z26','z27','z28','z29','z30','z31','z32','z33','z34','z35','z36','z37','z38','z39','z40','z41','z42','z43','z44','z45','z46','z47','z48','z49','z50','z51','z52','z53','z54','z55','z56','z57','z58','z59','z60','z61','z62','z63','z64','z65','z66','z67','z68','z69','z70','z71','z72','z73','z74','z75','z76','z77','z78','z79','z80']

#print('Head')
#print(df.head())

raio = .007
minPts = 3

# O data set de imagemMPEG7 possui 69 grupos, mas utilizamos somente 10
dbscan = DBSCAN(eps = raio, metric = 'cosine', min_samples = minPts).fit(df)

#print(dbscan.labels_[:50])

# Return sequencial labels
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)

print('')
print('Execução do algoritmo DBSCAN utilizando os seguintes parâmetros:')
print('Raio: ' + str(raio))
print('Mínimo de objetos: ' + str(minPts))
print('Distância: distância do Cosseno')
print('')
print('Total de objetos: %d' % len(labels_true))
print('Número de grupos estimado: %d' % n_clusters_)
print('Número de ruídos/outliers estimado: %d' % n_noise_)
print("Informação Mútua Ajustada (AMI): %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
# The Rand Index computes a similarity measure between two clusterings by considering
# all pairs of samples and counting pairs that are assigned in the same
# or different clusters in the predicted and true clusterings.
print("Indice de Rand Ajustado: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Homogeneidade: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completude: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))

print("Coeficiente de silhueta: %0.3f" % metrics.silhouette_score(x, labels))

#print("Fowlkes-Mallows: %0.3f" % metrics.fowlkes_mallows_score(labels_true, labels))


# TODO: Testar
#ax = fig.add_subplot(geo + 5, projection='3d', title='dbscan')

core = dbscan.core_sample_indices_
#print(repr(core))

#size = [5 if i not in core else 40 for i in range(len(x))]
#print(repr(size))

# #############################################################################
# Plot result

#print('')

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

plt.title('Distribuição dos grupos: %d' % n_clusters_)
#plt.show()


###############################
# https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py

score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    ami_score,
    metrics.mutual_info_score,
]


# Random labeling with varying n_clusters against ground class labels
# with fixed number of clusters

n_samples = 1845
n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
n_classes = 10

plt.figure(2)

plots = []
names = []
for score_func in score_funcs:
    #print("Computing %s for %d values of n_clusters and n_samples=%d" % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                                      fixed_n_classes=n_classes)
    #print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for random uniform labeling\n"
          "against reference assignment with %d classes" % n_classes)
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names)
#plt.show()