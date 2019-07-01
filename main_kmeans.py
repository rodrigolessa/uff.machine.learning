#Importing all the needed libraries
import pandas as pd
import numpy as np
import pylab as pl
import pickle as cp
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Loading the dataset
#iris = datasets.load_iris()

imageMomentsFile = 'index.pkl'

# Read pickle file to a dict
#unpickled_df = pd.read_pickle(imageMomentsFile)
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

# using list comprehension 
#print(sum([len(sparse_matrix[x]) for x in sparse_matrix if isinstance(sparse_matrix[x], dict)])) 
print(str(len(sparse_matrix)) + ' itens/imagens no total:')
print('5    - primeiros itens:')
for k in list(sparse_matrix.keys())[:5]:
    print(' - ' + k)

# Original labels
labels_true = pd.factorize([k.split('_')[0] for k in sparse_matrix.keys()])[0]

print(labels_true[:200])

# Convert the dict to a numpy array
features = np.array(list(sparse_matrix.values()))

#Converting into Datafarme
x = pd.DataFrame(features)

#x.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24']
x.columns = ['z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20','z21','z22','z23','z24','z25','z26','z27','z28','z29','z30','z31','z32','z33','z34','z35','z36','z37','z38','z39','z40','z41','z42','z43','z44','z45','z46','z47','z48','z49','z50','z51','z52','z53','z54','z55','z56','z57','z58','z59','z60','z61','z62','z63','z64','z65','z66','z67','z68','z69','z70','z71','z72','z73','z74','z75','z76','z77','z78','z79','z80']

print('')
print('Head')
print(x.head())

# Finding the optimum number of clusters for k-means clustering
###Nc = range(1, 25)
###kmeans = [KMeans(n_clusters=i) for i in Nc]
#print('kmeans')
#print(kmeans)
###score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
#print('score')
#print(score)

#pl.plot(Nc,score)
#pl.xlabel('Number of Clusters')
#pl.ylabel('Score')
#pl.title('Elbow Curve')
#pl.show()

# Implementation of K-Means Clustering
# Com 10 classes conhecidas
model = KMeans(n_clusters = 10)
model.fit(x)

print('')
print('Predict labels')
print(model.labels_)

#Accuracy of K-Means Clustering
accuracy = accuracy_score(labels_true, model.labels_)

print('')
print('accuracy')
print(accuracy)

# Plot

colormap = np.array(['Red', 'Blue', 'Green', 'Cyan', 'Magenta', 'Yellow', 'Black', 'Blue', 'Green', 'Red'])

z = plt.scatter(x.z1, x.z7, x.z21, c = colormap[model.labels_])

x.insert(11, "class", model.labels_)
x.replace({'class': {0: "Ak47", 1: "Backpack", 2: "Bat", 3: "Glove", 4: "Elephant", 5: "Homer", 6: "Traffic-light", 7: "Umbrella", 8: "Zebra", 9: "Airplanes"}})

plt.show()