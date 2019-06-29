#Importing all the needed libraries
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Loading the dataset
iris = datasets.load_iris()

#Converting into Datafarme
x = pd.DataFrame(iris.data)

x.head()

x.columns = ['sepal_length','sepal_width','petal_length','petal_width']

print('')
print('Head')
print(x.head())

# Finding the optimum number of clusters for k-means clustering
Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
print('')
print('kmeans')
print(kmeans)

score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
print('')
print('score')
print(score)

pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

#Implementation of K-Means Clustering
model = KMeans(n_clusters = 3)
model.fit(x)

print('')
print('Predict labels')
print(model.labels_)

#Accuracy of K-Means Clustering
accuracy = accuracy_score(iris.target, model.labels_)

print('')
print('accuracy')
print(accuracy)

# Plot

colormap = np.array(['Red', 'Blue', 'Green'])

z = plt.scatter(x.sepal_length, x.sepal_width, x.petal_length, c = colormap[model.labels_])

x.insert(4, "class", model.labels_)
x.replace({'class': {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}})

plt.show()