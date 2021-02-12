#Importing all the needed libraries
import pandas as pd
import numpy as np
import pylab as pl
import pickle as cp
import matplotlib.pyplot as plt
#import sklearn.metrics as sm
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Loading the dataset
#iris = datasets.load_iris()

imageMomentsFile = 'index.pkl'

# Read pickle file to a dict
#unpickled_df = pd.read_pickle(imageMomentsFile)
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

# using list comprehension 
#print(sum([len(sparse_matrix[x]) for x in sparse_matrix if isinstance(sparse_matrix[x], dict)])) 
print(str(len(sparse_matrix)) + ' itens/imagens no total')
# print('5    - primeiros itens:')
# for k in list(sparse_matrix.keys())[:5]:
#     print(' - ' + k)

# Original labels
labels_true = pd.factorize([k.split('_')[0] for k in sparse_matrix.keys()])[0]

# print(labels_true[:200])

# Convert the dict to a numpy array
features = np.array(list(sparse_matrix.values()))

#Converting into Datafarme
x = pd.DataFrame(features)

x.head()

#Preprocessing
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 4].values
X = x.values
y = labels_true

#Creating training and test splits
testsize = 0.15
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=10)

 
#Performing Feature Scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='black', linestyle='dashed', 
    marker='o',
    markerfacecolor='grey', 
    markersize=10)
plt.title('Erro %')
plt.xlabel('K')
plt.ylabel('Erro')
#plt.show()

# menor erro k = error.index(min(error))
print('min error:' + str(min(error)))
print('k:' + str(error.index(min(error))))

print("test size: "+str(testsize)+" %")

pesos = 'uniform' # uniform distance 
print("weights: "+pesos)
algoritimo = 'brute' # ball_tree kd_tree brute auto
print("algoritimo: "+algoritimo)
Power  = 2  # 1 - manhattan 2 - euclidean
print("p: "+str("manhattan" if Power == 1 else "euclidean") )

classifier = KNeighborsClassifier(
    n_neighbors=error.index(min(error)), 
    weights = pesos,
    algorithm = algoritimo,
    p = Power
    )

classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)

print('confusion_matrix:')
print(confusion_matrix(y_test, y_pred))
print('report:')
print(classification_report(y_test, y_pred))

a = 1