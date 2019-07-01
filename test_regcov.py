# Import the necessary packages
# Image Descriptors
from descriptor_zernike_moments import ZernikeMoments
# Math
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import cv2
import pickle as cp
import glob
import imutils
import os
import sys
import time

imageFolder = "C:\\caltech256"
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'thresholder')
imageExtension = '.jpg'
imageFinder = '{}\\*{}'.format(imageFolder, imageExtension)
imageDebugName = '235_0020' # 064_0028, 003_0029, 004_0015, 005_0008, 104_0007, 226_0050, 235_0020, 250_0079, 251_0014
imageDebug = '{}{}'.format(imageDebugName, imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'
# Regcov
regcovFolder = 'data\\regcov'
regcovInFolder = glob.glob('{}\\*{}'.format(regcovFolder, '.txt'))
regcovFile = 'index_regcov.pkl'
# Number of interations
ni = 0 # global variable
niMax = 3 # global variable
# Names of the images visited
visit = [] # global array

# initialize our dictionary to save features
index = {}

print('Regcov features')

#path = Path('data\\regcov')
#path.ls()

# , engine = 'python'
dsak40001 = pd.read_csv('data\\regcov\\001_0001.jpg.txt', sep=' ', header=None)
dsak40002 = pd.read_csv('data\\regcov\\001_0002.jpg.txt', sep=' ', header=None)

# 28 colunas - 21 linhas
#print(dsak40001)

from sklearn.metrics.pairwise import cosine_similarity
d = cosine_similarity(dsak40001, dsak40002)

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
dsak0001 = np.loadtxt('data\\regcov\\001_0001.jpg.txt')
dsak0002 = np.loadtxt('data\\regcov\\001_0002.jpg.txt')

#print(dsak0001)

(line, column) = dsak0002.shape

arr_dsak0001 = np.reshape(dsak0001, line * column)
arr_dsak0002 = np.reshape(dsak0002, line * column)

#print(arr_dsak0001)

d = dist.cosine(arr_dsak0001, arr_dsak0002)

print(d)

for path in regcovInFolder:
	
    # Extract image name, this will serve as unqiue key into the index dictionary.
    regname = path[path.rfind('\\') + 1:].lower().replace(imageExtension, '').replace('.txt', '')

    matrix = np.loadtxt(path)
    (line, column) = matrix.shape
    vector = np.reshape(matrix, line * column)

    index[regname] = vector

# cPickle for writing the index in a file
with open(regcovFile, "wb") as outputFile:
    cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)
 


####################################################################################
# Pandas Dataframe
#unpickled_df = pd.read_pickle(imageMomentsFile)

# Load dictionary of features
# with open(imageMomentsFile, 'rb') as pickle_file:
#     sparse_matrix = cp.load(pickle_file)

# print('')
# print('Moments/features from images:')
# print(list(sparse_matrix.values())[:1])

# def searcher(imgName):
#     """
#     Search for similirity
#     """

#     # Set global
#     global ni, visit

#     print('')
#     print('Search for images similar to "{}"'.format(imgName))
#     query = sparse_matrix[imgName]

#     # Mark de image
#     visit.append(imgName)

#     # initialize our dictionary of results
#     # my_dict = {} or my_dict = dict()
#     # my_dict = {'key':'value', 'another_key' : 0}
#     # my_dict.update({'third_key' : 1})
#     # del my_dict['key']
#     results = {}

#     # loop over the images in our index
#     for (k, features) in sparse_matrix.items():
#         # Compute the distance between the query features
#         # and features in our index, then update the results
#         #d = dist.euclidean(query, features)
#         d = dist.cosine(query, features)
#         #d = np.linalg.norm(query - features)
#         #print('Caracterídtica: {} - {}'.format(features[0], features[1]))
#         #print('Distance: {}'.format(d))
#         results[k] = d

#     # Sort our results, where a smaller distance indicates
#     # higher similarity
#     results = sorted([(v, k) for (k, v) in results.items()])[:niMax]

#     ni+=1

#     imgNameNew = ''

#     print('')
#     for r in results:
#         #imageZeros = '{-:0>3}'.format(imageNumber)
#         if imgNameNew == '' and (r[1] not in visit):
#             imgNameNew = r[1]
#         print('The object "{}" - similarity: {}'.format(r[1], r[0]))

#     if ni <= niMax and imgNameNew != '':
#         searcher(imgNameNew)

# # Start
# searcher(imageDebugName)