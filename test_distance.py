from zernike_moments import ZernikeMoments
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

imageFolder = "D:\\mpeg7"
#imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'threshold')
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageDebugName = 'butterfly-1'
imageDebug = '{}{}'.format(imageDebugName, imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'
# Number of interations
ni = 0 # global variable
niMax = 5 # global variable
# Names of the images visited
visit = [] # global array
 
# Pandas Dataframe
#unpickled_df = pd.read_pickle(imageMomentsFile)

# Load dictionary of features
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

print('')
print('Moments/features from images:')
print(list(sparse_matrix.values())[:1])

def searcher(imgName):
    """
    Search for similirity
    """

    # Set global
    global ni, visit

    print('')
    print('Search for images similar to "{}"'.format(imgName))
    query = sparse_matrix[imgName]

    # Mark de image
    visit.append(imgName)

    # initialize our dictionary of results
    # my_dict = {} or my_dict = dict()
    # my_dict = {'key':'value', 'another_key' : 0}
    # my_dict.update({'third_key' : 1})
    # del my_dict['key']
    results = {}

    # loop over the images in our index
    for (k, features) in sparse_matrix.items():
        # Compute the distance between the query features
        # and features in our index, then update the results
        #d = dist.euclidean(query, features)
        d = dist.cosine(query, features)
        #d = np.linalg.norm(query - features)
        #print('Caracterídtica: {} - {}'.format(features[0], features[1]))
        #print('Distance: {}'.format(d))
        results[k] = d

    # Sort our results, where a smaller distance indicates
    # higher similarity
    results = sorted([(v, k) for (k, v) in results.items()])[:7]

    ni+=1

    imgNameNew = ''

    print('')
    for r in results:
        #imageZeros = '{-:0>3}'.format(imageNumber)
        if imgNameNew == '' and (r[1] not in visit):
            imgNameNew = r[1]
        print('The object "{}" - similarity: {}'.format(r[1], r[0]))

    if ni <= niMax and imgNameNew != '':
        searcher(imgNameNew)

# Start
searcher(imageDebugName)