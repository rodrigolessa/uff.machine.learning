# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
# Image Descriptors
from descriptor_zernike_moments import ZernikeMoments
# Pre-processing
from image_pre_processing import ImagePreProcessing
# Math
import numpy as np
# Data
import pandas as pd
import pickle as cp
import glob
# Image
import cv2
import imutils
# Other
import os
import sys
import time

# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--folder", required = True, help = "Path to where the files has stored")
#ap.add_argument("-e", "--extension", required = True, help = "Extension of the images")
#ap.add_argument("-i", "--index", required = True, help = "Path to where the index file will be stored")

#args = vars(ap.parse_args())

imageFolder = "C:\\caltech256\\251.airplanes-101"
#imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'thresholder')
imageExtension = '.jpg'
imageFinder = '{}\\*{}'.format(imageFolder, imageExtension)
imageDebug = '{}{}'.format('251_0002', imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'
imageSize = 300
imageRadius = 180
zernikeDegree = 8
debug = False

# initialize our dictionary to save features
index = {}

qtd = len(imagesInFolder)

i = 1

#print(imageFinder)
#print('images in the folder: {}'.format(qt))

try:
	# If index file exists, try to delete
    os.remove(imageMomentsFile)
except OSError:
    pass

try:
	# If folder to hold thresholder exists, try to delete
    os.makedirs(imageFolderThreshold)
except OSError as e:
	pass
	#import errno
    #if e.errno != errno.EEXIST:
    #raise

# Simulate a progress bar
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius, zernikeDegree)
# Pre-processing
ip = ImagePreProcessing(imageDebug, imageSize, False)
# Equalização baseado em histograma da imagem
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

startIndexing = time.time()

# Loop over the sprite images
for path in imagesInFolder:
	
	# Extract image name, this will serve as unqiue key into the index dictionary.
	imageName = path[path.rfind('\\') + 1:].lower().replace(imageExtension, '')

	progress(i, qtd)
	
	#Pre-processing
	#outline = ip.getTheBestContour(path)

	# RGB image
	original = cv2.imread(path)
	# Convert it to grayscale, image with one channel 
	grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	# Bilateral Filter can reduce unwanted noise very well, and preserve border
	##blur = cv2.bilateralFilter(equalized, 9, 75, 75)
	##blur = cv2.bilateralFilter(equalized, 3, 21, 21)
	##blur = cv2.bilateralFilter(equalized, 5, 35, 35)
	##blur = cv2.bilateralFilter(equalized, 7, 49, 49)
	##blur = cv2.bilateralFilter(equalized, 9, 63, 63)
	##blur = cv2.bilateralFilter(equalized, 11, 77, 77)
	# Parameters
	# Gaussian
	##blur = cv2.imshow("7", cv2.GaussianBlur(equalized, (7, 7), 0))
	# Median blur, more simple
	blur = cv2.medianBlur(grayscale, 5)
	# Equalização baseado em histograma da imagem
	equalized = clahe.apply(blur)
	# Binarization
	# Threshold to get just the signature
	# retval, thresh_gray = cv2.threshold(blur, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
	_, thresholder = cv2.threshold(equalized, thresh=200, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco
	# Canny edge detector finds edge like regions in the image
	# The Canny edge detector is an edge detection operator that uses 
	# a multi-stage algorithm to detect a wide range of edges in images.
	# It was developed by John F. Canny in 1986. 
	# Canny also produced a computational theory of edge detection explaining why the technique works.
	edged = cv2.Canny(thresholder, 30, 200)
	# Pode ser uma boa opção aplicar uma erosão para eliminar ruídos
	kernel = np.ones((3, 3), np.uint8)
	# Erosion: Shrinking the foreground
	# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
	# Redução das bordas do objeto. Consegue eliminar objetos 
	# muito pequenos mantendo somente pixels de elementos estruturantes.
	#erosion = cv2.erode(binary, kernel, iterations = 1)
	# Debugging: Show the original binary image and its boundary
	#debugging = np.hstack((grayscale, equalized, blur))
	# Debugging:
	if debug:
		#cv2.imshow("Debugging: Grayscale + Equalization + Threshold", debugging)
		cv2.imshow("Original", original)
		cv2.imshow("Grayscale", grayscale)
		cv2.imshow("Blur", blur)
		cv2.imshow("Equalization", equalized)
		cv2.imshow("Threshold", thresholder)
		cv2.imshow("Edge", edged)
		cv2.waitKey(0)
	# TODO: Draw center of mass
	#cv2.imwrite("{}\\{}.jpg".format(imageFolderThreshold, imageName), outline)
	# Compute Zernike moments to characterize the shape of object outline
	moments = zm.describe(thresholder)

	# Debugging: analyse descriptions of form
	#if imageName.find(imageDebug) >= 0:
		#print(moments.shape)
		#print('{}: {}'.format(imageName, moments))

	# then update the index
	index[imageName] = moments

	i+=1

# cPickle for writing the index in a file
with open(imageMomentsFile, "wb") as outputFile:
	cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)

doneIndexing = time.time()

elapsed = (doneIndexing - startIndexing) / 1000

print(" ")
print(elapsed)

cv2.destroyAllWindows()