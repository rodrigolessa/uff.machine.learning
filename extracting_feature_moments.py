# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
# Image Descriptors
from descriptor_zernike_moments import ZernikeMoments
# Pre-processing
from extracting_pre_processing import ImagePreProcessing
# UI
from helper_ui import UIHelper
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

def _scale(img, size, borderSize):
	"""
	Decreasing the image so the process is faster. 
	Added borders for better logo visualization.
	"""
	new = imutils.resize(img, height = size)

	if new.shape[1] > size:
		new = imutils.resize(new, width = size)

	borderColor = [0, 0, 0]

	new = cv2.copyMakeBorder(
		new, 
		top = borderSize,
		bottom = borderSize,
		left = borderSize,
		right = borderSize,
		borderType = cv2.BORDER_CONSTANT,
		value = borderColor #cv2.BORDER_REPLICATE
	)

	return new

# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--folder", required = True, help = "Path to where the files has stored")
#ap.add_argument("-e", "--extension", required = True, help = "Extension of the images")
#ap.add_argument("-i", "--index", required = True, help = "Path to where the index file will be stored")

#args = vars(ap.parse_args())

imageFolder = "C:\\caltech256"
#imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'thresholder')
imageExtension = '.jpg'
imageFinder = '{}\\*{}'.format(imageFolder, imageExtension)
imageDebug = '{}{}'.format('001_0065', imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'
imageSize = 224
imageBorderSize = 100
imageRadius = 224
zernikeDegree = 16
debug = False

# initialize our dictionary to save features
index = {}

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

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius, zernikeDegree)
# Pre-processing
ip = ImagePreProcessing(imageDebug, imageSize, False)
# Equalização baseado em histograma da imagem
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Kernel for search
kernel = np.ones((3, 3), np.uint8)
# User Interface functions
uih = UIHelper(len(imagesInFolder))

startIndexing = time.time()

# Loop over the sprite images
for path in imagesInFolder:
	
	# Extract image name, this will serve as unqiue key into the index dictionary.
	imageName = path[path.rfind('\\') + 1:].lower().replace(imageExtension, '')
	
	# Show progress bar
	uih.progress()
	
	#Pre-processing
	#outline = ip.getTheBestContour(path)
	# RGB image
	original = cv2.imread(path)
	
	# Convert it to grayscale, image with one channel 
	grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	
	# Bilateral Filter can reduce unwanted noise very well, and preserve border
	blur = cv2.bilateralFilter(grayscale, 9, 75, 75)
	##blur = cv2.bilateralFilter(grayscale, 3, 21, 21)
	##blur = cv2.bilateralFilter(grayscale, 5, 35, 35)
	##blur = cv2.bilateralFilter(grayscale, 7, 49, 49)
	##blur = cv2.bilateralFilter(grayscale, 9, 63, 63)
	##blur = cv2.bilateralFilter(grayscale, 11, 77, 77)
	# Parameters
	# Gaussian
	##blur = cv2.imshow("7", cv2.GaussianBlur(grayscale, (7, 7), 0))
	# Median blur, more simple
	#blur = cv2.medianBlur(grayscale, 5)
	# Equalização baseado em histograma da imagem
	#equalized = clahe.apply(blur)
	
	# Binarization
	# Threshold to get just the signature
	# retval, thresh_gray = cv2.threshold(blur, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
	#_, thresholder = cv2.threshold(equalized, thresh=200, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco
	# Canny edge detector finds edge like regions in the image
	# The Canny edge detector is an edge detection operator that uses 
	# a multi-stage algorithm to detect a wide range of edges in images.
	# It was developed by John F. Canny in 1986. 
	# Canny also produced a computational theory of edge detection explaining why the technique works.
	edged = cv2.Canny(blur, 30, 200)
	
	# Pode ser uma boa opção aplicar uma erosão para eliminar ruídos
	# Erosion: Shrinking the foreground
	# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
	# Redução das bordas do objeto. Consegue eliminar objetos 
	# muito pequenos mantendo somente pixels de elementos estruturantes.
	#erosion = cv2.erode(binary, kernel, iterations = 1)
	
	# 2 - Dilation: Expanding the foreground
	# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
	# Expandir as bordas do objeto, podendo preencher pixels faltantes.
	# Completar a imagem com um objeto estruturante.
	dilation = cv2.dilate(edged, kernel, iterations = 1)
	
	# Crop
	normalized = _scale(dilation, imageSize, imageBorderSize)

	bitwised = cv2.bitwise_not(normalized)
	# outline = np.zeros((200,200), dtype = "uint8")

	# # Initialize the outline image,
	# # find the outermost contours (the outline) of the object, 
	# # cv2.RETR_EXTERNAL - telling OpenCV to find only the outermost contours.
	# # cv2.CHAIN_APPROX_SIMPLE - to compress and approximate the contours to save memory
	# #img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# (contours, hierarchy) = cv2.findContours(normalized.copy(),
	# 	cv2.RETR_EXTERNAL, 
	# 	cv2.CHAIN_APPROX_SIMPLE)

	# # Sort the contours based on their area, in descending order. 
	# # keep only the largest contour and discard the others.
	# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

	# # The outline is drawn as a filled in mask with white pixels:
	# for cnt in contours:

	# 	if(cv2.contourArea(cnt) > 0):

	# 		# cv2.arcLength and cv2.approxPolyDP. 
	# 		# These methods are used to approximate the polygonal curves of a contour.
	# 		#peri = cv2.arcLength(cnt, True)

	# 		# Level of approximation precision. 
	# 		# In this case, we use 2% of the perimeter of the contour.
	# 		# * The Ramer–Douglas–Peucker algorithm, also known as the Douglas–Peucker algorithm and iterative end-point fit algorithm, 
	# 		# is an algorithm that decimates a curve composed of line segments to a similar curve with fewer point
	# 		#approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

	# 		cv2.drawContours(outline, [cnt], -1, 255, -1)

	# if (outline.any()):
	# 	outline = cv2.bitwise_not(outline)

	# Debugging: Show the original binary image and its boundary
	#debugging = np.hstack((grayscale, equalized, blur))
	# Debugging:
	if debug:
		#cv2.imshow("Debugging: Grayscale + Equalization + Threshold", debugging)
		cv2.imshow("Original", original)
		cv2.imshow("Grayscale", grayscale)
		cv2.imshow("Blur", blur)
		#cv2.imshow("Equalization", equalized)
		#cv2.imshow("Threshold", thresholder)
		cv2.imshow("Edge", edged)
		cv2.imshow("Normalize", normalized)
		cv2.imshow("Bitwised", bitwised)
		cv2.imshow("Dilation", dilation)
		#cv2.imshow("Outline", outline)
		cv2.waitKey(0)
	
	# TODO: Draw center of mass
	cv2.imwrite("{}\\{}{}".format(imageFolderThreshold, imageName, imageExtension), bitwised)
	
	# Compute Zernike moments to characterize the shape of object outline
	moments = zm.describe(bitwised)

	# Debugging: analyse descriptions of form
	#if imageName.find(imageDebug) >= 0:
		#print(moments.shape)
		#print('{}: {}'.format(imageName, moments))

	# then update the index
	index[imageName] = moments

# cPickle for writing the index in a file
with open(imageMomentsFile, "wb") as outputFile:
	cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)

doneIndexing = time.time()

elapsed = (doneIndexing - startIndexing) / 1000

print(" ")
print("Tempo de execução:")
print(elapsed)

cv2.destroyAllWindows()