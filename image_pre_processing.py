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

class ImagePreProcessing:

	def __init__(self, imageDebug, size, whiteBackground=False):
		#self.path = path
		self.size = size
		self.whitebg = whiteBackground
		self.imageForDebugging = imageDebug
		#self.original = cv2.imread(path)
		#self.grayscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
		#self.threshold = self._get_threshold()

	def get_threshold(self, img):
		"""
		Threshold to get just the signature
		"""
		# Convert it to grayscale, image with one channel 
		grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Bilateral Filter can reduce unwanted noise very well, and preserve border
		blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

		# For segmentation: Flip the values of the pixels 
		# (black pixels are turned to white, and white pixels to black).
		if (self.whitebg):
			blur = cv2.bitwise_not(blur)

		# Then, any pixel with a value greater than zero (black) is set to 255 (white)
		#thresh[thresh > 0] = 255
		#retval, thresh_gray = cv2.threshold(blur, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
		_, threshold = cv2.threshold(blur, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco

		return threshold

	def remove_noise(self, img):
		"""
		"""
		bordersize = 100
		bordered = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )

		row, col = bordered.shape
		mean = 0
		gauss = np.random.normal(mean,1,(row,col))
		gauss = gauss.reshape(row,col)
		noisy = bordered + gauss
		noisy = (noisy).astype('uint8')

		for i in range(3):
			noisy = cv2.fastNlMeansDenoising(noisy, templateWindowSize=5, searchWindowSize=25, h=65)

		noisy = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

		return noisy

	def crop(self, threshold_of_img):
		"""
		Find where the signature is and make a cropped region

		Parameters
		----------
		threshold_of_img: Only black and white pixels
		----------
		Return
			Only part of the image that contains drawings
		"""
		y, x = threshold_of_img.shape

		# find where the black pixels are
		points = np.argwhere(threshold_of_img == 0) 
		# store them in x,y coordinates instead of row, col indices
		points = np.fliplr(points)
		
		# create a rectangle around those points
		x, y, w, h = cv2.boundingRect(points)
		
		del points
		
		# make the box a little bigger
		x, y, w, h = x-10, y-10, w+20, h+20
		
		if x < 0: x = 0
		if y < 0: y = 0

		return threshold_of_img[y:y+h, x:x+w]

	def scale(self, img):
		"""
		"""
		new = imutils.resize(img, height=self.size)

		if new.shape[1] > self.size:
			new = imutils.resize(new, width=self.size)

		border_size_x = (self.size - new.shape[1])//2
		border_size_y = (self.size - new.shape[0])//2

		new = cv2.copyMakeBorder(
			new, 
			top=border_size_y + self.size, 
			bottom=border_size_y + self.size, 
			left=border_size_x + self.size, 
			right=border_size_x + self.size,
			borderType=cv2.BORDER_CONSTANT,
			value=[255,255,255]
			#cv2.BORDER_REPLICATE
		)

		return new

	def draw(self, output=None):
		"""
		Draw the image by its shapes.

		Parameters
		----------
		output: Path where the image should be drawn.
		"""

	def view(self, original, normalized):
		cv2.imshow('Original', original)
		cv2.imshow('Normalized', normalized)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def debugging(original, threshold, outline, normilazed):
		#debug1 = np.hstack((grayscale, threshold, outline))
		cv2.imshow('original', original)
		cv2.imshow('threshold', threshold)
		cv2.imshow('outline', outline)
		cv2.imshow('normilazed', normilazed)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def getTheBestContour(self, imgPath):
		"""
		Primary method
		"""
		# Load the image
		original = cv2.imread(imgPath)
		# Signature
		threshold = self.get_threshold(original)

		# Accessing Image Properties
		# Image properties include number of rows, columns and channels, 
		# type of image data, number of pixels etc.
		# Shape of image is accessed by img.shape. It returns a tuple of number of rows, 
		# columns and channels (if image is color):
		outline = np.zeros(threshold.shape, dtype = "uint8")

		# Initialize the outline image,
		# find the outermost contours (the outline) of the object, 
		# cv2.RETR_EXTERNAL - telling OpenCV to find only the outermost contours.
		# cv2.CHAIN_APPROX_SIMPLE - to compress and approximate the contours to save memory
		#img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		(contours, hierarchy) = cv2.findContours(threshold.copy(),
			cv2.RETR_EXTERNAL, 
			cv2.CHAIN_APPROX_SIMPLE)

		# Sort the contours based on their area, in descending order. 
		# keep only the largest contour and discard the others.
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]

		# The outline is drawn as a filled in mask with white pixels:
		for cnt in contours:

			if(cv2.contourArea(cnt) > 0):

				# cv2.arcLength and cv2.approxPolyDP. 
				# These methods are used to approximate the polygonal curves of a contour.
				#peri = cv2.arcLength(cnt, True)

				# Level of approximation precision. 
				# In this case, we use 2% of the perimeter of the contour.
				# * The Ramer–Douglas–Peucker algorithm, also known as the Douglas–Peucker algorithm and iterative end-point fit algorithm, 
				# is an algorithm that decimates a curve composed of line segments to a similar curve with fewer point
				#approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

				cv2.drawContours(outline, [cnt], -1, 255, -1)

		if (outline.any()):
			outline = cv2.bitwise_not(outline)

		# get only important shape from image
		normalized = self.crop(outline)
		# resize and add border
		normalized = self.scale(normalized)

		# Debugging: show steps of processing
		#if imgPath.find(self.imageForDebugging) >= 0:
			#self.debugging(original, threshold, outline, normalized)

		#cv2.imwrite("{}\\{}.png".format(imageFolderThreshold, imageName), outline)

		return normalized