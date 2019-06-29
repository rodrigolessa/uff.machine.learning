from extracting_pre_processing import ImagePreProcessing
from PIL import Image
import numpy as np
import cv2
import glob
import imutils
import os
import sys

imageFolder = "D:\\caltech256"
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

try:
    os.makedirs(imageFolderConverted)
except OSError as e:
	pass

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

imagesInFolder = glob.glob(imageFinder)

qt = len(imagesInFolder)

i = 1

for spritePath in imagesInFolder:
    # Extract image name, this will serve as unqiue key into the index dictionary.
    imageName = spritePath[spritePath.rfind('\\') + 1:].lower().replace(imageExtension, '')

    #print('images name: {}'.format(imageName))

    progress(i, qt)

    img = Image.open(spritePath)
    wpercent = (imageBasewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((imageBasewidth, hsize), PIL.Image.ANTIALIAS)
    img.save("{}\\{}.png".format(imageFolderConverted, imageName))

    i+=1