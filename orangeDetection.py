# This program will be able to detect the biggest orange on a given image
# Will use OpenCV to detect the image

# Scale image and convert to RGB
# Add mask and overlay mask
# Find all contours, circle the largest one

from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

def showImage(image):
	plt.figure(figsize =(15,15))
	plt.imshow(image, interpolation='nearest')

def overlayMasks(mask, image):
	# Apply mask to the image

	# Create an RGB Mask from grayscale mask
	rgbMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

	# Our weights will be our image + mask which will have to be both in rgb to obtain our new image

	newImage = cv2.addWeighted(rgbMask, 0.5, image, 0.5, 0)

	return newImage

def findLargestContour(image):
	 # Create a copy of our image

	 image = image.copy()

	 # OpenCV can find contours in our image for us, get us a list, then we can find the max of a list using Python

	 # OpenCV's findContours will give us our list

	 contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	 # contourSizeList gives us the findings of OpenCV

	 contourSizeList = [(cv2.contourArea(contour), contour) for contour in contours]

	 # Largest contour will be in that list in the first position at the top of the array

	 largestContour = max(contourSizeList, key=lambda x:x[0])[1]

	 # Draw ellipse around the biggest contour

	 mask = np.zeros(image.shape, np.uint8)
	 cv2.drawContours(mask, [largestContour], -1, 255, -1)

	 return largestContour, mask

def circleContours(image, contour):
	# Bound the ellipse on the image and add it on the image

	ellipsedImage = image.copy()
	ellipse = cv2.fitEllipse(contour)

	cv2.ellipse(ellipsedImage, ellipse, (0, 255, 0), 2, cv2,CV_AA)

	return ellipsedImage

def findOrange(image):
	# Convert image to RBG scale, we are going to use RGB for our img

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Take our image and resize it to a square

	imageDimension = max(image.shape)
	scale = 700/imageDimension
	image = cv2.resize(image, None, fx=scale, fy=scale)

	# We have to smooth our image, remove all noises

	imageBlur = cv2.GaussianBlur(image, (7,7), 0)

	# We will have to filter by color and intensity
	# HSV is needed for that scale

	imageHSV = cv2.cvtColor(imageBlur, cv2.COLOR_RGB2HSV)

	# Filter for Color & Intesity using a mask

	colorMin = np.array([28, 93 ,94])
	colorMax = np.array([48, 76, 98])

	brightMin = np.array([198, 93 ,94])
	brighMax = np.array([218, 76, 98])

	imageMask = cv2.inRange(imageHSV, colorMin, colorMax) + cv2.inRange(imageHSV, brightMin, brighMax)

	# We have to seperate the image from all other things in the image: Remove all noise

	# we will locate the shape using an ellipse

	# Use morphology to refine the image in both directions : Smoother picture all around

	structuredElement = cv2.getStructuringElement(cv2,MORPH_ELLIPSE, (20, 20))
	morphClose = cv2.morphologyEx(imageMask, cv2.MORPH_CLOSE, structuredElement)
	morphOpen = cv2.morphologyEx(imageMask, cv2.MORPH_OPEN, structuredElement)

	# Locate the biggest contour of the image

	largestContour, mask = findLargestContour(morphOpen)

	# We have to overlay the masks put on the image

	imageOverlay = overlayMasks(morphOpen, image)

	# Locate the largest contour and draw an ellipse around

	locatedContour = locateContour(imageOverlay, largestContour)

	showImage(locatedContour)

	# We will have to convert the image back to its original color scheme

	imageBGR = cv2.cvtColor(locatedContour, cv2.COLOR_RGB2BGR)

	return imageBGR

image = cv2.imread('orange.jpg')
resultImage = findOrange(image)
cv2.imwrite('newOrange.jpg', resultImage)


