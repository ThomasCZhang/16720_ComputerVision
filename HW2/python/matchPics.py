import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

	#Convert Images to GrayScale
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	I1_feature_locs = corner_detection(I1, sigma)
	I2_feature_locs = corner_detection(I2, sigma)

	#Obtain descriptors for the computed feature locations
	I1_descriptors, locs1 = computeBrief(I1, I1_feature_locs)
	I2_descriptors, locs2 = computeBrief(I2, I2_feature_locs)

	#Match features using the descriptors
	matches = briefMatch(I1_descriptors, I2_descriptors, ratio)
	
	locs1 = locs1[:, [1, 0]]
	locs2 = locs2[:, [1, 0]]
	return matches, locs1, locs2
