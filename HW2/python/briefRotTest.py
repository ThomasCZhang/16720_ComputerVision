import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import matplotlib.pyplot as plt

opts = get_opts()
from scipy import ndimage
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')

histogram = [0 for _ in range(36)]
for i in range(36):
	print(i)
	#Rotate Image
	rot_cover = ndimage.rotate(cv_cover, 10*i)
	#Compute features, descriptors and Match features
	matches, loc1, loc2 = matchPics(cv_cover, rot_cover, opts)
	#Update histogram
	histogram[i] = len(matches)

#Display histogram
plt.hist([10*i for i in range(36)], bins = [10*i for i in range(37)], weights=histogram)
plt.xticks([10*i for i in range(36)], rotation = 90)
plt.xlabel('Degree of rotation')
plt.ylabel('Matches')
plt.show()
