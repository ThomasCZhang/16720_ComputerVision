import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import *
from planarH import *
#Import necessary functions


#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, loc1 , loc2 = matchPics(cv_desk, cv_cover, opts)

print('Matches Found:', len(loc1[matches[:, 0], :]), len(loc2[matches[:, 1], :]))

x1 = loc1[matches[:, 0], :]
x2 = loc2[matches[:, 1], :]
H, inliers = computeH_ransac(x1, x2, opts)
print(inliers)

hp_warped = cv2.warpPerspective(hp_cover, H, (cv_desk.shape[1],cv_desk.shape[0]))

template = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

hp_warped_2 = compositeH(H, template, cv_desk)

save_name = f'hp_warped_iters{opts.max_iters}_tol{opts.inlier_tol}.png'
cv2.imwrite(save_name, hp_warped_2)

###########

# x1 = np.array([[1, 2], [2, 1], [4, 3], [5, 1]])
# x2 = 1+x1
# print('x1\n', x1)
# print('x2\n', x2)
# H = computeH_norm(x2, x1)
# H2 = computeH(x2, x1)
# H = 0.5*np.eye(3)
# H[2,2] = 1
# print(H)
# H = np.eye(3)*2
# x2_homo = np.vstack((x2.T, np.ones(x2.shape[0])))
# test1 = H@x2_homo
# test1 = test1[:2,:]/test1[2, :]
# print('test1\n', test1)

# test2 = H2@x2_homo
# test2 = test2[:2,:]/test2[2, :]
# print('test2\n', test1)
