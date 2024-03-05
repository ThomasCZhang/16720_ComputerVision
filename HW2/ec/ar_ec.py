import numpy as np
import cv2

#Import necessary functions

from loadVid import *
from planarH import *
# from matchPics import *
import matplotlib.pyplot as plt
from opts import get_opts

def HorizontalCrop(img, dimensions):
    deltaX = int(0.5*(img.shape[1] - dimensions[1])) # Crop off left and right evenly
    return img[:, deltaX:deltaX+dimensions[1]]

def VerticalCrop(img, dimensions):
    deltaY = int(0.5*(img.shape[0] - dimensions[0])) # Crop off top and bottom evenly
    return img[deltaY:deltaY+dimensions[0], :]

def CropFrame(img, dimensions):
    img = HorizontalCrop(img, dimensions)
    img = VerticalCrop(img, dimensions)
    return img

def MakeVideo(images, save_path, fps):    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    assert(len(images.shape) == 4)
    frameSize = (images.shape[2], images.shape[1])
    video = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
    for img in images:
        video.write(np.uint8(img))
    video.release()

def FindMatches(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

#Write script for Q3.1

opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
ar_source = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')

print('AR_source: ', ar_source.shape, '\nbook: ', book.shape, '\ncv_book: ', cv_cover.shape)
vid_length = min(ar_source.shape[0], book.shape[0]) # Take the shorter video length
# vid_length = 3 # For troubleshooting
black_bar_height = 45
scale = (ar_source.shape[1]-2*black_bar_height)/cv_cover.shape[0]
crop_dimensions = (ar_source.shape[1]-2*black_bar_height, int(cv_cover.shape[1]*scale))

final_frames = book.copy()[:vid_length, :, :, :]
for i in range(vid_length):
    dest_img = final_frames[i, :, :, :]
    ar_img = ar_source[i, :, :, :]
    ar_img = CropFrame(ar_img, crop_dimensions)
    matches, locs1, locs2 = FindMatches(dest_img, cv_cover)

    x1 = locs1[matches[:, 0], :]
    x2 = locs2[matches[:, 1], :]
    try:
        H, inliers = computeH_ransac(x1, x2, opts)
        template = cv2.resize(ar_img, (cv_cover.shape[1], cv_cover.shape[0]))

        compositeFrame = compositeH(H, template, dest_img)
        dest_img = compositeFrame
    except:
        continue
    print(f'Frame {i}: \tInliers: {np.sum(inliers)} \tVideo Size: {final_frames.shape}')



save_path = '../data/ModifiedAR.mp4'
MakeVideo(final_frames, save_path, fps=30)
