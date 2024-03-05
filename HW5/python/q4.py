import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# import matplotlib.patches
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    # sigma = skimage.restoration.estimate_sigma(image, channel_axis=2)
    ##########################
    ##### your code here #####
    import matplotlib.pyplot as plt
    im = image.copy()
    im = skimage.color.rgb2gray(im)
    im = im/np.max(im)
    im = skimage.restoration.denoise_bilateral(im, sigma_spatial = 1)
    thresh = skimage.filters.threshold_isodata(im)
    bw = (im > thresh).astype(float) # Letters = 0, background = 1
    im = skimage.filters.gaussian(bw, sigma = 2)
    thresh = skimage.filters.threshold_isodata(im)
    bw = (im > thresh).astype(float) # Letters = 0, background = 1
    # im = skimage.filters.gaussian(bw, sigma = (5,3))
    im = skimage.morphology.binary_erosion(bw, ((np.ones((1, 5)), 1), (np.ones((3, 1)), 1)))
    thresh = skimage.filters.threshold_isodata(im)
    bw2 = (im > thresh).astype(float) # Letters = 0, background = 1
    # plt.imshow(bw2, cmap = 'gray')
    # plt.show()
    # raise Exception()

    num_erosions = 1 + int((im.shape[0] * im.shape[1])//1.5e6) 
    if im.shape[1] == 4032:
        num_erosions += 1 # image 4 needs more erosion
    # print(im.shape[0] * im.shape[1], im.shape[1], num_erode)

    footprint = np.zeros((3,3))
    footprint[footprint.shape[0]//2, :] = 1
    footprint[:, footprint.shape[1]//2] = 1
    for _ in range(num_erosions):
        bw = skimage.morphology.binary_erosion(bw, footprint)
    # bw = skimage.morphology.binary_erosion(bw, ((np.ones((1, footprint_width-3)), 1), (np.ones((footprint_width, 1)), 1)))
    for _ in range(3):
        bw = skimage.morphology.binary_closing(bw)
    bw = skimage.img_as_float(bw)
    # plt.imshow(bw)
    # plt.show()
    # labels, num = skimage.measure.label(bw, return_num = True, background = 1)
    labels, num = skimage.measure.label(bw2, return_num = True, background = 1)

    mean_area = 0
    for i in range(num):
        idxs = np.nonzero(labels == i+1)
        minr, minc, maxr, maxc = [min(idxs[0]), min(idxs[1]), max(idxs[0]), max(idxs[1])]
        if minr > 0:
            minr -= 1
        if minc > 0:
            minc -= 1
        if maxr < labels.shape[0]-1:
            maxr += 1
        if maxc < labels.shape[1]-1:
            maxc += 1
        bboxes.append([minr, minc, maxr, maxc])
        mean_area += (maxr-minr)*(maxc-minc)
    mean_area /= num

    bboxes = [bbox for bbox in bboxes if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > 0.07*mean_area]
    remove_idxs = []
    for i in range(len(bboxes)):
        i_minr, i_minc, i_maxr, i_maxc = bboxes[i]
        for j in range(len(bboxes)):
            if j == i:
                continue
            j_minr, j_minc, j_maxr, j_maxc = bboxes[j]
            j_centr = 0.5*(j_minr + j_maxr)
            j_centc = 0.5*(j_minc + j_maxc)
            leeway = 20
            y_condition = (j_centr > i_minr-leeway) and (j_centr < i_maxr+leeway)
            x_condition = (j_centc > i_minc) and (j_centc < i_maxc)
            if x_condition and y_condition:
                minr, minc, maxr, maxc = [min(i_minr, j_minr), min(i_minc, j_minc), max(i_maxr, j_maxr), max(i_maxc, j_maxc)]
                bboxes[min(i,j)] = [minr, minc, maxr, maxc]
                remove_idxs.append(max(i,j))

    # print(remove_idxs)
    for i in sorted(np.unique(remove_idxs), reverse = True):
        del bboxes[i]

    # fig, ax = plt.subplots(1)
    # ax.imshow(bw, cmap = 'gray')
    # fig, ax = plt.subplots(1)
    # ax.imshow(labels, cmap = 'jet')
    # plt.show()
    ##########################
    return bboxes, bw