import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

i = 0
for img in os.listdir('../images'):
    # if i!= 3:
    #     i+= 1
    #     continue
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.imshow(bw, cmap = 'gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    nrows, ncols = bw.shape
    group_y_centers = {}
    groups = {}
    new_key = 0
    for i, bbox in enumerate(bboxes):
        grouped = False
        minr, minc, maxr, maxc = bbox
        y_center = 0.5*(maxr + minr)
        for key in groups:
            if y_center > np.mean(group_y_centers[key]) - 0.075*nrows and y_center < np.mean(group_y_centers[key]) + 0.075*nrows:
                groups[key].append(i)
                group_y_centers[key].append(y_center)
                grouped = True
                break
        if not grouped:
            group_y_centers[new_key] = [y_center]
            groups[new_key] = [i]
            new_key += 1
    
    # Order the bboxes in a row
    for key in groups:
        group_ids = groups[key]
        row = [(id, bboxes[id][1]) for id in group_ids]
        row = sorted(row, key = lambda x: x[1])
        row = [x[0] for x in row]
        groups[key] = row

    # for key in groups:
    #     group_ids = groups[key]
    #     plt.imshow(bw, cmap = 'gray')
    #     for id in group_ids:
    #         bbox = bboxes[id]
    #         minr, minc, maxr, maxc = bbox
    #         rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                 fill=False, edgecolor='red', linewidth=2)
    #         plt.gca().add_patch(rect)
    #     plt.show()
    ##### your code here #####
    ##########################


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    # max_height = 0
    # max_width = 0
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     if maxr-minr > max_height:
    #         max_height = maxr-minr
    #     if maxc-minc > max_width:
    #         max_width = maxc-minc
    # if max_height > max_width:
    #     # box_width = int(np.ceil(max_height/32))*32
    #     box_width = max_height
    # else:
    #     # box_width = int(np.ceil(max_width/32))*32
    #     box_width = max_width
    data = []
    for key in groups:
        group_ids = groups[key]
        for i, id in enumerate(group_ids):
            bbox = bboxes[id]
            minr, minc, maxr, maxc = bbox
            char_img = bw[minr:maxr, minc:maxc]
            box_width = max(char_img.shape)+int(0.45*max(char_img.shape))
            delta_y = box_width - char_img.shape[0]
            delta_x = box_width - char_img.shape[1]
            pad_y = (delta_y//2, delta_y - delta_y//2)
            pad_x = (delta_x//2, delta_x - delta_x//2)
            # print(f'pad_y: {pad_y},pad x: {pad_x}')
            char_img = np.pad(char_img, (pad_y, pad_x), constant_values=1)

            # output_size = 32
            # bin_size = box_width // output_size
            # small_image = char_img.reshape((output_size, bin_size, 
            #                                 output_size, bin_size)).mean(3).max(1)
            small_image = skimage.transform.rescale(char_img, 32/box_width)
            # small_image = skimage.filters.gaussian(small_image, sigma = (1, 0))
            # thresh = skimage.filters.threshold_isodata(small_image)
            # small_image = (small_image > thresh).astype(float)
            # small_image = skimage.filters.gaussian(small_image, sigma = 2)
            data.append(small_image.T.flatten())
            skimage.io.imsave(f"../crop_imgs/{img.split('.')[0]}_{i}.png", (255*small_image).astype(np.uint8))
            # fig, axs = plt.subplots(2)
            # axs[0].imshow(char_img, cmap = 'gray')
            # axs[1].imshow(small_image, cmap = 'gray')
            # plt.show()
    data = np.array(data)
    # print(data.shape)
    #########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    h1 = forward(data, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    preds = np.argmax(probs, axis = 1)
    im_letters = ""
    for i in preds:
        im_letters += letters[i]
    print(im_letters)
    ##########################
    