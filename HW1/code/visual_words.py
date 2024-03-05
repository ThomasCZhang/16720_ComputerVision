import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import scipy.ndimage
import skimage.color


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    n_scales = len(filter_scales)

    # ----- TODO -----
    img = skimage.color.rgb2lab(img)
    output_images = np.zeros((img.shape[0], img.shape[1], img.shape[2]*4*n_scales))
    
    for i in range(n_scales):
        scale = filter_scales[i]
        for j in range(3):
            output_images[:, :, 12*i+j] = scipy.ndimage.gaussian_filter(img[:, :, j], sigma=scale)
            output_images[:, :, 12*i+j+3] = scipy.ndimage.gaussian_laplace(img[:, :, j], sigma=scale)
            output_images[:, :, 12*i+j+6] = scipy.ndimage.gaussian_filter1d(img[:, :, j], axis = 0, sigma=scale, order = 1)
            output_images[:, :, 12*i+j+9] = scipy.ndimage.gaussian_filter1d(img[:, :, j], axis = 1, sigma=scale, order = 1)

    return output_images

def compute_dictionary_one_image(opts, file_path):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----

    alpha = opts.alpha
    out_dir = opts.out_dir
    data_dir = opts.data_dir

    img = Image.open(join(data_dir, file_path))
    img = np.array(img).astype(np.float32)/255
    output_images = extract_filter_responses(opts, img)
 
    x_cords = np.random.randint(low = 0, high = img.shape[0], size = alpha)
    y_cords = np.random.randint(low = 0, high = img.shape[1], size = alpha)

    with open(join(out_dir, 'features.npy'), 'ab') as f:
        np.save(f, output_images[x_cords, y_cords, :])

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # args = [(opts, file) for file in train_files]
    # with multiprocessing.Pool(n_worker) as pool:
    #     pool.starmap(compute_dictionary_one_image, args)
    # ----- TODO -----
    for file in train_files:
        compute_dictionary_one_image(opts, file)

    with open(join(out_dir, 'features.npy'), 'rb') as f:
        feature_matrix = np.load(f)
        try:
            while True:
                feature_matrix = np.vstack((feature_matrix, np.load(f)))
        except:
            print("EoF")

    dictionary = KMeans(n_clusters = K, n_init=10).fit(feature_matrix)
    np.save(join(out_dir, 'dictionary.npy'), dictionary.cluster_centers_)

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    response_words = extract_filter_responses(opts,img)
    response_words = np.reshape(response_words, (-1,response_words.shape[2])) # Reshape the words into 2d array so we can use cdist.
    wordmap = np.argmin(scipy.spatial.distance.cdist(response_words,dictionary), axis = 1)
    wordmap = np.reshape(wordmap,(img.shape[0],img.shape[1]))
    return wordmap

    
    # ----- TODO -----
    pass

