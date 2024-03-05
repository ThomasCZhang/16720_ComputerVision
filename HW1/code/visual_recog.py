import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist = np.histogram(wordmap, bins = np.arange(0, K+1), density=True) # output is (count, bins)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----

    nrows, ncols = wordmap.shape
    hist_all = np.zeros(K*(4**(L+1)-1)//3)
    counter = 0
    for l in range(L+1): # L+1 layers
        # Split wordmap into 2^l by 2^l subsections
        # e.g. if l = 1 wordmap is split into 4 sections (2 by 2).
        row_indices = [*[round((i)*(nrows/(2**l))) for i in range(2**l)], nrows]    # indicies of the break points
        col_indices = [*[round((i)*(ncols/(2**l))) for i in range(2**l)], ncols]
        if l == 0:
            weight = 2**(-L)
        else:
            weight = 2**(l-L-1)

        for i in range(2**l):
            for j in range(2**l):
                sub_wordmap = wordmap[row_indices[i]:row_indices[i+1], col_indices[j]:col_indices[j+1]]
                hist = get_feature_from_wordmap(opts, sub_wordmap)
                hist_all[counter*K:(counter+1)*K] = hist[0]*weight
                counter += 1
    return (hist_all)

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    
    # ----- TODO -----
    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32)/255
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    return feature
   


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # print(len(train_labels), len(train_files))
    # ----- TODO -----
    N = len(train_files)
    K = opts.K
    features = np.zeros((N, K*(4**(SPM_layer_num+1)-1)//3))
    for i in range(N):
        features[i, :] = get_image_feature(opts, train_files[i], dictionary)

    print("Finished Making Feature Matrix. Saving Model.")
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    hist_dist = 1 - np.sum(np.minimum(histograms, word_hist), axis = 1)
    return hist_dist    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    

    conf = np.zeros((8, 8))
    for i, file in enumerate(test_files):
        print(f'Index: %d File: %s'% (i, join(data_dir, file)), end = "\n")
        word_hist = get_image_feature(opts, join(data_dir, file), dictionary)
        min_dist_index = np.argmin(distance_to_set(word_hist, trained_system['features']))
        pred_label = trained_system['labels'][min_dist_index]
        true_label = test_labels[i]
        conf[pred_label,true_label] += 1
        if pred_label == 3 and true_label == 4:
            break

    accuracy = np.sum(np.diag(conf))/len(test_labels)
    return (conf, accuracy)
