import numpy as np
from LucasKanadeAffine import *
from InverseCompositionAffine import *
from scipy.ndimage import binary_erosion, binary_dilation

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    M_inv = np.linalg.pinv(np.vstack((M, np.array([0.0, 0.0, 1.0]))))
    M_inv = M_inv[:2,:]
    image1_warped = affine_transform(image1, M_inv)
    temp_mask = affine_transform(np.ones(image1.shape), M_inv)
    delta = np.abs(np.multiply(temp_mask, image2)-image1_warped)
    mask = np.where(delta >= tolerance, 1, 0)
    # mask = binary_erosion(mask)
    # mask = binary_dilation(mask)
    # mask = np.ones(image1.shape, dtype=bool)

    return mask
