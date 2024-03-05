import numpy as np
import copy
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = copy.deepcopy(p0)
    delta_p = np.array([[5], [5]])

    It_spline = SetupBivariateSplineInterpolator(It)
    It1_spline = SetupBivariateSplineInterpolator(It1)

    width, height = GetRectShape(rect)
    x = np.linspace(rect[0], rect[2], width + 1)
    y = np.linspace(rect[1], rect[3], height + 1)
    X, Y = np.meshgrid(x, y)
    template = It_spline.ev(Y, X)

    num_iter = 0
    # delta_error = np.inf
    while num_iter < num_iters and np.square(delta_p).sum() > threshold:
        # print(num_iter)
        # Update P
        
        x1 = np.linspace(rect[0] + p[0], rect[2] + p[0], width + 1)
        y1 = np.linspace(rect[1] + p[1], rect[3] + p[1], height + 1)
        X1, Y1 = np.meshgrid(x1, y1)
        error_img = template - It1_spline.ev(Y1, X1)

        grad_x = It1_spline.ev(Y1, X1, dx = 0, dy = 1)
        grad_y = It1_spline.ev(Y1, X1, dx = 1, dy = 0)
        grad = np.vstack((grad_x.flatten(), grad_y.flatten())).T
        jacob = np.eye(2) # 2 by 2
        A = grad @ jacob
        delta_p = np.linalg.pinv(A.T @ A) @ A.T @ error_img.flatten()
        p[0] += delta_p[0]
        p[1] += delta_p[1]
        num_iter += 1
    
    return p

def GetRectShape(rect):
    """
    Calculates and returns the height and then the width of a rectangle. The rectangle is represented by
    [x0, y0, x1, y1] where (x0, y0) is the coordinates of the top left corner and (x1, y1) is the coordinates of the
    lower right corner of the rectangle.

    Return: (x1 - x0, y1 - y0) Pair of integers. First integer is the width of the image, second is the height.
    """
    return (int(rect[2]-rect[0]) , int(rect[3]-rect[1]))

def SetupBivariateSplineInterpolator(image):
    interpolator = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)
    return interpolator