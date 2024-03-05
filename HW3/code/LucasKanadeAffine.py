import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    iters = 0
    It_spline = SetupBivariateSplineInterpolator(It)
    It1_spline = SetupBivariateSplineInterpolator(It1)
    delta_p = np.ones(6).flatten()*np.inf
    
    while iters < num_iters and np.square(delta_p).sum() > threshold:

        u1 = np.array([0, 0, 1]).reshape(-1, 1)
        u2 = np.array([It.shape[1], It.shape[0], 1]).reshape(-1, 1)
        
        x = np.linspace(u1[0], u2[0], It.shape[1])
        y = np.linspace(u1[1], u2[1], It.shape[0])
        X, Y = np.meshgrid(x, y)

        u1_w = np.vstack((M, np.array([0.0, 0.0, 1.0])))@u1
        u2_w = np.vstack((M, np.array([0.0, 0.0, 1.0])))@u2

        x_w = np.linspace(u1_w[0], u2_w[0], It.shape[1])
        y_w = np.linspace(u1_w[1], u2_w[1], It.shape[0])
        X_w, Y_w = np.meshgrid(x_w, y_w)
        
        mask = (X_w >= 0) & (X_w < It.shape[1]) & (Y_w >= 0) & (Y < It.shape[0])

        X = X[mask]
        Y = Y[mask]
        X_w = X_w[mask]
        Y_w = Y_w[mask]

        error_img = It_spline.ev(Y, X) - It1_spline.ev(Y_w, X_w)

        # mask = affine_transform(np.ones(It1.shape), M)
        # It1_warped = affine_transform(It1, M)

        # error_img = np.multiply(It, mask) - It1_warped
        gradient = CalculateWarpedGradient(It1_spline, It1.shape, M).T # n by 2

        A = np.zeros((It.shape[0]*It.shape[1], 6))
        for y in np.arange(It1.shape[0]):
            for x in np.arange(It1.shape[1]):
                pixel_idx = y*It.shape[1]+x
                jacob = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
                A[pixel_idx, :] = np.matmul(gradient[pixel_idx, :], jacob)

        A_ = A[mask.flatten(), :]
       
        delta_p = np.linalg.pinv(np.matmul(A_.T, A_)) @ A_.T @ error_img.reshape(-1, 1) # 6 by 1
        M = M + delta_p.reshape(3, 2).T
        iters += 1
    
    return M

def CalculateWarpedGradient(It1_spline, It1_shape, M):
    """
    Calculates the warped gradient
    It1_spline: The interpolator for the I_(t+1) image
    rect: The location of interest
    p: the warp matrix

    return:
    gradient = np array of gradient in x direction and gradient in y direction
    """

    # NEED TO WARP X1 and Y1 using M BEFORE calculating SPLINE
    M = np.vstack((M, np.array([0.0, 0.0, 1.0])))

    top_left = np.array([0,0,1]).reshape(-1,1)
    bot_right = np.array([It1_shape[1], It1_shape[0], 1]).reshape(-1,1)
    warped_top_left = np.matmul(M, top_left)
    warped_top_left = (warped_top_left/warped_top_left[2])[:2]
    warped_bot_right = np.matmul(M, bot_right)
    warped_bot_right = (warped_bot_right/warped_bot_right[2])[:2]

    x1 = np.linspace(warped_top_left[0], warped_bot_right[0], It1_shape[1])
    y1 = np.linspace(warped_top_left[1], warped_bot_right[1], It1_shape[0])

    X1, Y1 = np.meshgrid(x1, y1)

    gradient_x = It1_spline.ev(Y1, X1, dx = 0, dy = 1).flatten()
    gradient_y = It1_spline.ev(Y1, X1, dx = 1, dy = 0).flatten()
    return np.vstack((gradient_x, gradient_y))

def SetupBivariateSplineInterpolator(image):
    interpolator = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)
    return interpolator


# seq = np.load("../data/girlseq.npy")
# rect = [280, 152, 330, 318]
# width, height = GetRectShape(rect)

# It = seq[:, :, 0]
# It1 = seq[:, :, 1]
# threshold = 0.001
# num_iters = 1000
# M_ = LucasKanadeAffine(It, It1, threshold, num_iters)
# print(M_)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1,1)
# ax.imshow(It, cmap = 'gray')
# fig.savefig('../results/test_og.png')

# fig, ax = plt.subplots(1,1)
# ax.imshow(affine_transform(It, M_), cmap = 'gray')
# fig.savefig('../results/test_warped.png')

# fig, ax = plt.subplots(1,1)
# ax.imshow(It1, cmap = 'gray')
# fig.savefig('../results/target.png')

# n = 10
# test = np.zeros((n,n))
# test[2:8, 2:8] = np.arange(1, 37, 1).reshape(6, 6)
# # test = np.arange(n*n).reshape(n,n)
# print(test)
# deg = 45 * np.pi/180
# shift = [0,0]
# M = np.array([[np.cos(deg), -np.sin(deg), shift[0]],
#                [np.sin(deg), np.cos(deg), shift[1]],
#                  [0.0, 0.0, 1.0]])

# M_inv = np.linalg.pinv(M)
# # print(M_inv)
# M_inv = M_inv[:2, :]
# M = M[:2, :]
# print(np.round(affine_transform(test, M)))