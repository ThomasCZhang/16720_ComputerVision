import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    iters = 0

    delta_p = np.ones(6).flatten()*np.inf

    grad_y, grad_x = np.gradient(It)
    gradient = np.vstack((grad_x.flatten(), grad_y.flatten())).T # n by 2

    # It_spline = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    A = np.zeros((It.shape[0]*It.shape[1], 6))
    for y in np.arange(It.shape[0]):
        for x in np.arange(It.shape[1]):
            pixel_idx = y*It.shape[1]+x
            jacob = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
            A[pixel_idx, :] = np.matmul(gradient[pixel_idx, :], jacob)
    
    while iters < num_iters  and np.square(delta_p).sum() > threshold:
        
        # print(np.square(delta_p).sum())
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
        
        mask = (X_w >= 0) & (X_w < It.shape[1]) & (Y_w >= 0) & (Y_w < It.shape[0])

        X = X[mask]
        Y = Y[mask]
        X_w = X_w[mask]
        Y_w = Y_w[mask]
        A_ = A[mask.flatten(), :]

        error_img = It[mask].flatten() - It1_spline.ev(Y_w, X_w)
        error_img = error_img.flatten()

        # mask = affine_transform(np.ones(It1.shape), M)
        # It1_warped = affine_transform(It1, M)
        # error_img = np.multiply(It-It1_warped, mask)

        delta_p = -np.linalg.pinv(A_.T @ A_) @ A_.T @ error_img.reshape(-1, 1) # 6 by 1
        delta_M = np.vstack((delta_p.reshape(3,2).T, [0.0, 0.0, 1.0]))
        delta_M = delta_M + np.array([[1, 0 , 0], [0, 1, 0], [0, 0, 0]])

        M = np.vstack((M, np.array([0.0, 0.0, 1.0])))
        M = M @ np.linalg.pinv(delta_M)
        
        M = M[:2]
        # print(f'Iteration {iters}, delta_p magnitude: {np.square(delta_p).sum()}')
        iters += 1
    
    return M


