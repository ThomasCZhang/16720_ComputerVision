"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import matplotlib.pyplot as plt
import util
import helper
import cv2

from scipy.optimize import leastsq

gen = np.random.default_rng(2023)

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # Convert everything to float so we can do floating point division.
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)
    M = float(M)
    N = len(pts1)
    T = np.diag([1/M, 1/M, 1]) # Scaling matrix to be used later

    # Normalize points
    pts1 /= M
    pts2 /= M

    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    # Make the matrix
    A = np.zeros((N, 9)) # Pre-allocate matrix to make things easier.
    A[:, 0] = x1 * x2
    A[:, 1] = y1 * x2
    A[:, 2] = x2
    A[:, 3] = x1 * y2
    A[:, 4] = y1 * y2
    A[:, 5] = y2
    A[:, 6] = x1 
    A[:, 7] = y1 
    A[:, 8] = np.ones(N)

    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1, :].reshape(3,3)
    F = util._singularize(F)
    F = util.refineF(F, pts1, pts2)
    F = T.T @ F @ T
    np.savez('../results/q2_1.npz', F = F, M = M)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    xl, yl = pts1[:, 0], pts1[:, 1]
    xr, yr = pts2[:, 0], pts2[:, 1]
    N = len(pts1)

    # Transpose after making matrix so that we can vertically stack.
    A1 = np.array([C1[0, 0]-C1[2,0]*xl, C1[0, 1]-C1[2,1]*xl, C1[0, 2]-C1[2,2]*xl, C1[0, 3]-C1[2,3]*xl]).T
    A2 = np.array([C1[1, 0]-C1[2,0]*yl, C1[1, 1]-C1[2,1]*yl, C1[1, 2]-C1[2,2]*yl, C1[1, 3]-C1[2,3]*yl]).T
    A3 = np.array([C2[0, 0]-C2[2,0]*xr, C2[0, 1]-C2[2,1]*xr, C2[0, 2]-C2[2,2]*xr, C2[0, 3]-C2[2,3]*xr]).T
    A4 = np.array([C2[1, 0]-C2[2,0]*yr, C2[1, 1]-C2[2,1]*yr, C2[1, 2]-C2[2,2]*yr, C2[1, 3]-C2[2,3]*yr]).T
    
    # A = np.stack((A1,A2,A3,A4), axis = 1).reshape(-1, 4) # 4*N by 4 where N = number of points
    # U, S, Vh = np.linalg.svd(A)
    # print(Vh.shape)
    W = np.zeros((N, 3))
    for i in range(N):
        A = np.vstack((A1[i,:], A2[i,:], A3[i,:], A4[i,:]))
        # raise Exception(f'\n{A}\nC: ############\n{C1}\npts1: ##########\n{pts1[0]}')
        U, S, Vh = np.linalg.svd(A)
        W[i, :] = Vh[-1,:3]/Vh[-1,3]

    # need to calculate error as well.
    W_homogenous = np.hstack((W, np.ones((N,1))))
    err = 0
    for i in range(N):
        proj1 = C1@W_homogenous[i,:].T
        proj1 = proj1[:2]/proj1[2]
        proj2 = C2@W_homogenous[i,:].T
        proj2 = proj2[:2]/proj2[2]
        err += np.linalg.norm(pts1[i,:] - proj1)**2 + np.linalg.norm(pts2[i,:] - proj2)**2
    return W, err


def GetX2Bounds(x1, y1, l2, max_dist):
    '''
    Deterimines the bounds of x2 such that the distance between x2,y2
    and x1,y1 is less than max_dist.
    Input: x1, the x coordinate in img1
           y1, the y coordinate in img1
           l2, the epipolar line in image 2.
           max_dist, the max distance between x1,y1 and x2,y2    
    '''
    # y = mx + k 
    m = -l2[0]/l2[1]
    k = -l2[2]/l2[1]

    a = 1+m**2
    b = 2*(m*k - x1 - y1*m)
    c = x1**2 + (y1-k)**2 - max_dist**2

    min_x = x1 - 1
    max_x = x1 + 1
    if (b**2 - 4*a*c) >= 0:
        min_x = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        max_x = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
    return min_x, max_x

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    from scipy.interpolate import RectBivariateSpline
    # setup interpolator for im2.
    height, width, _, = im2.shape
    im2R_spline = RectBivariateSpline(np.arange(height), np.arange(width), im2[:, :, 0])
    im2G_spline = RectBivariateSpline(np.arange(height), np.arange(width), im2[:, :, 1])
    im2B_spline = RectBivariateSpline(np.arange(height), np.arange(width), im2[:, :, 2])

    patch_size = 5 # half the height and width of the patch in pixels.
    search_size = 20 # distance from x1, y1 that we will search in im2 (since images don't move much).

    p1 = np.array([x1, y1, 1]).T
    l2 = F@p1 # l2 is the epipolar line in im2
       
    # Possible x coords
    min_x2, max_x2 = GetX2Bounds(x1, y1, l2, search_size)

    X2s = np.linspace(min_x2, max_x2, search_size)
    Y2s = -(l2[0]*X2s + l2[2])/l2[1]    # ax + by + c = 0 --> y = -(a x+ c)/b
    inbounds_mask = (X2s > 0) & (X2s < width) & (Y2s > 0) & (Y2s < height)
    X2s = X2s[inbounds_mask]
    Y2s = Y2s[inbounds_mask]

    # Handling out of bounds stuff
    patch_x = np.array([-patch_size, patch_size+1])+x1
    if patch_x[0] < 0:
        patch_x[0] = 0
    if patch_x[1] > width:
        patch_x[1] = width
    patch_y = np.array([-patch_size, patch_size+1])+y1 
    if patch_y[0] < 0:
        patch_y[0] = 0
    if patch_y[1] > height:
        patch_y[1] = height
    patch_width = patch_x[1]-patch_x[0]
    patch_height = patch_y[1]-patch_y[0]

    weight_mask_x = np.arange(patch_x[0]-x1, patch_x[1]-x1).reshape(-1, 1)- patch_size
    weight_mask_y = np.arange(patch_y[0]-y1, patch_y[1]-y1).reshape(-1, 1)- patch_size
    weight_mask = np.meshgrid(weight_mask_x, weight_mask_y)
    weight_mask = np.stack((weight_mask[0], weight_mask[1]), axis = -1)
    weight_mask = np.linalg.norm(weight_mask, axis = 2)
    weight_mask = np.exp(-(0.5*weight_mask/patch_size)**2)/(patch_size*np.sqrt(2*np.pi))

    # Reminder: need to invert axis ordering because (row, col) -> (y, x) not (x, y)
    im1_patch = im1[patch_y[0]:patch_y[1], patch_x[0]:patch_x[1],:]

    min_dist = np.inf
    best_xy = np.zeros(2)
    for x2,y2 in zip(X2s,Y2s):
        patch2_x = patch_x-x1+x2
        patch2_y = patch_y-y1+y2
        x2_patch = np.linspace(patch2_x[0], patch2_x[1], round(patch_x[1]-patch_x[0]))
        y2_patch = np.linspace(patch2_y[0], patch2_y[1], round(patch_y[1]-patch_y[0]))
        x2_patch, y2_patch = np.meshgrid(x2_patch, y2_patch)

        # Mask to remove values that are from out of bounds.
        boundary_mask = (x2_patch>0) & (x2_patch<width) & (y2_patch>0) & (y2_patch<height)
        boundary_mask = np.expand_dims(boundary_mask, axis = -1)

        # Reminder: need to invert axis ordering because (row, col) -> (y, x) not (x, y)
        im2_patch = np.zeros((patch_height, patch_width, 3))
        im2_patch[:, :, 0] = im2R_spline.ev(y2_patch, x2_patch)
        im2_patch[:, :, 1] = im2G_spline.ev(y2_patch, x2_patch)
        im2_patch[:, :, 2] = im2B_spline.ev(y2_patch, x2_patch)
        im2_patch = np.multiply(im2_patch,boundary_mask)

        dist = np.linalg.norm(im2_patch - im1_patch, axis = 2)*weight_mask
        dist = np.sum(dist)
        if dist < min_dist:
            min_dist = dist
            best_xy = np.array([x2, y2])
    
    return best_xy[0], best_xy[1]

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    N = pts1.shape[0]
    pts1_homogenous = np.hstack((pts1, np.ones((N,1))))
    pts2_homogenous = np.hstack((pts2, np.ones((N,1))))

    best_score = 0
    best_F = []
    best_inliers = []

    for i in range(nIters):
        print(f'Iteration {i}')
        temp_ids = gen.choice(np.arange(N), 8) # Choose 8 random indicies.   

        F = eightpoint(pts1[temp_ids, :], pts2[temp_ids, :], M)
        
        dists = np.zeros((N,1))
        for j in range(N):
            l2 = F@pts1_homogenous[j,:].T
            dists[j,0] = np.abs(pts2_homogenous[j, :]@l2)/np.linalg.norm(l2[:2])

        inliers = np.where(dists < tol, True, False)
        score = np.sum(inliers)

        if score > best_score:
            print('######################################################')
            best_score = score
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers
            

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    """
    Implemented based off of the following pdf
    https://courses.cs.duke.edu/cps274/fall13/notes/rodrigues.pdf
    """
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        u = r/theta
        u = u.reshape((3,1))
        K = np.array([[0, -u[2, 0], u[1, 0]],
                      [u[2, 0], 0, -u[0, 0]],
                      [-u[1,0], u[0,0], 0]])
        R = np.eye(3)*np.cos(theta) + (1-np.cos(theta))*u@u.T+np.sin(theta)*K
        return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R-R.T)/2
    rho = np.array([A[2,1], A[0, 2], A[1,0]]).reshape((3,1))
    s = np.linalg.norm(rho)
    c = (np.trace(R)-1)/2

    if s == 0 and c == 1:
        r = np.zeros((3,1))
    elif s == 0 and c == -1:
        RI = R+np.eye(3)
        for i in range(3):
            if np.linalg.norm(RI[:, i]) > 0:
                v = RI[:, i]
                break
        u = v/np.linalg.norm(v)
        r = np.pi*u
        cond1a = np.linalg.norm(r) == np.pi 
        cond1b = (r[0] == 0 and r[1] == 0 and r[2] < 0) or (r[0] == 0 and r[1] < 0) or (r[0] < 0)
        if cond1a and cond1b:
            r = -r
    else:
        u = rho/s
        theta = np.arctan2(s, c)
        r = u*theta
    
    r = r.reshape(3)
    return r
       
    # theta = np.arccos((np.trace(R)-1)/2)

    # # Axis of rotation. 
    # k = np.array([R[2, 1] - R[1, 2],
    #               R[0, 2] - R[2, 0],
    #               R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    
    # r = theta*k
    # return r


'''
Q5.3: Extra Credit Rodrigues residual.  
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    N = len(p1)
    W = x[:-6].reshape((N, 3))
    W_homogenous = np.hstack((W, np.ones((N,1))))
    r2 = x[-6:-3]
    R = rodrigues(r2)
    t2 = np.array(x[-3:]).reshape((3,1))
    M2 = np.hstack((R, t2))
    C1 = K1@M1
    C2 = K2@M2 
    p1_hat = (C1@W_homogenous.T).T
    p1_hat = p1_hat[:, :2]/p1_hat[:,2].reshape((-1,1))
    p2_hat = (C2@W_homogenous.T).T
    p2_hat = p2_hat[:, :2]/p2_hat[:,2].reshape((-1,1))
    residual = np.concatenate([(p1-p1_hat).reshape([-1]),
                (p2-p2_hat).reshape([-1])])
    return residual

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    P_init = P_init.flatten()
    r2_init = invRodrigues(M2_init[:3, :3])
    t2_init = M2_init[:, 3]
    x_init = np.concatenate((P_init, r2_init, t2_init))
    func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    x_sol, _ = leastsq(func, x_init)

    r2 = x_sol[-6:-3]
    t2 = x_sol[-3:].reshape((3,1))
    P2 = x_sol[:-6].reshape((-1,3))
    M2 = np.hstack((rodrigues(r2), t2))
    return M2, P2

# if __name__ == '__main__':
#     corr_pts = np.load('..\data\some_corresp_noisy.npz')
#     pts1 = corr_pts['pts1']
#     pts2 = corr_pts['pts2']
#     img1 = cv2.imread('..\data\im1.png')
#     img2 = cv2.imread('..\data\im2.png')
#     M = max(img2.shape)

#     intrinsics = np.load('../data/intrinsics.npz')
#     K1 = intrinsics['K1']
#     K2 = intrinsics['K2']
   
#     M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
#     C1 = K1@M1

#     # F = eightpoint(pts1, pts2, M)
#     # helper.displayEpipolarF(img1, img2, F)

#     F_ransac, inliers = ransacF(pts1, pts2, M, nIters = 200, tol = 1.5)
#     print(f'Number of Inliers: {np.sum(inliers)}')
#     E = essentialMatrix(F_ransac, K1, K2)
#     M2s = helper.camera2(E)

#     # helper.displayEpipolarF(img1, img2, F_ransac)

#     min_error = np.inf
#     best_W = []
#     best_i = 0
#     for i in range(M2s.shape[2]):
#         C2 = K2@M2s[:, :, i]
#         W, err = triangulate(C1, pts1[inliers[:,0]], C2, pts2[inliers[:,0]])
#         if err <= min_error and min(W[:, -1]) >= 0:
#             min_error = err
#             best_W = W
#             best_i = i
#     print(f'Initial Error: {err}')

#     M2 = M2s[:, :, best_i]
#     # print(M2)
#     # print(invRodrigues(M2[:, :3]))
#     # print(rodrigues(invRodrigues(M2[:, :3])))

#     M2_final, W_final = bundleAdjustment(K1, M1, pts1[inliers[:,0]], K2, M2, pts2[inliers[:,0]], best_W)
    
#     C2 = K2@M2_final
#     W, err = triangulate(C1, pts1[inliers[:,0]], C2, pts2[inliers[:,0]])
#     print(f'Post optimization Error: {err}')

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(best_W[:, 0], best_W[:, 1], best_W[:, 2])

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Pre-Optimization')

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(W_final[:, 0], W_final[:, 1], W_final[:, 2])

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Post-Optimization')
    
#     plt.show()

    # nIters = [50, 100, 200, 400]
    # tol = [0.5, 1, 2]

    # inliers_list = []
    # for n in nIters:
    #     for t in tol:
    #         F, inliers = ransacF(pts1, pts2, M, n, t)
    #         inliers_list.append(inliers)

    # print('nIters\ttol\tscore')
    # print('----------------------------------------------------')
    # for i, n in enumerate(nIters):
    #     for j, t in enumerate(tol):
    #         k = i*len(tol)+j
    #         score = np.sum(inliers_list[k])
    #         print(f'{n}\t{t}\t{score}') 