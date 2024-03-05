'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
from submission import *
import helper

if __name__ == '__main__':
    corr_pts = np.load('..\data\some_corresp.npz')
    pts1 = corr_pts['pts1']
    pts2 = corr_pts['pts2']
    img1 = cv2.imread('..\data\im1.png')
    img2 = cv2.imread('..\data\im2.png')
    M = max(img2.shape)
    F = eightpoint(pts1, pts2, M)
    
    # helper.displayEpipolarF(img1, img2, F)

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    np.savez('../results/q3_1.npz', E = E)
    print('SAVED!')

    Ms = helper.camera2(E)
    for i in range(4):
        print(f'M2 #{i}:\n {Ms[:, :, i]}')

    C1 = np.hstack((K1, np.zeros((3,1))))
    min_error = np.inf
    best_W = []
    best_i = 0
    best_C2 = []
    for i in range(Ms.shape[2]):
        C2 = K2@Ms[:, :, i]
        W, err = triangulate(C1, pts1, C2, pts2)
        # print('#################################')
        # Make sure z-coordinate is positive
        if err <= min_error and min(W[:, -1]) >= 0:
            min_error = err
            best_W = W
            best_i = i
    print(f'Minimum Error: {min_error}')
    M2 = Ms[:, :, best_i]
    C2 = K2@M2
    print(f'Best M2 matrix: \n{M2}')
    np.savez('../results/q3_3.npz', M2=M2, C2=C2, P = best_W) 
