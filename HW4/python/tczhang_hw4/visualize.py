'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import *
import helper
import matplotlib.pyplot as plt

if __name__ == '__main__':
    corr_pts = np.load('..\data\some_corresp.npz')
    pts1 = corr_pts['pts1']
    pts2 = corr_pts['pts2']
    img1 = cv2.imread('..\data\im1.png')
    img2 = cv2.imread('..\data\im2.png')
    M = max(img2.shape)
    F = eightpoint(pts1, pts2, M)
    
    # helper.displayEpipolarF(img1, img2, F)

    temple_coord = np.load('../data/templeCoords.npz')
    x1 = temple_coord['x1'][:,0]
    y1 = temple_coord['y1'][:,0]

    x2 = np.zeros(x1.shape[0])
    y2 = np.zeros(y1.shape[0])
    for i in range(x1.shape[0]):
        x2[i], y2[i] = epipolarCorrespondence(img1, img2, F, x1[i], y1[i])

    # helper.epipolarMatchGUI(img1, img2, F)

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    E = essentialMatrix(F, K1, K2)

    M2s = helper.camera2(E)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = K1@M1
    min_error = np.inf
    best_W = []
    best_i = 0
    
    for i in range(M2s.shape[2]):
        C2 = K2@M2s[:, :, i]
        W, err = triangulate(C1, np.stack([x1, y1], axis = -1), C2, np.stack([x2, y2], axis = -1))
        # Make sure z-coordinate is positive
        if err <= min_error and min(W[:, -1]) >= 0:
            min_error = err
            best_W = W
            best_i = i
    print(min_error)
    M2 = M2s[:, :, best_i]
    C2 = K2@M2

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(best_W[:, 0], best_W[:, 1], best_W[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    np.savez('../results/q4_2.npz', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2)
    plt.show()