# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    U, S, Vh = np.linalg.svd(I, full_matrices = False)

    B = np.diag(np.sqrt(S[:3]))@Vh[:3, :]
    L = np.diag(np.sqrt(S[:3]))@U[:, :3].T
    # B = Vh[:3, :]
    # L = np.diag(S[:3]) @ U[:, :3].T

    return B, L


if __name__ == "__main__":

    # Put your main code here
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    

    ##########
    # # 2c
    # print('L0\n', L0)
    # print('L\n', np.round(L, 4))
    # albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    # print(albedoIm)
    ##########
    # # 2d
    # albedos, normals = estimateAlbedosNormals(B)
    # surface = estimateShape(normals, s)
    # plotSurface(surface)
    ##########
    # # 2e
    # Nt = enforceIntegrability(B, s)
    # albedos, normals = estimateAlbedosNormals(Nt)
    # surface = estimateShape(normals, s)
    # plotSurface(surface)
    ##########
    # 2f
    Nt = enforceIntegrability(B, s)
    lamb = 1
    u = 10
    v = 0
    
    for _ in range(1):
        # G = np.diag([1,1,1])
        # G[2] = np.array([u, v, lamb])
        G_inv = np.diag([lamb, lamb, 1]).astype(float)
        G_inv[2] = [-u, -v, 1]
        G_inv /= float(lamb)
        
        Nt2 = G_inv.T @ Nt
        albedos, normals = estimateAlbedosNormals(Nt2)
        surface = estimateShape(normals, s)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        x = np.arange(surface.shape[1])
        y = np.arange(surface.shape[0])
        x, y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_title(r'$\mu = $ {mu}, $v$ = {v}, $\lambda$ = {lamb}'.format(mu = u, v = v, lamb = lamb))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot_surface(x, y, surface, cmap = 'coolwarm')
        plt.show()


    pass
