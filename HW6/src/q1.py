# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # image = None
    # L = rho * I * n \dot s

    # Camera Resolution will be something like "1080 x 720"
    # In otherwords we need that many pixels
    x = np.arange(0, res[0]).astype(int)
    y = np.arange(0, res[1]).astype(int)
    x, y= np.meshgrid(x,y)
    x -= res[0]//2
    y -= res[1]//2
    y *= -1
    # Flip the y coordinates so +y is going up

    # dist_from_center = np.linalg.norm(np.stack([center[0]-x*pxSize, center[1] -y*pxSize]), axis = 0)
    # Get the normal vectors:
    nx = x*pxSize-center[0]
    ny = y*pxSize-center[1]
    nz = (rad - np.sqrt(nx**2 + ny**2))
    n = np.stack([nx, ny, nz], axis = -1)
    n /= np.linalg.norm(n, axis = 2)[:, :, np.newaxis]
    mask = (nz >= 0).astype(float)[:, :, np.newaxis] # Mask for where the hemisphere is
    n *= mask


    rho = 1 # Albedo constant
    image = rho * np.sum((n*light), axis = 2)
    # image += np.abs(np.min(image))
    # image *= mask[:, :, 0]
    # image += np.abs(np.min(image)) # make sure the minimum value is 0.

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    from skimage import color

    all_images = []
    im_nums = [0, 1, 2, 3, 4, 5, 6]
    # im_nums = im_nums[3:]
    for i in im_nums:
        im_path = path + f'input_{i+1}.tif'
        rgb_image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        rgb_image = rgb_image.astype(float)
        rgb_image *= 255/(np.power(2, 16)-1)
        xyz_image = color.rgb2xyz(rgb_image)
        # intensity_vals = xyz_image[:, :, 1].flatten()
        # plt.hist(intensity_vals, bins = 50)
        # plt.show()
        all_images.append(xyz_image[:, :, 1].flatten())
        # im = plt.imread(im_path)
        # plt.imshow(im)
        # plt.show()
        
        # print('rgb_image:', rgb_image.shape, rgb_image.dtype)
        # print('xyz_image:', xyz_image.shape, xyz_image.dtype)
        # print(all_images[-1].shape)

    I = np.stack(all_images, axis = 0)
    L = np.load(path + 'sources.npy').T
    L = L[:, im_nums]
    s = rgb_image.shape[:-1]
    # print('I:', I.shape)
    # print('L:', L.shape, L.dtype)
    # print('s:', s)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # from scipy.sparse.linalg import lsqr
    # print(L.T.shape)
    B = np.linalg.lstsq(L.T, I, rcond = None)[0]
    # B = lsqr(L.T, I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    B = B.copy()
    albedos = np.linalg.norm(B, axis = 0)
    # if 0 in albedos:
    #     albedos += 1e-6
    normals = B/albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedos = albedos.copy()
    normals = normals.copy()
    albedoIm = albedos.reshape(s)
    albedoIm /= np.max(albedoIm)
    normalIm = normals.T.reshape(s[0], s[1], 3)
    normalIm += np.abs(np.min(normalIm, axis = (0,1)))
    normalIm /= np.max(normalIm, axis = (0, 1))

    fig, ax = plt.subplots(1)
    ax.imshow(albedoIm, cmap='gray')
    ax.set_title('Albedos plot')
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(1)
    ax.imshow(normalIm, cmap = 'rainbow')
    ax.set_title('Normals plot')
    plt.show()
    plt.close(fig)

    # print(normalIm[0, 0, :], normals[:, 0])
    # print(albedoIm.shape, normalIm.shape)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    normals = normals.copy()
    dfdx = -normals[0,:]/(normals[2,:])
    dfdy = -normals[1,:]/(normals[2,:])
    surface = integrateFrankot(dfdx.reshape(s), dfdy.reshape(s))
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    from mpl_toolkits.mplot3d import Axes3D
    x = np.arange(surface.shape[1])
    y = np.arange(surface.shape[0])
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x, y, surface, cmap = 'coolwarm')
    plt.show()

    pass


if __name__ == '__main__':

    ##########
    # # Question 1b
    # # Put your main code here
    center = np.array([0, 0, 0])
    rad = 0.75 # cm
    pxSize = 7e-4# um
    res = np.array([3840, 2160])

    for i, light in enumerate(np.array([[1,1,1], [1,-1, 1], [-1, -1, 1]]).astype(float)):
        image = renderNDotLSphere(center, rad, light/np.sqrt(3.), pxSize, res)
        fig,ax = plt.subplots(1)
        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'Light Source Vector: [{int(light[0])}, {int(light[1])}, {int(light[2])}]/'+ r'$\sqrt{3}$')
        fig.savefig(f'..\\results\\1_b{i}.png')
        plt.close(fig)
    ##########

    I, L, s = loadData()
    ##########
    # Question 1d
    # SingularValues= np.linalg.svd(I, compute_uv= False)
    # print("Singular values of I: ", SingularValues)
    ##########
    # Question 1e
    # B = estimatePseudonormalsCalibrated(I, L) # Is a 3 x p matrix
    # albedos, normals = estimateAlbedosNormals(B) 
    # print(albedos.shape, normals.shape)
    ##########
    # Question 1f
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    ##########
    # Question 1i
    # surface = estimateShape(normals, s)
    # plotSurface(surface)

    pass
