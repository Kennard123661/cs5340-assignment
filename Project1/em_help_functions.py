import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import least_squares
from time import time


vp_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
P_m_prior = [0.13, 0.24, 0.13, 0.5]
sig = 0.5
mu = 0.0


def cam_intrinsics(path):
    '''
    :param path: path where you store the camera parameters
    :return: camera intrinsic matrix K
    '''
    cam_data = scipy.io.loadmat(path)
    f = cam_data['focal']
    pixelSize = cam_data['pixelSize']
    pp = cam_data['pp']
    K = np.array([[f[0][0] / pixelSize[0][0], 0, pp[0][0]], [0, f[0][0] / pixelSize[0][0], pp[0][1]], [0, 0, 1]],
                 dtype=np.float32)
    return K


def remove_polarity(x):
    '''
    :param x:  the angle differences between the predicted normal direction and the gradient direction of a pixel.
               x is in shape [3,] which represent the normal direction with respect to the three edge models.
    :return: the minimal value after add pi and -pi
    '''
    x = np.expand_dims(x, axis=0)
    new = np.abs(np.concatenate([x, x + np.pi, x - np.pi], axis= 0))
    diff = np.min(new, axis=0)
    return diff


def angle2matrix(a, b, g):
    '''

    :param a: the rotation angle around z axis
    :param b: the rotation angle around y axis
    :param g: the rotation angle around x axis
    :return: rotation matrix
    '''

    R = np.array([[np.cos(a)*np.cos(b), -np.sin(a)*np.cos(g)+np.cos(a)*np.sin(b)*np.sin(g),  np.sin(a)*np.sin(g)+np.cos(a)*np.sin(b)*np.cos(g), 0],
                  [np.sin(a)*np.cos(b),  np.cos(a)*np.cos(g)+np.sin(a)*np.sin(b)*np.sin(g), -np.cos(a)*np.sin(g)+np.sin(a)*np.sin(b)*np.cos(g), 0],
                  [-np.sin(b) ,         -np.cos(b)*np.sin(g),                                np.cos(b)*np.cos(g),                               0]], dtype=np.float32)

    return R

def vector2matrix(S):

    '''
    :param S: the Cayley-Gibbs-Rodrigu representation
    :return: rotation matrix R
    '''
    S = np.expand_dims(S, axis=1)
    den = 1 + np.dot(S.T, S)
    num = (1 - np.dot(S.T, S))*(np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
    R = num/den
    homo = np.zeros([3,1], dtype=np.float32)
    R = np.hstack([R, homo])
    return R

def skew(a):
    s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
    return s

def matrix2quaternion(T):

    R = T[:3, :3]

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def matrix2vector(R):
    '''
    :param R: the camera rotation marix
    :return:  the Cayley-Gibbs-Rodrigu representation
    '''
    Q = matrix2quaternion(R)
    S = Q[1:]/Q[0]
    return S


def vp2dir(K, R, u):
    '''
    :param K: camera intrinsic matrix
    :param R: camera rotation matrix
    :param u: pixel location represented in homogeneous coordinate [x, y, 1]
    :return: the estimated normal direction for edge that pass through pixel u
    '''
    vp_trans = K.dot(R).dot(vp_dir)
    # edges = np.cross(vp_trans, u)
    # thetas_es = np.arctan2(edges[1], edges[0])
    vp_trans = K.dot(R).dot(vp_dir)
    edges = np.cross(vp_trans.transpose(), u)
    thetas_es = np.arctan2(edges[:, 1], edges[:, 0])
    return thetas_es

def down_sample(Gmag_, Gdir_):
    '''
    :param Gmag_: gradient magtitude of the original image
    :param Gir_: gradient direction of the original image
    :return: pixels we will use in the EM algorithm and the corresponding gradient direction
    '''
    Gmag = Gmag_[4::5, 4::5]
    Gdir = Gdir_[4::5, 4::5]
    threshold = np.sort(np.reshape(Gmag, [Gmag.shape[0]*Gmag.shape[1]]))
    idx = np.argwhere(Gmag > threshold[-2001])
    return Gdir, idx


def pro_mixture(a, b, g):
    '''
    :param a, b, g_: camera rotation parameters
    :return: p_image
    '''

    R = angle2matrix(a, b, g) # Convert the angles into rotation matrix
    p_image = 0.0   # initial posterior setting to zero

    # to be implemented, to compute the posterior of the gaussian mixture model, which is equation (6) in the manhattan paper

    return p_image


def E_step(S):
    '''
    :param S : the Cayley-Gibbs-Rodrigu representation of camera rotation parameters
    :return: w_pm
    '''
    R = vector2matrix(S)  # Note that the 'S' is just for optimization, it has to be converted to R during computation
    w_pm = np.zeros([idx.shape[0], 4], dtype=np.float32)

    # to be implemented, the E-step to compute the weights for each vanishing point at each pixel

    return w_pm

def M_step(S0, w_pm):
    '''
    :param S0 : the camera rotation parameters from the previous step
    :param w_pm : weights from E-step
    :return: R_m : the optimized camera rotation matrix
    '''
    S_m = least_squares(error_fun, S0, args= (w_pm,))

    return S_m

def error_fun(S, w_pm):

    '''
    :param S : the variable we are going to optimize over
    :param w_pm : weights from E-step
    :return: error : the error we are going to minimize
    '''

    error = 0.0    # initial error setting to zero
    R = vector2matrix(S) # Note that the 'S' is just for optimization, it has to be converted to R during computation

    # to be implemented, the error function to be minimized by the M-setp.


    return error
#
# img = cv2.imread('path')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0)
# sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1)
# Gmag_ = np.sqrt(sobelx**2.0 + sobely**2.0)
# Gdir_ = np.arctan2(sobely, sobelx)
# Gdir, idx = down_sample(Gmag_, Gdir_)
#
#
# beta = np.linspace(-np.pi/3, np.pi/3, 60)
# P = np.zeros_like(beta)
#
# for k in range(beta.shape[0]):
#     a = 0.0
#     b = beta[k]
#     g = 0.0
#     P[k] = pro_mixture(a, b, g)
# idx_coarse = np.argsort(P)
# b_opt = beta[idx_coarse[-1]]
#
# search_range = [-1.0, 0.0, 1.0]
# R_list = []
# P_list = np.zeros([len(search_range)**3,])
# c = 0
# for i in range(len(search_range)):
#     b = b_opt + (np.pi)/90 * search_range[i]
#     for j in range(len(search_range)):
#         a = (np.pi)/36 * search_range[j]
#         for k in range(len(search_range)):
#             g = (np.pi) / 36 * search_range[k]
#             R_list.append([a, b, g])
#             P_list[c] = pro_mixture(a, b, g)
#             c += 1
# idx_fine1 = np.argsort(P_list)
# R_opt = R_list[idx_fine1[-1]]
#
#
#
# search_range = [-2.0, -1.0, 0.0, 1.0, 2.0]
# R_list = []
# P_list = np.zeros([len(search_range)**2,])
# c = 0
# b = R_opt[1]
# for i in range(len(search_range)):
#     a = R_opt[0] + np.pi/36 * search_range[i]
#     for j in range(len(search_range)):
#         g = R_opt[2] + np.pi/36 * search_range[j]
#         R_list.append([a, b, g])
#         P_list[c] = pro_mixture(a, b, g)
#         c += 1
# idx_fine2 = np.argsort(P_list)
# R_opt = R_list[idx_fine2[-1]]
#
#
#
# num_iter = 20
# R = angle2matrix(R_opt[0], R_opt[1], R_opt[2])
# S = matrix2vector(R)
# for i in range(num_iter):
#     t = time()
#     w_pm = E_step(S)
#     opt = M_step(S, w_pm)
#     S = opt.x
#     print('iter {}: {}'.format(i, time()-t))
# R_em = vector2matrix(S)
#
