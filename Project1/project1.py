import em_help_functions as helper_fn
from em_help_functions import cam_intrinsics as get_camera_intrinsics
import cv2
import numpy as np
import scipy.stats as stats
from scipy.optimize import least_squares
import os
import time
import math

from utils import save_vanishing_points, save_assignments, save_rotation_matrix

PRIOR_DISTRIBUTION = helper_fn.P_m_prior
P_ANG = stats.norm(helper_fn.mu, helper_fn.sig)

base_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'data')
result_dir = os.path.join(base_dir, 'result')
image_filenames = ['P1030001.jpg', 'P1080055.jpg']
camera_intrinsics_filepath = os.path.join(data_dir, 'cameraParameters.mat')

N_ITERATIONS = 20
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VP_THRESHOLD = 1000


def read_image(image_filepath):
    image = cv2.imread(image_filepath)
    return image


def get_grayscale_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def get_image_gradients(grayscale_image):
    sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_dir = np.arctan2(sobel_y, sobel_x)
    return grad_mag, grad_dir


def compute_posterior(a, b, g, pixel_idxs, pixel_grads, camera_intrinsics):
    rotation_mtx = helper_fn.angle2matrix(a, b, g)
    posterior = 0

    for i, pixel_idx in enumerate(pixel_idxs):
        grad_dir = pixel_grads[i]
        loc = [pixel_idx[1], pixel_idx[0], 1]
        vp_thetas = helper_fn.vp2dir(camera_intrinsics, rotation_mtx, loc)

        pixel_posterior = 0
        for m, prior in enumerate(PRIOR_DISTRIBUTION):
            if m < 3:
                err = helper_fn.remove_polarity(grad_dir - vp_thetas[m])
                likelihood = P_ANG.pdf(err)
            else:
                likelihood = 1 / (2 * np.pi)
            pixel_posterior += likelihood * prior
        posterior += math.log(pixel_posterior)
    return posterior


def get_initial_rot(pixel_idxs, pixel_grads, camera_intrinsics):
    print("*** finding initial camera rotations ***")
    betas = np.linspace(-np.pi / 3, np.pi / 3, 60)
    posteriors = np.zeros_like(betas)
    for i, b in enumerate(betas):
        posteriors[i] = compute_posterior(0, b, 0, pixel_idxs, pixel_grads, camera_intrinsics)
    opt_idx = np.argmax(posteriors).item()
    opt_b = betas[opt_idx]

    search_range = [-1.0, 0, 1.0]
    angles = list()
    posteriors = list()
    for i in range(len(search_range)):
        b = opt_b + np.pi / 90 * search_range[i]
        for j in range(len(search_range)):
            a = np.pi / 36 * search_range[j]
            for k in range(len(search_range)):
                g = np.pi / 36 * search_range[k]
                angles.append([a, b, g])
                posteriors.append(compute_posterior(a, b, g, pixel_idxs, pixel_grads, camera_intrinsics))
    opt_idx = np.argmax(posteriors).item()
    return angles[opt_idx]


def get_vp(rotation_mtx, camera_instrinsics):
    vp = np.matmul(camera_instrinsics, np.matmul(rotation_mtx, helper_fn.vp_dir))
    return vp


def error_fn(rotation_vec, w_pm, pixel_idxs, pixel_grads, camera_intrinsics):
    residuals = list()
    rotation_mtx = helper_fn.vector2matrix(rotation_vec)
    for i, pixel_idx in enumerate(pixel_idxs):
        loc = [pixel_idx[1], pixel_idx[0], 1]
        pixel_grad = pixel_grads[i]
        thetas = helper_fn.vp2dir(camera_intrinsics, rotation_mtx, loc)
        pixel_wpm = w_pm[i]
        for j, wpm in enumerate(pixel_wpm[:3]):
            residuals.append(math.sqrt(wpm) * (helper_fn.remove_polarity(pixel_grad - thetas[j])))
    return np.array(residuals)


def e_step(rotation_vec, pixel_idxs, pixel_grads, camera_intrinsics):
    """
    :param rotation_vec : the Cayley-Gibbs-Rodrigu representation of camera rotation parameters
    :return: w_pm
    """
    rotation_mtx = helper_fn.vector2matrix(rotation_vec)
    w_pm = np.zeros([len(pixel_idxs), 4], dtype=np.float32)

    for i, pixel_idx in enumerate(pixel_idxs):
        pixel_grad = pixel_grads[i]
        loc = [pixel_idx[1], pixel_idx[0], 1]
        vp_thetas = helper_fn.vp2dir(camera_intrinsics, rotation_mtx, loc)

        for m, prior in enumerate(PRIOR_DISTRIBUTION):
            if m < 3:
                err = helper_fn.remove_polarity(pixel_grad - vp_thetas[m])
                likelihood = P_ANG.pdf(err)
            else:
                likelihood = 1 / (2 * np.pi)
            w_pm[i, m] = likelihood * prior

        # normalize
        total_prob = np.sum(w_pm[i, :])
        w_pm[i, :] /= total_prob
    return w_pm


def m_step(rotation_vec, w_pm, pixel_idxs, pixel_grads, camera_intrinsics):
    """
    :param rotation_vec: the camera rotation parameters from the previous step
    :param pixel_idxs: weights from E-step
    :param pixel_grads:
    :param camera_intrinsics:
    :return:
    """
    rotation_vec = least_squares(error_fn, rotation_vec, args=(w_pm, pixel_idxs, pixel_grads, camera_intrinsics))
    return rotation_vec


def process_image(image_filename):
    image_filepath = os.path.join(data_dir, image_filename)
    camera_intrinsics = get_camera_intrinsics(camera_intrinsics_filepath)
    image = read_image(image_filepath)
    grayscale_image = get_grayscale_image(image)

    grad_mags, grad_dirs = get_image_gradients(grayscale_image)
    grad_dirs, idxs = helper_fn.down_sample(grad_mags, grad_dirs)

    # serach for initial vp
    pixel_grads = list()
    for i, j in idxs:
        pixel_grads.append(grad_dirs[i, j])
    pixel_grads = np.array(pixel_grads)
    pixel_idxs = np.reshape(idxs, newshape=(-1, 2)) * 5 + 4
    initial_euler = get_initial_rot(pixel_idxs, pixel_grads, camera_intrinsics)

    # save initial vanishing points
    rotation_mtx = helper_fn.angle2matrix(*initial_euler)
    initial_vp = get_vp(rotation_mtx, camera_intrinsics)
    save_vanishing_points(image, initial_vp, image_filename, os.path.join(result_dir, 'initial-vp'))
    save_rotation_matrix(rotation_mtx, image_filename.split('.')[0], os.path.join(result_dir, 'initial-vp'))

    print("*** executing EM optimization ***")
    rotation_vec = helper_fn.matrix2vector(rotation_mtx)
    w_pm = None
    for i in range(N_ITERATIONS):
        start_t = time.time()
        w_pm = e_step(rotation_vec, pixel_idxs, pixel_grads, camera_intrinsics)
        opt = m_step(rotation_vec, w_pm, pixel_idxs, pixel_grads, camera_intrinsics)
        rotation_vec = opt.x
        print('iter {}: {}'.format(i, time.time() - start_t))
    rotation_mtx = helper_fn.vector2matrix(rotation_vec)
    final_vp = get_vp(rotation_mtx, camera_intrinsics)
    save_vanishing_points(image, final_vp, image_filename, os.path.join(result_dir, 'final-vp'))
    save_rotation_matrix(rotation_mtx, image_filename.split('.')[0], os.path.join(result_dir, 'final-vp'))
    assigned_image = save_assignments(image, w_pm, pixel_idxs, image_filename, os.path.join(result_dir, 'assignment'))
    save_vanishing_points(assigned_image, final_vp, image_filename, os.path.join(result_dir, 'final-assign-vp'))


def main():
    for image_filename in image_filenames:
        process_image(image_filename)


if __name__ == '__main__':
    main()